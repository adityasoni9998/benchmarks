"""Apptainer workspaces with a disposable directory overlay on local scratch.

Ported verbatim from hyperion (platoon/openhands/apptainer.py, commit fac6a27) so SWE-bench evals
spawn Apptainer sandboxes exactly like hyperion training.
"""

import os
import random
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from benchmarks.utils.apptainer_namespace import (
    NAMESPACE_ENV,
    namespace_command,
    shared_apptainer_namespace,
)
from openhands.sdk import get_logger
from openhands.workspace import ApptainerWorkspace


logger = get_logger(__name__)


class DirectoryOverlayWorkspace(ApptainerWorkspace):
    """Keep writes on disk and coordinate server ports across rollout processes."""

    scratch_dir: str
    server_command: list[str] = Field(default_factory=list)
    # Host names resolved to an unroutable address inside the container (opt-in; loopback is unaffected).
    blocked_hosts: list[str] = Field(default_factory=list)
    cleaned_up: bool = Field(default=False, exclude=True)
    _runtime_dir: Path | None = PrivateAttr(default=None)
    _port_claim: socket.socket | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        try:
            if self.extra_ports:
                raise ValueError(
                    "DirectoryOverlayWorkspace supports only the agent-server port"
                )
            self._claim_port()
            super().model_post_init(context)
        except BaseException:
            # A failed constructor never returns a workspace to the rollout.
            self.cleanup()
            raise

    def _claim_port(self) -> None:
        """Hold an abstract Unix socket as a port claim shared by all our launchers.

        Claims coordinate workspace processes; the TCP probe also rejects ports
        occupied by other services. The claim socket is never used for I/O.
        """
        requested_port = self.host_port
        rng = random.SystemRandom()
        for _ in range(5):
            port = (
                requested_port
                if requested_port is not None
                else rng.randint(10000, 65535)
            )
            claim = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                claim.bind(f"\0platoon-apptainer-port-{port}")
                # Do not enable SO_REUSEADDR: an existing listener or bind
                # must make this candidate unavailable.
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("0.0.0.0", port))
            except BaseException as exc:
                claim.close()
                if requested_port is None and isinstance(exc, Exception):
                    continue
                raise
            self.host_port = port
            self._port_claim = claim
            return
        raise RuntimeError(
            "Could not claim an available Apptainer port after 5 attempts"
        )

    def _start_container(self) -> None:
        """Start the agent server with a disposable directory overlay."""
        self._runtime_dir = Path(
            tempfile.mkdtemp(prefix="platoon-apptainer-", dir=self.scratch_dir)
        )
        overlay_root_dir = self._runtime_dir / "overlay"
        for name in ("overlay/upper", "overlay/work", "workdir", "runtime-tmp"):
            (self._runtime_dir / name).mkdir(parents=True)

        # Prepare environment variables
        env_args: list[str] = []
        for key in self.forward_env:
            if key in os.environ:
                env_args += ["--env", f"{key}={os.environ[key]}"]

        # Prepare bind mounts
        # Prebuilt SIFs can contain an empty resolver file.
        bind_args: list[str] = ["--bind", "/etc/resolv.conf:/etc/resolv.conf:ro"]
        if self.blocked_hosts:
            try:
                base_hosts = Path("/etc/hosts").read_text()
            except OSError:
                base_hosts = "127.0.0.1 localhost\n::1 localhost\n"
            blocked = "".join(f"0.0.0.0 {h}\n:: {h}\n" for h in self.blocked_hosts)
            hosts_file = self._runtime_dir / "hosts"
            hosts_file.write_text(base_hosts.rstrip("\n") + "\n" + blocked)
            bind_args += ["--bind", f"{hosts_file}:/etc/hosts:ro"]
        if self.mount_dir:
            mount_path = "/workspace"
            bind_args += ["--bind", f"{self.mount_dir}:{mount_path}"]
            logger.info(
                "Mounting host dir %s to container path %s",
                self.mount_dir,
                mount_path,
            )
        env_extra_binds = [
            item.strip()
            for item in os.getenv("OPENHANDS_APPTAINER_EXTRA_BINDS", "").split(",")
            if item.strip()
        ]
        for bind_spec in [*self.extra_bind_mounts, *env_extra_binds]:
            bind_args += ["--bind", bind_spec]
            logger.info("Adding Apptainer bind mount: %s", bind_spec)

        # Build container options
        namespace = os.environ.get(NAMESPACE_ENV) if self.use_fakeroot else None
        container_opts: list[str] = []
        if self.use_fakeroot and not namespace:
            container_opts.append("--fakeroot")
        # Replace --compat's writable-tmpfs with a directory overlay. This
        # workspace always uses the overlay, including when streaming logs.
        container_opts.append("--containall")
        container_opts.append("--no-eval")
        container_opts.append("--no-init")
        container_opts.append("--no-umask")
        container_opts += ["--overlay", str(overlay_root_dir)]
        container_opts += ["--workdir", str(self._runtime_dir / "workdir")]
        # The overlay is discarded at cleanup, so bytecode caches are never reused. Writing
        # them (~4,000 files per agent-server start) contends on scratch in start bursts.
        container_opts += ["--env", "PYTHONDONTWRITEBYTECODE=1"]
        if self.enable_gpu:
            container_opts.append("--nv")
        if self.disable_mount_locations:
            for loc in self.disable_mount_locations:
                container_opts += ["--no-mount", loc]

        server_cmd = [
            "apptainer",
            "exec" if self.server_command else "run",
            *container_opts,
            *env_args,
            *bind_args,
            self._sif_path,
            *self.server_command,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.host_port),
        ]
        if namespace:
            server_cmd = namespace_command(namespace, server_cmd)

        # Keep runtime temporary files on scratch without launcher configuration.
        env = os.environ.copy()
        assert self.cache_dir is not None
        env["APPTAINER_CACHEDIR"] = self.cache_dir
        env["APPTAINER_TMPDIR"] = str(self._runtime_dir / "runtime-tmp")
        env["TMPDIR"] = env["APPTAINER_TMPDIR"]

        self._process = subprocess.Popen(
            server_cmd,
            env=env,
            # Do not leave an unread PIPE when log streaming is disabled.
            stdout=subprocess.PIPE if self.detach_logs else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        # Optionally stream logs in background
        if self.detach_logs:
            self._logs_thread = threading.Thread(target=self._stream_logs, daemon=True)
            self._logs_thread.start()

    def cleanup(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        try:
            super().cleanup()
        finally:
            if self._port_claim is not None:
                self._port_claim.close()
                self._port_claim = None
            if self._runtime_dir is not None:
                try:
                    # Files created as container users belong to subordinate host
                    # IDs. Delete with the same mappings after stopping the sandbox.
                    with shared_apptainer_namespace(enabled=self.use_fakeroot):
                        namespace = (
                            os.environ.get(NAMESPACE_ENV) if self.use_fakeroot else None
                        )
                        if namespace:
                            subprocess.run(
                                namespace_command(
                                    namespace,
                                    ["/bin/rm", "-rf", "--", str(self._runtime_dir)],
                                ),
                                capture_output=True,
                                text=True,
                                check=True,
                                timeout=60,
                            )
                        else:
                            shutil.rmtree(self._runtime_dir)
                except Exception as error:
                    # Cleanup failures must not lose the rollout, but report the
                    # retained path so disk leaks are visible.
                    logger.warning(
                        "Could not remove Apptainer runtime directory %s: %s",
                        self._runtime_dir,
                        error,
                    )
                else:
                    self._runtime_dir = None
