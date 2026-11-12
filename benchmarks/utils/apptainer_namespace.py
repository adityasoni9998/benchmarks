"""Share fakeroot ID mappings across local rollout processes.

The trainer keeps a user-namespace descriptor open for the run. Each container
gets a private child namespace with the same IDs, without rerunning the host's
potentially slow newuidmap/newgidmap helpers. No network namespace is created.

This file also runs as a small standalone launcher, without importing the SDK.
The approach follows hpc-sandbox's pods.id_namespace and holder._nest.

Ported verbatim from hyperion (platoon/openhands/apptainer_namespace.py, commit fac6a27) so SWE-bench evals
spawn Apptainer sandboxes exactly like hyperion training.
"""

import logging
import os
import pwd
import select
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


NAMESPACE_ENV = "PLATOON_APPTAINER_USER_NAMESPACE"
logger = logging.getLogger(__name__)


def _subordinate_range(path: str) -> tuple[int, int]:
    uid = os.getuid()
    owners = {str(uid), pwd.getpwuid(uid).pw_name}
    for line in Path(path).read_text().splitlines():
        fields = line.split(":")
        if len(fields) == 3 and fields[0] in owners:
            start, count = map(int, fields[1:])
            if count > 0:
                return start, count
    raise RuntimeError(f"No subordinate ID range for this user in {path}")


def _create_namespace() -> int:
    ranges = [
        (_subordinate_range(path), tool, own)
        for path, tool, own in (
            ("/etc/subuid", "newuidmap", os.getuid()),
            ("/etc/subgid", "newgidmap", os.getgid()),
        )
    ]
    # Closing stdin releases the holder, even if setup fails or the trainer dies.
    with subprocess.Popen(
        [sys.executable, "-I", "-S", __file__, "hold"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as holder:
        assert holder.stdin is not None and holder.stdout is not None
        try:
            readable, _, _ = select.select([holder.stdout], [], [], 30)
            if not readable or holder.stdout.readline() != b"ready\n":
                raise RuntimeError(
                    "Could not create the shared Apptainer user namespace"
                )
            for (start, count), tool, own in ranges:
                result = subprocess.run(
                    [
                        tool,
                        str(holder.pid),
                        "0",
                        str(own),
                        "1",
                        "1",
                        str(start),
                        str(count),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode:
                    raise RuntimeError(f"{tool} failed: {result.stderr.strip()}")
            descriptor = os.open(
                f"/proc/{holder.pid}/ns/user", os.O_RDONLY | os.O_CLOEXEC
            )
        finally:
            holder.stdin.close()
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait()
    return descriptor


@contextmanager
def shared_apptainer_namespace(*, enabled: bool) -> Iterator[None]:
    """Own the mapping for a local training run, before starting rollout workers.

    Setup is synchronous and runs once at startup, before the training loop.
    Children inherit a /proc reference, not the descriptor itself, so exec'd
    rollout processes can use it too. Keep this context open until they exit.
    """
    if not enabled or os.environ.get(NAMESPACE_ENV):
        yield
        return
    logger.info(
        "Preparing shared Apptainer UID/GID mappings once for this training run"
    )
    descriptor = _create_namespace()
    previous = os.environ.get(NAMESPACE_ENV)
    try:
        os.environ[NAMESPACE_ENV] = f"/proc/{os.getpid()}/fd/{descriptor}"
        logger.info("Shared Apptainer UID/GID mappings ready")
        yield
    finally:
        if previous is None:
            os.environ.pop(NAMESPACE_ENV, None)
        else:
            os.environ[NAMESPACE_ENV] = previous
        os.close(descriptor)


def namespace_command(namespace: str, command: list[str]) -> list[str]:
    return [sys.executable, "-I", "-S", __file__, "enter", namespace, *command]


def _enter(namespace: str, command: list[str]) -> None:
    descriptor = os.open(namespace, os.O_RDONLY)
    try:
        os.setns(descriptor, os.CLONE_NEWUSER)
    finally:
        os.close(descriptor)
    # Capture the parent's mapping before unshare. A short-lived child stays in
    # that parent, where it may map all these IDs into our new private namespace.
    mappings = {}
    for name in ("uid_map", "gid_map"):
        extents = [
            line.split() for line in Path(f"/proc/self/{name}").read_text().splitlines()
        ]
        mappings[name] = "".join(
            f"{inside} {inside} {count}\n" for inside, _, count in extents
        )
    read_fd, write_fd = os.pipe()
    target = os.getpid()
    mapper = os.fork()
    if mapper == 0:
        os.close(write_fd)
        try:
            if os.read(read_fd, 1) != b"1":
                os._exit(1)
            for name, mapping in mappings.items():
                Path(f"/proc/{target}/{name}").write_text(mapping)
        except OSError as error:
            print(
                f"Could not map the Apptainer child namespace: {error}",
                file=sys.stderr,
                flush=True,
            )
            os._exit(1)
        os._exit(0)
    os.close(read_fd)
    try:
        os.unshare(os.CLONE_NEWUSER)
        os.write(write_fd, b"1")
    finally:
        os.close(write_fd)
        _, status = os.waitpid(mapper, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError("Could not map the Apptainer child namespace")
    # The namespace already supplies root and subordinate IDs. Asking Apptainer
    # for --fakeroot again would try to look up a new allocation for 'root'.
    os.execvp(command[0], command)


if __name__ == "__main__":
    if sys.argv[1] == "hold":
        os.unshare(os.CLONE_NEWUSER)
        print("ready", flush=True)
        sys.stdin.buffer.read(1)
    elif sys.argv[1] == "enter":
        _enter(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit("Expected hold or enter")
