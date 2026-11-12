import json
import os
import re
import tempfile
import time
import uuid
from typing import Any, List

from jinja2 import Environment, FileSystemLoader

from benchmarks.swebench import constants
from benchmarks.swebench.apptainer_build import ensure_apptainer_agent_image
from benchmarks.swebench.build_images import (
    extract_custom_tag,
    get_official_docker_image,
    should_wrap_instance_id,
    wrap_image,
)
from benchmarks.swebench.config import INFER_DEFAULTS
from benchmarks.utils.acp import (
    add_acp_agent_metadata,
    build_acp_agent,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
    workspace_keepalive,
)
from benchmarks.utils.apptainer_namespace import shared_apptainer_namespace
from benchmarks.utils.apptainer_overlay import DirectoryOverlayWorkspace
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.build_utils import ensure_local_image
from benchmarks.utils.console_logging import summarize_instance
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.image_utils import remote_image_exists
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
)
from benchmarks.utils.tool_presets import get_tools_for_preset
from benchmarks.utils.version import get_phased_image_tag_prefix
from openhands.sdk import Agent, Conversation, Tool, get_logger
from openhands.sdk.agent import ACPAgent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm.exceptions.classifier import (
    LONG_PROMPT_PATTERNS,
    MALFORMED_HISTORY_PATTERNS,
)
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.task import TaskToolSet
from openhands.workspace import APIRemoteWorkspace, DockerWorkspace


logger = get_logger(__name__)

# Apptainer sandboxes match hyperion training (platoon DirectoryOverlayWorkspace):
# prebuilt SIFs are read from <OPENHANDS_APPTAINER_SIF_DIR>/<instance_id>.sif, and each
# sandbox's disposable overlay lives under OPENHANDS_APPTAINER_SCRATCH_DIR (node-local).
APPTAINER_SIF_DIR_ENV = "OPENHANDS_APPTAINER_SIF_DIR"
APPTAINER_SCRATCH_DIR_ENV = "OPENHANDS_APPTAINER_SCRATCH_DIR"
# Sandbox start attempts per instance attempt (hyperion's NUM_RETRIES_SANDBOX_START).
APPTAINER_START_ATTEMPTS = 3
# When set to 1, a conversation that ends with an error caused by the agent ends the
# episode and its final patch is graded, as in hyperion training
# (plugins/openhands/platoon/openhands/env.py _run_conversation): genuine pass@1.
# Any other conversation error (sandbox or LLM-server failure, or an unrecognized
# message) is raised, so the instance is retried.
END_EPISODE_ON_AGENT_ERROR_ENV = "SWEBENCH_END_EPISODE_ON_AGENT_ERROR"


def _any_of(*phrases: str) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(p) for p in phrases), re.IGNORECASE)


# Agent-caused conversation errors, matched against the error message and the
# ConversationErrorEvent ("code: detail") in the conversation history. Context-window
# and malformed-history phrases include the SDK's own lists (llm/exceptions/classifier.py)
# and its error class name, whose default message is "Conversation history longer than
# LLM context window limit".
AGENT_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("turn_limit", _any_of("MaxIterationsReached", "maximum iterations limit")),
    ("stuck", _any_of("got stuck")),
    (
        "context_window",
        _any_of(
            "LLMContextWindowExceedError",
            "ContextWindowExceeded",
            "LLM context window limit",
            "maximum context length",
            "longer than the model's context length",
            *LONG_PROMPT_PATTERNS,
        ),
    ),
    ("timeout", _any_of("Run timed out after")),
    (
        "malformed_tool_call",
        _any_of(
            "function.arguments must be a JSON object",
            "LLMMalformedConversationHistoryError",
            *MALFORMED_HISTORY_PATTERNS,
        ),
    ),
)


def end_episode_on_agent_error() -> bool:
    return os.getenv(END_EPISODE_ON_AGENT_ERROR_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


def classify_agent_error(message: str) -> str | None:
    """Return the agent-caused error category of a conversation error, if any."""
    for category, pattern in AGENT_ERROR_PATTERNS:
        if pattern.search(message):
            return category
    return None


def conversation_error_detail(conversation: Any) -> str | None:
    """Return the history's last ConversationErrorEvent as "code: detail".

    The ERROR status can reach the client before its detailed event, so run()
    may raise only "Remote conversation ended with error". As hyperion does,
    refresh and reconcile the history first, then read the error event.
    """
    try:
        state = conversation.state
        refresh = getattr(state, "refresh_from_server", None)
        if callable(refresh):
            refresh()
        reconcile = getattr(state.events, "reconcile", None)
        if callable(reconcile):
            reconcile()
        for event in reversed(list(state.events)):
            if isinstance(event, ConversationErrorEvent):
                parts = (event.code.strip(), event.detail.strip())
                return ": ".join(part for part in parts if part) or None
    except Exception as error:
        logger.warning("Could not read the conversation error event: %s", error)
    return None


# After an agent-caused error, how long to wait for the server to stop the run.
STOP_WAIT_SECONDS = 300
STOP_POLL_SECONDS = 5


def _conversation_running(conversation: Any) -> bool:
    state = conversation.state.refresh_from_server()
    return str(state.get("execution_status", "")).lower() == "running"


def stop_conversation(conversation: Any, instance_id: str) -> bool:
    """Stop the run on the server before the final patch is captured.

    After a client-side timeout the server may still be running a step. interrupt()
    cancels the in-flight step immediately (the agent server runs the default Agent
    with arun(), so this takes effect mid-LLM-call), unlike pause(), which waits for
    the current step to finish. Wait until the server reports the run is no longer
    running. Returns whether that was confirmed: if the run is still running after
    STOP_WAIT_SECONDS the episode's patch is captured anyway (its outcome is kept);
    only an unreachable agent server, a crash, raises so the instance is retried.
    """
    try:
        conversation.interrupt()
    except Exception as error:
        logger.warning(
            "Could not interrupt conversation for %s: %s", instance_id, error
        )
    deadline = time.monotonic() + STOP_WAIT_SECONDS
    failures = 0
    while True:
        try:
            if not _conversation_running(conversation):
                return True
            failures = 0
        except Exception as error:
            failures += 1
            if failures >= 3:  # the agent server is gone: the sandbox crashed
                raise RuntimeError(
                    f"Cannot read run status for {instance_id}: {error}"
                ) from error
        if time.monotonic() >= deadline:
            logger.warning(
                "Instance %s: conversation still running %ds after interrupt; "
                "capturing its patch anyway",
                instance_id,
                STOP_WAIT_SECONDS,
            )
            return False
        time.sleep(STOP_POLL_SECONDS)


def workspace_responds(workspace: RemoteWorkspace) -> bool:
    """Whether the sandbox still runs commands (it did not crash)."""
    try:
        return workspace.execute_command("true", timeout=60).exit_code == 0
    except Exception:
        return False


# The SWE-bench image activates the testbed environment from /root/.bashrc, but
# Apptainer's --containall gives the agent server an empty home, so the terminal
# tool's interactive bash (it reads only ~/.bashrc) would start in conda base. Restore
# the activation, using the harness's lookup order for conda.sh (same as hyperion).
ACTIVATE_TESTBED = (
    "if [ -f /opt/conda/etc/profile.d/conda.sh ]; then "
    "source /opt/conda/etc/profile.d/conda.sh; "
    "elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then "
    "source /opt/miniconda3/etc/profile.d/conda.sh; fi\n"
    "conda activate testbed\n"
)


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance["repo"].split("/")[-1]
    assert metadata.details is not None

    # Set up Jinja2 environment
    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
    }
    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction


class SWEBenchEvaluation(Evaluation):
    """
    Process-based SWE-bench evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def get_official_docker_image(self, instance: EvalInstance) -> str:
        return get_official_docker_image(instance.id)

    def extract_custom_tag(self, official_docker_image: str) -> str:
        return extract_custom_tag(official_docker_image)

    def should_wrap_instance(self, instance: EvalInstance) -> bool:
        return should_wrap_instance_id(instance.id)

    def get_source_repo_path(self, instance: EvalInstance) -> str:
        return "/testbed"

    def get_repo_path(self, instance: EvalInstance) -> str:
        if self.metadata.workspace_type == "apptainer":
            # Apptainer sandboxes run with --fakeroot, so the agent can edit the
            # image's repository in place, as in hyperion training.
            return self.get_source_repo_path(instance)
        # Docker/remote agent servers run as a non-root user that owns only
        # /workspace, so they work on a writable copy of /testbed there.
        return f"/workspace/{instance.data['repo'].split('/')[-1]}/"

    def get_patch_base(self, instance: EvalInstance) -> str:
        """Return the commit the agent started from; patches are diffs against it.

        The SWE-bench images add a setup commit on top of base_commit (astropy's, for
        example, pins setuptools in pyproject.toml and makes every file 755). Grading
        applies the patch to the same image, so a diff against base_commit would carry
        that commit along and fail to apply. evaluate_instance records the image's HEAD
        before the agent runs; until then the agent has made no changes, so HEAD is it.
        """
        return instance.data.get("start_commit", "HEAD")

    def get_git_patch(
        self,
        instance: EvalInstance,
        workspace: RemoteWorkspace,
    ) -> str:
        repo_path = self.get_repo_path(instance)
        patch_base = self.get_patch_base(instance)
        git_patch_result = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager diff --no-color {patch_base} HEAD")
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        return git_patch_result.stdout

    def get_staged_git_patch(
        self,
        instance: EvalInstance,
        workspace: RemoteWorkspace,
    ) -> str:
        repo_path = self.get_repo_path(instance)
        patch_base = self.get_patch_base(instance)
        git_patch_result = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager diff --no-color --cached {patch_base}")
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        return git_patch_result.stdout

    def collect_failure_test_result(
        self,
        instance: EvalInstance,
        workspace: RemoteWorkspace,
        error: Exception,
    ) -> dict[str, Any]:
        repo_path = self.get_repo_path(instance)

        # Include newly-created files in the recovered diff. This intentionally
        # avoids committing so the failed workspace remains close to its final
        # agent-visible state.
        git_add = workspace.execute_command(f"cd {repo_path} ; git add -A")
        if git_add.exit_code != 0:
            logger.warning(
                "Failed to stage changes while collecting failure patch for %s: %s",
                instance.id,
                git_add.stderr,
            )

        try:
            git_patch = self.get_staged_git_patch(instance, workspace)
        except Exception as patch_error:
            logger.warning(
                "Failed to collect failure patch for %s after %s: %s",
                instance.id,
                type(error).__name__,
                patch_error,
            )
            return {}

        logger.info(
            "Collected failure patch for %s: %d chars",
            instance.id,
            len(git_patch),
        )
        return {
            "git_patch": git_patch,
            "git_patch_captured_on_error": True,
        }

    def get_apptainer_extra_bind_mounts(self) -> list[str]:
        """Return host paths that must be visible inside Apptainer agent servers."""
        bind_mounts: list[str] = []
        custom_tokenizer = self.metadata.llm.custom_tokenizer
        if custom_tokenizer:
            tokenizer_path = os.path.abspath(os.path.expanduser(custom_tokenizer))
            if os.path.exists(tokenizer_path):
                bind_mounts.append(f"{tokenizer_path}:{tokenizer_path}:ro")
            else:
                logger.warning(
                    "custom_tokenizer path %s does not exist on host; "
                    "not adding an Apptainer bind mount",
                    custom_tokenizer,
                )
        return bind_mounts

    def get_apptainer_mount_dir(self, instance: EvalInstance) -> str:
        """Return a writable host directory to bind onto /workspace."""
        workspace_root = os.getenv("OPENHANDS_APPTAINER_WORKSPACE_ROOT")
        if not workspace_root:
            workspace_root = os.path.join(
                tempfile.gettempdir(),
                f"openhands-apptainer-workspaces-{os.getuid()}",
            )
        safe_instance_id = instance.id.replace("/", "__")
        for _ in range(3):
            mount_dir = os.path.join(
                workspace_root,
                f"{safe_instance_id}-attempt{self.current_attempt}-{uuid.uuid4().hex[:8]}",
            )
            try:
                os.makedirs(mount_dir, mode=0o700, exist_ok=False)
                return mount_dir
            except FileExistsError:
                continue
        raise RuntimeError(
            f"Could not create a unique Apptainer workspace under {workspace_root}"
        )

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up SWE-bench evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    # ---- Hook: prepare a workspace per instance ----------------------------------
    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """
        Use DockerWorkspace by default.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
                           Higher values allocate more CPU/memory resources.
                           Used by APIRemoteWorkspace for remote runtime allocation.
        """
        forward_env = get_acp_forward_env(self.metadata.agent_type, forward_env)

        official_docker_image = self.get_official_docker_image(instance)
        build_target = constants.DEFAULT_BUILD_TARGET
        custom_tag = self.extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = (
            f"-{build_target}" if build_target != constants.BUILD_TARGET_BINARY else ""
        )
        base_agent_image = f"{EVAL_AGENT_SERVER_IMAGE}:{get_phased_image_tag_prefix()}-{custom_tag}{suffix}"
        wrap_needed = self.should_wrap_instance(instance)
        agent_server_image = base_agent_image

        if self.metadata.workspace_type == "docker":
            built = ensure_local_image(
                agent_server_image=base_agent_image,
                base_image=official_docker_image,
                custom_tag=custom_tag,
                target=build_target,
            )
            if built and wrap_needed:
                wrapped_result = wrap_image(base_agent_image, push=False)
                if wrapped_result.error:
                    raise RuntimeError(
                        "Wrapped image build failed: "
                        f"{wrapped_result.error}; log={wrapped_result.log_path}"
                    )
            elif not built and wrap_needed:
                logger.info(
                    f"Using pre-built image {base_agent_image} "
                    "(assumed already wrapped)"
                )

            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
            )
        elif self.metadata.workspace_type == "apptainer":
            force_local_build = os.getenv(
                "OPENHANDS_APPTAINER_FORCE_BUILD", ""
            ).lower() in {"1", "true", "yes"}
            sif_dir = os.getenv(APPTAINER_SIF_DIR_ENV)
            if sif_dir:
                # Prebuilt SIF per instance, as hyperion's ApptainerConfig expects.
                sif_path = os.path.join(sif_dir, f"{instance.id}.sif")
                if not os.path.isfile(sif_path):
                    raise FileNotFoundError(
                        f"Prebuilt Apptainer SIF not found: {sif_path}"
                    )
                logger.info("Using prebuilt Apptainer SIF %s", sif_path)
                return self.start_apptainer_workspace(
                    instance, forward_env, sif_file=sif_path
                )
            elif not force_local_build and remote_image_exists(agent_server_image):
                logger.info(
                    f"Using apptainer workspace with pre-built image {agent_server_image} "
                    f"(tag prefix: {get_phased_image_tag_prefix()})"
                )
                if wrap_needed:
                    logger.info(
                        "Using pre-built wrapped apptainer image for wrapped repo"
                    )

                return self.start_apptainer_workspace(
                    instance, forward_env, server_image=agent_server_image
                )
            else:
                logger.info(
                    "Agent server image %s is not available in the registry; "
                    "building a local Apptainer SIF from %s",
                    agent_server_image,
                    official_docker_image,
                )
                local_agent_image = ensure_apptainer_agent_image(
                    base_image=official_docker_image,
                    custom_tag=custom_tag,
                    target=build_target,
                    wrap_swebench_deps=wrap_needed,
                )
                return self.start_apptainer_workspace(
                    instance, forward_env, sif_file=str(local_agent_image)
                )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            agent_server_image = f"{EVAL_AGENT_SERVER_IMAGE}:{get_phased_image_tag_prefix()}-{custom_tag}{suffix}"
            if not remote_image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry, "
                    "make sure to build, push it, and make it public accessible before using remote workspace."
                )
            logger.info(
                f"Using remote workspace with image {agent_server_image} "
                f"(tag prefix: {get_phased_image_tag_prefix()}, resource_factor: {resource_factor})"
            )
            startup_timeout = float(
                os.getenv(
                    "REMOTE_RUNTIME_STARTUP_TIMEOUT",
                    str(constants.DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT),
                )
            )
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", constants.DEFAULT_RUNTIME_API_URL
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                target_type="source" if "source" in build_target else "binary",
                forward_env=forward_env or [],
                resource_factor=resource_factor,
                init_timeout=startup_timeout,
                startup_wait_timeout=startup_timeout,
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        self.setup_workspace(instance, workspace)
        return workspace

    def start_apptainer_workspace(
        self,
        instance: EvalInstance,
        forward_env: list[str] | None,
        *,
        sif_file: str | None = None,
        server_image: str | None = None,
    ) -> DirectoryOverlayWorkspace:
        """Start and set up an Apptainer sandbox, retrying failed starts as hyperion does.

        A start can fail for reasons unrelated to the instance, e.g. the agent-server
        port being taken by another process between the claim and the server's bind.
        Hyperion's rollout retries sandbox setup NUM_RETRIES_SANDBOX_START (3) times
        before the episode begins; doing the same keeps such failures from using up
        the instance's evaluation attempts.
        """
        last_error: Exception | None = None
        for attempt in range(1, APPTAINER_START_ATTEMPTS + 1):
            workspace: DirectoryOverlayWorkspace | None = None
            try:
                workspace = self.make_apptainer_workspace(
                    instance, forward_env, sif_file=sif_file, server_image=server_image
                )
                self.setup_workspace(instance, workspace)
                return workspace
            except ValueError:
                # Configuration errors (e.g. no scratch directory) are not transient.
                raise
            except Exception as error:
                last_error = error
                if workspace is not None:
                    workspace.cleanup()
                if attempt < APPTAINER_START_ATTEMPTS:
                    logger.warning(
                        "Apptainer workspace start attempt %d/%d failed for %s: %s",
                        attempt,
                        APPTAINER_START_ATTEMPTS,
                        instance.id,
                        error,
                    )
        raise RuntimeError(
            f"Error preparing apptainer workspace for {instance.id}: {last_error}"
        ) from last_error

    def setup_workspace(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> None:
        """Run the environment setup commands in a freshly started workspace."""
        for cmd in self.metadata.env_setup_commands or []:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")

        # Pre-activate testbed in the agent's terminal before the conversation starts.
        activate = workspace.execute_command(
            f"cat >> ~/.bashrc <<'EOF'\n{ACTIVATE_TESTBED}EOF", timeout=60
        )
        if activate.exit_code != 0:
            # The agent can still activate the environment itself, as the prompt describes.
            logger.warning(
                "Could not pre-activate testbed for %s: %s",
                instance.id,
                activate.stderr,
            )

    def make_apptainer_workspace(
        self,
        instance: EvalInstance,
        forward_env: list[str] | None,
        *,
        sif_file: str | None = None,
        server_image: str | None = None,
    ) -> DirectoryOverlayWorkspace:
        """Create an Apptainer sandbox with hyperion training's settings.

        Same class and arguments as hyperion's ApptainerConfig.create_workspace:
        --containall with a directory overlay on node-local scratch (instead of
        --compat's writable tmpfs), working directly in /testbed, no /workspace bind.
        """
        scratch_dir = os.getenv(APPTAINER_SCRATCH_DIR_ENV)
        if not scratch_dir or not os.path.isdir(scratch_dir):
            raise ValueError(
                f"{APPTAINER_SCRATCH_DIR_ENV} must be an existing local directory "
                f"(e.g. the allocation's /local scratch); got {scratch_dir!r}"
            )
        cache_dir = os.getenv("APPTAINER_CACHEDIR") or os.path.expanduser(
            "~/.apptainer_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return DirectoryOverlayWorkspace(
            sif_file=sif_file,
            server_image=server_image,
            cache_dir=cache_dir,
            scratch_dir=scratch_dir,
            working_dir=self.get_repo_path(instance),
            forward_env=forward_env or [],
            extra_bind_mounts=self.get_apptainer_extra_bind_mounts(),
            detach_logs=False,
            health_check_timeout=600,
        )

    # ---- Hook: evaluate one instance ---------------------------------------------
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, run agent, collect history and git patch.
        Do not write files here; just return EvalOutput.
        """
        if is_acp_agent(self.metadata.agent_type):
            agent = build_acp_agent(self.metadata.agent_type, self.metadata.llm.model)
        else:
            agent_llm = build_eval_llm(self.metadata.llm)
            tools = get_tools_for_preset(
                preset=self.metadata.tool_preset,
                # Disable browser tools in CLI mode
                enable_browser=False,
            )
            if self.metadata.enable_delegation:
                tools.append(Tool(name=TaskToolSet.name))
            condenser = None
            if self.metadata.enable_condenser:
                condenser_llm = build_eval_llm(
                    self.metadata.llm,
                    usage_id="condenser",
                )
                if self.metadata.condenser_max_output_tokens is not None:
                    condenser_llm = condenser_llm.model_copy(
                        deep=True,
                        update={
                            "max_output_tokens": (
                                self.metadata.condenser_max_output_tokens
                            ),
                        },
                    )
                condenser = LLMSummarizingCondenser(
                    llm=condenser_llm,
                    max_size=self.metadata.condenser_max_size,
                    max_tokens=self.metadata.condenser_max_tokens,
                    keep_first=self.metadata.condenser_keep_first,
                )
            # No agent_context (public skills), matching hyperion training's Agent.
            agent = Agent(
                llm=agent_llm,
                tools=tools,
                system_prompt_kwargs={"cli_mode": True},
                condenser=condenser,
                agent_context=None,
                # TODO: we can enable security analyzer later
                # security_analyzer=LLMSecurityAnalyzer(),
            )

        assert isinstance(workspace, RemoteWorkspace)

        setup_acp_workspace(self.metadata.agent_type, workspace)

        repo_path = self.get_repo_path(instance)
        instance.data["repo_path"] = repo_path
        # Recorded below for this attempt's workspace; drop any earlier attempt's.
        instance.data.pop("start_commit", None)

        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[persist_callback],
            max_iteration_per_run=self.metadata.max_iterations,
            delete_on_close=True,
        )

        logger.info("repo_path: %s", repo_path)
        source_repo_path = self.get_source_repo_path(instance)
        if repo_path != source_repo_path:
            # Docker/remote: copy /testbed into the agent-owned /workspace. Apptainer
            # edits /testbed in place, as in hyperion training.
            cp_testbed_repo = workspace.execute_command(
                f"mkdir -p {repo_path} ; cp -r {source_repo_path}/. {repo_path}"
            )
            assert cp_testbed_repo.exit_code == 0, (
                f"cp_testbed_repo failed: {cp_testbed_repo.stderr}"
            )

        # git reset
        # A fresh overlay makes git re-hash the whole tree; 300 s as in hyperion's setup.
        git_reset = workspace.execute_command(
            f"cd {repo_path} ; git reset --hard", timeout=300
        )
        assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"

        # The patch is diffed against the commit the agent starts from (see get_patch_base).
        rev_parse = workspace.execute_command(f"cd {repo_path} ; git rev-parse HEAD")
        start_commit = rev_parse.stdout.strip()
        assert rev_parse.exit_code == 0 and re.fullmatch(
            r"[0-9a-f]{40}", start_commit
        ), f"git rev-parse HEAD failed: {rev_parse.stderr or start_commit!r}"
        instance.data["start_commit"] = start_commit

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        conversation_error: str | None = None
        conversation_error_category: str | None = None
        run_stop_confirmed: bool | None = None
        with workspace_keepalive(self.metadata.agent_type, workspace):
            conversation.send_message(instruction)
            # Run conversation with fake user responses to handle agent messages
            try:
                run_conversation_with_fake_user_response(conversation)
            except ConversationRunError as error:
                if not end_episode_on_agent_error():
                    raise
                message = str(error)
                detail = conversation_error_detail(conversation)
                if detail and detail not in message:
                    message = f"{message} | {detail}"
                category = classify_agent_error(message)
                if category is None:
                    logger.warning(
                        "Instance %s: conversation error not caused by the agent; "
                        "retrying the instance: %s",
                        instance.id,
                        message[:500],
                    )
                    raise
                # As in hyperion training, the episode ends here and its final
                # patch is graded; the instance is not retried.
                conversation_error = message
                conversation_error_category = category
                logger.warning(
                    "Instance %s: episode ended by the agent (%s); grading its "
                    "final patch: %s",
                    instance.id,
                    category,
                    message[:500],
                )
                # A timed-out run may still be executing on the server.
                run_stop_confirmed = stop_conversation(conversation, instance.id)

        patch_extraction_error: str | None = None
        try:
            # git add
            workspace.execute_command(f"cd {repo_path} ; git add -A")

            # git commit
            # Use --no-verify to bypass pre-commit hooks (e.g., husky) that can fail
            workspace.execute_command(
                f"cd {repo_path} && "
                f"git config --global user.email '{constants.GIT_USER_EMAIL}' && "
                f"git config --global user.name '{constants.GIT_USER_NAME}' && "
                f"git commit --no-verify -m '{constants.GIT_COMMIT_MESSAGE}'"
            )

            # Get git patch
            git_patch = self.get_git_patch(instance, workspace)
        except Exception as error:
            # For pass@1, a patch that cannot be extracted from a working sandbox
            # (e.g. the agent broke the repository) is graded as empty, as in
            # hyperion training; a sandbox that no longer responds is retried.
            if not end_episode_on_agent_error() or not workspace_responds(workspace):
                raise
            patch_extraction_error = str(error)
            git_patch = ""
            logger.warning(
                "Instance %s: could not extract the patch from a working sandbox; "
                "grading an empty patch: %s",
                instance.id,
                patch_extraction_error[:500],
            )

        # Log instance summary
        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            git_patch=git_patch,
            logger=logger,
        )

        # Build test_result with git patch and optional ACP agent metadata
        test_result: dict[str, Any] = {
            "git_patch": git_patch,
        }
        if conversation_error is not None:
            test_result["conversation_error"] = conversation_error
            test_result["conversation_error_category"] = conversation_error_category
            test_result["run_stop_confirmed"] = run_stop_confirmed
        if patch_extraction_error is not None:
            test_result["patch_extraction_error"] = patch_extraction_error
        if isinstance(agent, ACPAgent):
            add_acp_agent_metadata(test_result, conversation)

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    # Validate n_critic_runs
    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create critic instance from parsed arguments
    critic = create_critic(args)
    logger.info(f"Using critic: {type(critic).__name__}")
    logger.info(f"Using tool preset: {args.tool_preset}")

    # Handle condenser configuration
    # --disable-condenser takes precedence over --enable-condenser and defaults
    enable_condenser = args.enable_condenser
    if args.disable_condenser:
        enable_condenser = False

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        n_critic_runs=args.n_critic_runs,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        tool_preset=args.tool_preset,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
        enable_condenser=enable_condenser,
        condenser_max_size=args.condenser_max_size,
        condenser_max_tokens=args.condenser_max_tokens,
        condenser_max_output_tokens=args.condenser_max_output_tokens,
        condenser_keep_first=args.condenser_keep_first,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = SWEBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    # Like hyperion's trainer, own one shared fakeroot ID mapping for all sandboxes.
    with shared_apptainer_namespace(enabled=args.workspace == "apptainer"):
        evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    # Emit machine-readable path for callers
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
