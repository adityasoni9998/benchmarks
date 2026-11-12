"""SWE-bench conversation errors: retried by default; for genuine pass@1, agent-caused
errors end the episode (its final patch is graded) and all others are retried."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.swebench import run_infer
from benchmarks.swebench.run_infer import SWEBenchEvaluation, classify_agent_error
from benchmarks.utils import fake_user_response
from openhands.sdk import LLM
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.workspace import RemoteWorkspace
from tests.test_metrics import (
    _create_metadata_for_benchmark,
    _get_test_instance_for_benchmark,
    _mock_execute_command,
    _setup_mocks_for_benchmark,
)


# Messages as they appeared in the base-model SWE-bench Verified run.
AGENT_ERRORS = {
    "turn_limit": "MaxIterationsReached: Agent reached maximum iterations limit (75).",
    "stuck": "Remote conversation got stuck",
    "context_window": (
        "LLMBadRequestError: litellm.BadRequestError: OpenAIException - Requested "
        "token count exceeds the model's maximum context length of 66000 tokens."
    ),
    "timeout": (
        "Run timed out after 3600 seconds. The conversation may still be running "
        "on the server."
    ),
    "malformed_tool_call": (
        "LLMBadRequestError: litellm.BadRequestError: OpenAIException - Assistant "
        "tool call function.arguments must be a JSON object."
    ),
}


def _run_error(message: str) -> ConversationRunError:
    return ConversationRunError(uuid.uuid4(), RuntimeError(message))


def _evaluate(
    monkeypatch,
    error: Exception | None,
    end_episode: bool,
    history=(),
    execute_command=_mock_execute_command,
):
    if end_episode:
        monkeypatch.setenv(run_infer.END_EPISODE_ON_AGENT_ERROR_ENV, "1")
    else:
        monkeypatch.delenv(run_infer.END_EPISODE_ON_AGENT_ERROR_ENV, raising=False)
    evaluation = SWEBenchEvaluation(
        metadata=_create_metadata_for_benchmark("swebench", LLM(model="test-model"))
    )
    conversation = _setup_mocks_for_benchmark(
        "swebench", Metrics(model_name="test-model")
    )
    conversation.state.events = list(history)
    conversation.state.refresh_from_server.return_value = {"execution_status": "paused"}
    workspace = MagicMock(spec=RemoteWorkspace)
    workspace.working_dir = "/workspace"
    workspace.execute_command = MagicMock(side_effect=execute_command)
    with (
        patch.object(run_infer, "Conversation", return_value=conversation),
        patch.object(run_infer, "Agent"),
        patch.object(run_infer, "get_tools_for_preset"),
        patch.object(run_infer, "get_instruction", return_value="Test instruction"),
        patch.object(
            run_infer, "run_conversation_with_fake_user_response", side_effect=error
        ),
    ):
        output = evaluation.evaluate_instance(
            _get_test_instance_for_benchmark("swebench"), workspace
        )
    return output, conversation, workspace


def test_conversation_errors_are_retried_by_default(monkeypatch):
    with pytest.raises(ConversationRunError):
        _evaluate(monkeypatch, _run_error(AGENT_ERRORS["turn_limit"]), False)


@pytest.mark.parametrize("category,message", sorted(AGENT_ERRORS.items()))
def test_agent_error_ends_episode_and_grades_patch(monkeypatch, category, message):
    output, conversation, workspace = _evaluate(monkeypatch, _run_error(message), True)

    assert output.error is None
    assert output.test_result["conversation_error_category"] == category
    assert message in output.test_result["conversation_error"]
    assert "git_patch" in output.test_result
    conversation.interrupt.assert_called_once()
    conversation.pause.assert_not_called()
    assert output.test_result["run_stop_confirmed"] is True
    commands = [call.args[0] for call in workspace.execute_command.call_args_list]
    assert any("git add -A" in c for c in commands)
    assert any("git --no-pager diff" in c for c in commands)


def test_generic_error_is_classified_from_the_history(monkeypatch):
    # The ERROR status can arrive before its detailed event; the detail is read back.
    event = ConversationErrorEvent(
        source="environment",
        code="LLMBadRequestError",
        detail="The input (66594 tokens) is longer than the model's context length "
        "(66000 tokens).",
    )
    output, _, _ = _evaluate(
        monkeypatch,
        _run_error("Remote conversation ended with error"),
        True,
        history=[event],
    )

    assert output.test_result["conversation_error_category"] == "context_window"
    assert "LLMBadRequestError" in output.test_result["conversation_error"]


@pytest.mark.parametrize(
    "message",
    [
        "Remote conversation ended with error",  # no detail recovered
        "[Errno 111] Connection refused",  # sandbox or LLM server down
        "Container process stopped unexpectedly with exit code 1",
        "litellm.InternalServerError: OpenAIException - Internal Server Error",
    ],
)
def test_non_agent_errors_are_retried(monkeypatch, message):
    with pytest.raises(ConversationRunError):
        _evaluate(monkeypatch, _run_error(message), True)


def test_other_exceptions_are_retried(monkeypatch):
    with pytest.raises(OSError):
        _evaluate(monkeypatch, OSError("agent server unreachable"), True)


def test_classify_agent_error_unknown():
    assert classify_agent_error("something else went wrong") is None


@pytest.mark.parametrize("env_value,expected", [(None, 10), ("0", 0), ("3", 3)])
def test_max_fake_responses_default_comes_from_env(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("MAX_FAKE_RESPONSES", raising=False)
    else:
        monkeypatch.setenv("MAX_FAKE_RESPONSES", env_value)
    sent: list[str] = []
    conversation = MagicMock()
    conversation.state.execution_status = (
        fake_user_response.ConversationExecutionStatus.FINISHED
    )
    with (
        patch.object(
            fake_user_response, "_agent_finished_with_finish_action", return_value=False
        ),
        patch.object(fake_user_response, "_agent_sent_message", return_value=True),
    ):
        fake_user_response.run_conversation_with_fake_user_response(
            conversation,
            fake_user_response_fn=lambda c: sent.append("continue") or "continue",
        )

    assert len(sent) == expected


@pytest.mark.parametrize(
    "message",
    [
        # The SDK's own error class, with its default message and as an error code.
        "LLMContextWindowExceedError: Conversation history longer than LLM context "
        "window limit. Consider enabling a condenser or shortening inputs.",
        "Conversation history longer than LLM context window limit.",
        # Phrases from the SDK's classifier (LONG_PROMPT_PATTERNS).
        "litellm.BadRequestError: input length and `max_tokens` exceed context limit",
        "OpenAIException - Prompt is too long",
    ],
)
def test_sdk_context_window_errors_are_agent_caused(message):
    assert classify_agent_error(message) == "context_window"


def test_sdk_malformed_history_errors_are_agent_caused():
    message = "LLMBadRequestError: Failed to parse tool call arguments as JSON"
    assert classify_agent_error(message) == "malformed_tool_call"


def _conversation_with_statuses(statuses):
    conversation = MagicMock()
    conversation.state.refresh_from_server.side_effect = [
        {"execution_status": s} for s in statuses
    ]
    return conversation


def test_stop_conversation_interrupts_and_waits_until_the_run_stops(monkeypatch):
    monkeypatch.setattr(run_infer, "STOP_POLL_SECONDS", 0)
    conversation = _conversation_with_statuses(["running", "running", "paused"])

    assert run_infer.stop_conversation(conversation, "i") is True

    conversation.interrupt.assert_called_once()
    conversation.pause.assert_not_called()


def test_stop_conversation_keeps_the_episode_if_the_run_will_not_stop(monkeypatch):
    # Not a crash: the episode's outcome is kept and its patch captured.
    monkeypatch.setattr(run_infer, "STOP_POLL_SECONDS", 0)
    monkeypatch.setattr(run_infer, "STOP_WAIT_SECONDS", 0)
    conversation = MagicMock()
    conversation.state.refresh_from_server.return_value = {
        "execution_status": "running"
    }

    assert run_infer.stop_conversation(conversation, "i") is False


def test_stop_conversation_raises_fast_if_the_server_is_gone(monkeypatch):
    monkeypatch.setattr(run_infer, "STOP_POLL_SECONDS", 0)
    conversation = MagicMock()
    conversation.state.refresh_from_server.side_effect = ConnectionError("refused")

    with pytest.raises(RuntimeError, match="Cannot read run status"):
        run_infer.stop_conversation(conversation, "i")
    assert conversation.state.refresh_from_server.call_count == 3


def _broken_diff(cmd, timeout=None):
    if "git --no-pager diff" in cmd:
        return MagicMock(exit_code=128, stdout="", stderr="fatal: not a git repository")
    return _mock_execute_command(cmd, timeout)


def _broken_diff_dead_sandbox(cmd, timeout=None):
    if cmd == "true":
        raise ConnectionError("sandbox gone")
    return _broken_diff(cmd, timeout)


def test_patch_extraction_failure_grades_empty_patch_in_pass1(monkeypatch):
    output, _, _ = _evaluate(monkeypatch, None, True, execute_command=_broken_diff)

    assert output.error is None
    assert output.test_result["git_patch"] == ""
    assert "git diff failed" in output.test_result["patch_extraction_error"]


def test_patch_extraction_failure_is_retried_if_the_sandbox_is_gone(monkeypatch):
    with pytest.raises(AssertionError, match="git diff failed"):
        _evaluate(monkeypatch, None, True, execute_command=_broken_diff_dead_sandbox)


def test_patch_extraction_failure_is_retried_by_default(monkeypatch):
    with pytest.raises(AssertionError, match="git diff failed"):
        _evaluate(monkeypatch, None, False, execute_command=_broken_diff)
