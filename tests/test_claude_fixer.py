"""Tests for ci_fix.agent.fixer.ClaudeFixer (slice 4) with a scripted fake chat model.

No network: the LLM is a ``ScriptedChatModel`` that replays pre-written ``AIMessage``s and
records what the fixer sent it. The tools run for real on a git copy of the sample repo.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from conftest import git
from fixtures import copy_sample_repo
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pipeline_helpers import (
    MEAN,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SUBTRACT,
    SUBTRACT_BUG,
    SUBTRACT_FIX,
)
from pydantic import ValidationError

import ci_fix
from ci_fix.agent.fixer import ClaudeFixer
from ci_fix.config import ConfigError, Settings
from ci_fix.models import FixAttempt, FixRequest
from ci_fix.pipeline import fix_failing_tests
from ci_fix.tools.pytest_runner import TestResult, TestStatus

FAKE_KEY = "sk-ant-test-DO-NOT-LOG-0123456789"
TOOL_NAMES = {
    "list_files",
    "read_file",
    "search_code",
    "edit_file",
    "create_file",
    "run_test",
    "finish",
}
EXPLANATION = (
    "Root cause: subtract adds instead of subtracting.\n"
    "Fix: return a - b.\n"
    "Why this file: the bug is in subtract in ops.py."
)


# --------------------------------------------------------------------------- #
# Fake chat model
# --------------------------------------------------------------------------- #

_ids = itertools.count(1)


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"toolu_{next(_ids):04d}", "type": "tool_call"}


def ai(*calls: dict[str, Any], content: str = "", usage: dict[str, int] | None = None) -> AIMessage:
    kwargs: dict[str, Any] = {"content": content, "tool_calls": list(calls)}
    if usage is not None:
        kwargs["usage_metadata"] = usage
    return AIMessage(**kwargs)


def finish(outcome: str = "changed", explanation: str = EXPLANATION) -> dict[str, Any]:
    return call("finish", outcome=outcome, explanation=explanation)


class ScriptedChatModel:
    """Replays scripted ``AIMessage``s; records bound tools and every message list it got."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list[BaseMessage]] = []
        self.bound_tools: list[Any] | None = None
        self.bind_kwargs: dict[str, Any] = {}

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        self.bind_kwargs = kwargs
        return self

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("fake LLM called more times than scripted")
        return self.responses.pop(0)

    def tool_messages(self) -> list[ToolMessage]:
        """ToolMessages in the last message list the model was given."""
        return [m for m in self.calls[-1] if isinstance(m, ToolMessage)]

    def tool_message_for(self, tool_call: dict[str, Any]) -> ToolMessage:
        matches = [m for m in self.tool_messages() if m.tool_call_id == tool_call["id"]]
        assert len(matches) == 1, f"expected one ToolMessage for {tool_call['id']}"
        return matches[0]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = copy_sample_repo(tmp_path / "repo")
    git("init", "-q", "-b", "main", cwd=path)
    git("config", "user.email", "tester@example.com", cwd=path)
    git("config", "user.name", "Tester", cwd=path)
    git("config", "commit.gpgsign", "false", cwd=path)
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "initial", cwd=path)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(workspace_dir=tmp_path / "ws", anthropic_api_key=FAKE_KEY)


class FakeRunTest:
    """``FixRequest.run_test`` stand-in: target passes once ops.py has the subtract fix."""

    def __init__(self, repo: Path, target: str = SUBTRACT) -> None:
        self.repo = repo
        self.target = target
        self.executed: list[str] = []

    def __call__(self, node_id: str) -> TestResult:
        if node_id != self.target:
            msg = "only the target test can be run"
            return TestResult(node_id=node_id, status=TestStatus.NOT_FOUND, message=msg)
        self.executed.append(node_id)
        fixed = SUBTRACT_FIX in (self.repo / OPS_PY).read_text(encoding="utf-8")
        if fixed:
            return TestResult(node_id=node_id, status=TestStatus.PASSED)
        msg = "AssertionError: assert 8 == 2"
        return TestResult(node_id=node_id, status=TestStatus.FAILED, message=msg, details=msg)


def make_request(repo: Path, run_test: Callable[[str], TestResult] | None = None) -> FixRequest:
    return FixRequest(
        node_id=SUBTRACT,
        repo_path=repo,
        attempt=1,
        max_attempts=3,
        failure_message="AssertionError: assert 8 == 2",
        failure_details="E   assert 8 == 2",
        pr_title="Feature",
        pr_body="Adds things",
        pr_diff="diff --git a/PR_CHANGE.txt b/PR_CHANGE.txt\n+change from the PR\n",
        other_failing_tests=[MEAN],
        run_test=run_test if run_test is not None else FakeRunTest(repo),
    )


def run_fixer(
    settings: Settings, repo: Path, responses: list[AIMessage], **request_kw: Any
) -> tuple[FixAttempt, ScriptedChatModel]:
    llm = ScriptedChatModel(responses)
    fixer = ClaudeFixer(settings, llm=llm)
    return fixer.fix(make_request(repo, **request_kw)), llm


def is_error(text: str) -> bool:
    return "error" in text.lower() or "refus" in text.lower() or "not allowed" in text.lower()


def content_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)


# --------------------------------------------------------------------------- #
# Settings for the agent
# --------------------------------------------------------------------------- #


def test_agent_settings_defaults() -> None:
    s = Settings()
    assert s.max_agent_steps == 30
    assert s.max_output_tokens == 4096
    assert s.pr_diff_max_chars == 40_000
    assert s.read_max_lines == 400


@pytest.mark.parametrize(
    ("field", "bad", "ok"),
    [
        ("max_agent_steps", 0, 1),
        ("max_output_tokens", 255, 256),
        ("pr_diff_max_chars", 999, 1000),
        ("read_max_lines", 49, 50),
    ],
)
def test_agent_settings_bounds(field: str, bad: int, ok: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: bad})
    assert getattr(Settings(**{field: ok}), field) == ok


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_claude_fixer_exported_from_package() -> None:
    assert ci_fix.ClaudeFixer is ClaudeFixer


def test_no_key_and_no_llm_is_config_error(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        fixer = ClaudeFixer(Settings(workspace_dir=tmp_path / "ws"))
        fixer.fix(make_request(repo))


def test_key_without_llm_constructs_without_network(tmp_path: Path) -> None:
    ClaudeFixer(Settings(workspace_dir=tmp_path / "ws", anthropic_api_key=FAKE_KEY))


def test_llm_without_key_is_allowed(repo: Path, tmp_path: Path) -> None:
    llm = ScriptedChatModel([ai(finish("unfixable", "intent ambiguous"))])
    attempt = ClaudeFixer(Settings(workspace_dir=tmp_path / "ws"), llm=llm).fix(make_request(repo))
    assert attempt.outcome == "unfixable"


def test_fix_failing_tests_default_fixer_needs_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    github = MagicMock()
    github.get_pull_request.side_effect = AssertionError("must fail before touching GitHub")
    with pytest.raises(ConfigError):
        fix_failing_tests(
            REPO_URL,
            SAMPLE_PR,
            [SUBTRACT],
            settings=Settings(workspace_dir=tmp_path / "ws"),
            github=github,
        )


# --------------------------------------------------------------------------- #
# Tool-calling loop
# --------------------------------------------------------------------------- #


def test_binds_all_tools(settings: Settings, repo: Path) -> None:
    _, llm = run_fixer(settings, repo, [ai(finish("unfixable", "intent ambiguous"))])
    assert llm.bound_tools is not None
    names = {convert_to_openai_tool(t)["function"]["name"] for t in llm.bound_tools}
    assert TOOL_NAMES <= names


def test_first_turn_has_system_prompt_and_request(settings: Settings, repo: Path) -> None:
    _, llm = run_fixer(settings, repo, [ai(finish("unfixable", "intent ambiguous"))])
    first = llm.calls[0]
    assert isinstance(first[0], SystemMessage)
    human = [m for m in first if isinstance(m, HumanMessage)]
    assert human
    text = " ".join(content_of(m) for m in human)
    assert SUBTRACT in text
    assert "assert 8 == 2" in text
    assert MEAN in text  # other failing tests
    assert "change from the PR" in text  # PR diff


def test_read_edit_run_finish_fixes_subtract(settings: Settings, repo: Path) -> None:
    read = call("read_file", path=OPS_PY)
    search = call("search_code", pattern="def subtract")
    edit = call("edit_file", path=OPS_PY, old_str=SUBTRACT_BUG, new_str=SUBTRACT_FIX)
    run = call("run_test", node_id=SUBTRACT)
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    script = [
        ai(read, search, content="Let me look.", usage=usage),
        ai(edit, usage=usage),
        ai(run, usage=usage),
        ai(finish("changed"), usage=usage),
    ]
    attempt, llm = run_fixer(settings, repo, script)

    text = (repo / OPS_PY).read_text(encoding="utf-8")
    assert "def subtract(a: float, b: float) -> float:" in text
    assert "    return a - b\n" in text
    assert SUBTRACT_BUG not in text
    assert "return a + b\n" in text  # add() untouched
    assert attempt.outcome == "changed"
    assert attempt.files_changed == [OPS_PY]
    assert attempt.node_id == SUBTRACT
    assert attempt.attempt == 1
    assert "Root cause:" in attempt.explanation
    assert len(llm.calls) == 4

    # Each tool result goes back with the id of the call it answers.
    assert "SEEDED BUG" in content_of(llm.tool_message_for(read))
    assert "ops.py" in content_of(llm.tool_message_for(search))
    assert not is_error(content_of(llm.tool_message_for(edit)))
    assert "passed" in content_of(llm.tool_message_for(run)).lower()


def test_parallel_tool_calls_each_get_a_tool_message(settings: Settings, repo: Path) -> None:
    calls = [
        call("list_files"),
        call("read_file", path=OPS_PY, start_line=1, end_line=5),
        call("search_code", pattern="mean", path="src"),
    ]
    _, llm = run_fixer(
        settings, repo, [ai(*calls), ai(finish("unfixable", "intent ambiguous: both"))]
    )
    second = llm.calls[1]
    tool_ids = [m.tool_call_id for m in second if isinstance(m, ToolMessage)]
    assert tool_ids == [c["id"] for c in calls]
    # The AIMessage with the tool calls precedes its ToolMessages.
    ai_index = next(i for i, m in enumerate(second) if isinstance(m, AIMessage))
    first_tool = next(i for i, m in enumerate(second) if isinstance(m, ToolMessage))
    assert ai_index < first_tool


def test_finish_unfixable(settings: Settings, repo: Path) -> None:
    reason = "intent ambiguous: the docstring says a - b but callers may rely on a + b"
    before = (repo / OPS_PY).read_text(encoding="utf-8")
    attempt, llm = run_fixer(settings, repo, [ai(finish("unfixable", reason))])
    assert attempt.outcome == "unfixable"
    assert reason in attempt.explanation
    assert attempt.files_changed == []
    assert (repo / OPS_PY).read_text(encoding="utf-8") == before
    assert len(llm.calls) == 1


def test_finish_changed_without_edits_is_no_change(settings: Settings, repo: Path) -> None:
    attempt, _ = run_fixer(settings, repo, [ai(call("read_file", path=OPS_PY)), ai(finish())])
    assert attempt.outcome == "no_change"
    assert attempt.files_changed == []


def test_turn_without_tool_calls_gets_reminder(settings: Settings, repo: Path) -> None:
    script = [ai(content="I think the bug is in subtract."), ai(finish("unfixable", "ambiguous"))]
    attempt, llm = run_fixer(settings, repo, script)
    assert attempt.outcome == "unfixable"
    assert len(llm.calls) == 2
    second = llm.calls[1]
    assert isinstance(second[-1], HumanMessage)  # the reminder
    assert len([m for m in second if isinstance(m, HumanMessage)]) >= 2
    assert content_of(second[-1]) != content_of(
        next(m for m in second if isinstance(m, HumanMessage))
    )


def test_turn_without_tool_calls_counts_as_a_step(tmp_path: Path, repo: Path) -> None:
    settings = Settings(
        workspace_dir=tmp_path / "ws", anthropic_api_key=FAKE_KEY, max_agent_steps=1
    )
    attempt, llm = run_fixer(settings, repo, [ai(content="thinking..."), ai(finish())])
    assert len(llm.calls) == 1
    assert attempt.outcome == "no_change"


def test_unknown_tool_returns_error_and_loop_continues(settings: Settings, repo: Path) -> None:
    bogus = call("frobnicate", path=OPS_PY)
    edit = call("edit_file", path=OPS_PY, old_str=SUBTRACT_BUG, new_str=SUBTRACT_FIX)
    attempt, llm = run_fixer(settings, repo, [ai(bogus), ai(edit), ai(finish())])
    msg = content_of(llm.calls[1][-1])
    assert isinstance(llm.calls[1][-1], ToolMessage)
    assert llm.calls[1][-1].tool_call_id == bogus["id"]
    assert "frobnicate" in msg
    assert attempt.outcome == "changed"
    assert attempt.files_changed == [OPS_PY]


def test_bad_tool_args_return_error_and_loop_continues(settings: Settings, repo: Path) -> None:
    bad = call("edit_file", path=OPS_PY)  # missing old_str/new_str
    bad_type = call("read_file", path=OPS_PY, start_line="not-a-number")
    attempt, llm = run_fixer(
        settings, repo, [ai(bad), ai(bad_type), ai(finish("unfixable", "ambiguous"))]
    )
    assert is_error(content_of(llm.calls[1][-1]))
    assert llm.calls[1][-1].tool_call_id == bad["id"]
    assert llm.calls[2][-1].tool_call_id == bad_type["id"]
    assert attempt.outcome == "unfixable"


def test_tool_error_strings_are_passed_back(settings: Settings, repo: Path) -> None:
    edit = call("edit_file", path=OPS_PY, old_str="return a + b", new_str="return a - b")
    _, llm = run_fixer(settings, repo, [ai(edit), ai(finish("unfixable", "ambiguous"))])
    msg = content_of(llm.tool_message_for(edit))
    assert is_error(msg)
    assert "2" in msg  # two matches


def test_edit_outside_repo_is_refused(settings: Settings, repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me\n", encoding="utf-8")
    edit = call("edit_file", path="../outside.txt", old_str="keep me", new_str="pwned")
    create = call("create_file", path="../created.txt", content="pwned\n")
    attempt, llm = run_fixer(settings, repo, [ai(edit, create), ai(finish())])
    assert is_error(content_of(llm.tool_message_for(edit)))
    assert is_error(content_of(llm.tool_message_for(create)))
    assert outside.read_text(encoding="utf-8") == "keep me\n"
    assert not (tmp_path / "created.txt").exists()
    assert attempt.files_changed == []
    assert attempt.outcome == "no_change"


def test_steps_exhausted_with_edits_is_changed(tmp_path: Path, repo: Path) -> None:
    settings = Settings(
        workspace_dir=tmp_path / "ws", anthropic_api_key=FAKE_KEY, max_agent_steps=2
    )
    edit = call("edit_file", path=OPS_PY, old_str=SUBTRACT_BUG, new_str=SUBTRACT_FIX)
    script = [ai(edit), ai(call("read_file", path=OPS_PY)), ai(finish())]
    attempt, llm = run_fixer(settings, repo, script)
    assert len(llm.calls) == 2
    assert attempt.outcome == "changed"
    assert attempt.files_changed == [OPS_PY]
    assert "step" in attempt.explanation.lower() or "stop" in attempt.explanation.lower()


def test_steps_exhausted_without_edits_is_no_change(tmp_path: Path, repo: Path) -> None:
    settings = Settings(
        workspace_dir=tmp_path / "ws", anthropic_api_key=FAKE_KEY, max_agent_steps=2
    )
    script = [ai(call("list_files")), ai(call("read_file", path=OPS_PY)), ai(finish())]
    attempt, llm = run_fixer(settings, repo, script)
    assert len(llm.calls) == 2
    assert attempt.outcome == "no_change"
    assert attempt.files_changed == []
    assert "step" in attempt.explanation.lower() or "stop" in attempt.explanation.lower()


def test_files_changed_sorted_and_unique(settings: Settings, repo: Path) -> None:
    edit1 = call("edit_file", path=OPS_PY, old_str=SUBTRACT_BUG, new_str=SUBTRACT_FIX)
    edit2 = call(
        "edit_file",
        path=OPS_PY,
        old_str="  # SEEDED BUG: off-by-one, should be len(xs)",
        new_str="",
    )
    create = call("create_file", path="src/calc/aaa_helper.py", content="X = 1\n")
    attempt, _ = run_fixer(settings, repo, [ai(edit1, edit2, create), ai(finish())])
    assert attempt.files_changed == sorted({OPS_PY, "src/calc/aaa_helper.py"})


def test_run_test_target_reports_failure_before_fix(settings: Settings, repo: Path) -> None:
    run = call("run_test", node_id=SUBTRACT)
    _, llm = run_fixer(settings, repo, [ai(run), ai(finish("unfixable", "ambiguous"))])
    msg = content_of(llm.tool_message_for(run)).lower()
    assert "failed" in msg
    assert "assert 8 == 2" in msg


def test_run_test_for_other_id_is_refused(settings: Settings, repo: Path) -> None:
    runner = FakeRunTest(repo)
    run = call("run_test", node_id=MEAN)
    _, llm = run_fixer(
        settings, repo, [ai(run), ai(finish("unfixable", "ambiguous"))], run_test=runner
    )
    msg = content_of(llm.tool_message_for(run)).lower()
    assert "only the target" in msg or "not_found" in msg or "not found" in msg
    assert MEAN not in runner.executed


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_token_usage_logged_at_info(
    settings: Settings, repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    usage = {"input_tokens": 12345, "output_tokens": 678, "total_tokens": 13023}
    run_fixer(settings, repo, [ai(finish("unfixable", "ambiguous"), usage=usage)])
    info = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert "12345" in info or "12,345" in info
    assert "678" in info


def test_api_key_never_logged(
    settings: Settings, repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="ci_fix")
    usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    edit = call("edit_file", path=OPS_PY, old_str=SUBTRACT_BUG, new_str=SUBTRACT_FIX)
    script = [
        ai(call("read_file", path=OPS_PY), usage=usage),
        ai(call("frobnicate"), usage=usage),
        ai(edit, usage=usage),
        ai(call("run_test", node_id=SUBTRACT), usage=usage),
        ai(finish(), usage=usage),
    ]
    run_fixer(settings, repo, script)
    assert caplog.records  # the fixer did log something
    assert FAKE_KEY not in caplog.text
    for record in caplog.records:
        assert FAKE_KEY not in record.getMessage()
        assert FAKE_KEY not in str(record.args)


def test_api_key_never_sent_to_model(settings: Settings, repo: Path) -> None:
    _, llm = run_fixer(settings, repo, [ai(finish("unfixable", "ambiguous"))])
    for message in llm.calls[0]:
        assert FAKE_KEY not in content_of(message)


def test_default_llm_is_deterministic_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real model must be built from settings with temperature 0 (deterministic)."""
    import ci_fix.agent.fixer as fixer_module

    captured: dict = {}

    class _FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr(fixer_module, "ChatAnthropic", _FakeChatAnthropic)
    settings = Settings(anthropic_api_key="sk-test-not-real", max_output_tokens=2048)
    fixer_module.ClaudeFixer(settings)
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 2048
    key = captured["api_key"]
    assert (
        key.get_secret_value() if hasattr(key, "get_secret_value") else key
    ) == "sk-test-not-real"


def test_temperature_none_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    import ci_fix.agent.fixer as fixer_module

    captured: dict = {}

    class _FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr(fixer_module, "ChatAnthropic", _FakeChatAnthropic)
    settings = Settings(anthropic_api_key="sk-x", model="claude-sonnet-5-5", temperature=None)
    fixer_module.ClaudeFixer(settings)
    assert "temperature" not in captured
    assert captured["model"] == "claude-sonnet-5-5"


class _ApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }


class _RaisingModel:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        raise self.exc


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_fatal_api_errors_raise_fixer_fatal_error(tmp_path: Path, status: int) -> None:
    from ci_fix.models import FixerFatalError, FixRequest

    fixer = ClaudeFixer(
        Settings(anthropic_api_key="sk-x"),
        llm=_RaisingModel(_ApiError(status, "Your credit balance is too low")),
    )
    req = FixRequest(
        node_id="t.py::test_x",
        repo_path=tmp_path,
        attempt=1,
        max_attempts=3,
        failure_message="boom",
        failure_details="",
    )
    with pytest.raises(FixerFatalError, match="credit balance is too low"):
        fixer.fix(req)


def test_transient_api_errors_are_not_fatal(tmp_path: Path) -> None:
    from ci_fix.models import FixerFatalError, FixRequest

    fixer = ClaudeFixer(
        Settings(anthropic_api_key="sk-x"), llm=_RaisingModel(_ApiError(529, "overloaded"))
    )
    req = FixRequest(
        node_id="t.py::test_x",
        repo_path=tmp_path,
        attempt=1,
        max_attempts=3,
        failure_message="boom",
        failure_details="",
    )
    with pytest.raises(_ApiError):
        fixer.fix(req)
    try:
        fixer.fix(req)
    except FixerFatalError:  # pragma: no cover
        pytest.fail("529 must not be fatal")
    except _ApiError:
        pass
