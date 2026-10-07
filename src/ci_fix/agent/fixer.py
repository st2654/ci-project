"""ClaudeFixer: a tool-using LLM agent that fixes one failing test per call."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from ci_fix.agent.llm import build_chat_model, invoke_or_fatal
from ci_fix.agent.prompts import SYSTEM_PROMPT, build_user_message
from ci_fix.agent.tools import RepoTools
from ci_fix.config import Settings
from ci_fix.logging_setup import get_logger
from ci_fix.models import FixAttempt, FixRequest

log = get_logger(__name__)

RUN_TEST_DETAILS_MAX_CHARS = 4000
LOG_ARGS_MAX_CHARS = 200
NUDGE_MESSAGE = (
    "Use the tools to investigate and fix the failing test, then call finish. "
    "If it cannot be fixed honestly, call finish with outcome 'unfixable' and the reason."
)


# ---- tool argument schemas (also what the model sees) ---------------------------------------


class ListFilesArgs(BaseModel):
    path: str = Field(default=".", description="Repo-relative directory to list.")
    pattern: str | None = Field(
        default=None, description="Optional fnmatch pattern, e.g. '*.py' or 'tests/*'."
    )


class ReadFileArgs(BaseModel):
    path: str = Field(description="Repo-relative file path.")
    start_line: int = Field(default=1, ge=1, description="First line to read (1-based).")
    end_line: int | None = Field(default=None, ge=1, description="Last line (inclusive).")


class SearchCodeArgs(BaseModel):
    pattern: str = Field(description="Text to search for (fixed string unless regex=true).")
    path: str = Field(default=".", description="Repo-relative file or directory to search.")
    regex: bool = Field(default=False, description="Treat pattern as an extended regex.")


class EditFileArgs(BaseModel):
    path: str = Field(description="Repo-relative file path.")
    old_str: str = Field(description="Exact text to replace; must occur exactly once.")
    new_str: str = Field(description="Replacement text.")


class CreateFileArgs(BaseModel):
    path: str = Field(description="Repo-relative path of a file that does not exist yet.")
    content: str = Field(description="Full file content.")


class RunTestArgs(BaseModel):
    node_id: str | None = Field(
        default=None, description="Optional; only the failing target test can be run."
    )


class FinishArgs(BaseModel):
    outcome: Literal["changed", "unfixable"] = Field(
        description="'changed' if you made a fix, 'unfixable' if it cannot be fixed honestly."
    )
    explanation: str = Field(
        description="Concise, for the PR description: 'Root cause: ...\\nFix: ...\\n"
        "Why this file: ...' (about 80 words max), plus a 'Test change: <why the old "
        "expectation was wrong>' line if a test's expectation changed. For unfixable: the reason."
    )


_TOOL_SPECS: dict[str, tuple[type[BaseModel], str]] = {
    "list_files": (ListFilesArgs, "List tracked and untracked files under a directory."),
    "read_file": (ReadFileArgs, "Read a file with line numbers (a limited number per call)."),
    "search_code": (SearchCodeArgs, "Search the repository (git grep); returns file:line:text."),
    "edit_file": (EditFileArgs, "Replace one exact, unique occurrence of old_str in a file."),
    "create_file": (CreateFileArgs, "Create a new file. Fails if the file already exists."),
    "run_test": (RunTestArgs, "Run the failing target test and return its status and output."),
    "finish": (FinishArgs, "Finish this attempt with an outcome and a concise explanation."),
}


def _schema_only(**_: Any) -> str:  # tools are executed by the loop, never by langchain
    raise NotImplementedError


TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(
        func=_schema_only, name=name, description=description, args_schema=schema
    )
    for name, (schema, description) in _TOOL_SPECS.items()
]


def _short(value: Any, limit: int = LOG_ARGS_MAX_CHARS) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...'"


class _Finished(Exception):
    def __init__(self, args: FinishArgs) -> None:
        self.args_ = args


class ClaudeFixer:
    """``Fixer`` backed by a chat model with repo tools (Claude by default)."""

    def __init__(self, settings: Settings, llm: BaseChatModel | None = None) -> None:
        self.settings = settings
        if llm is None:
            llm = build_chat_model(settings)
        self.llm = llm
        try:
            self.llm_with_tools: Any = llm.bind_tools(TOOLS)
        except NotImplementedError:  # e.g. simple fake chat models in tests
            self.llm_with_tools = llm

    def fix(self, request: FixRequest) -> FixAttempt:
        nid, n, max_n = request.node_id, request.attempt, request.max_attempts
        tools = RepoTools(request.repo_path, self.settings.read_max_lines)
        handlers = self._handlers(tools, request)
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(build_user_message(request)),
        ]
        log.info("[fix] %s attempt %d/%d: agent started", nid, n, max_n)
        started = time.monotonic()
        usage = {"input": 0, "output": 0}
        max_steps = self.settings.max_agent_steps
        finished: FinishArgs | None = None
        steps = 0
        while steps < max_steps:
            steps += 1
            ai = invoke_or_fatal(self.llm_with_tools, messages)
            if not isinstance(ai, AIMessage):
                ai = AIMessage(content=getattr(ai, "content", str(ai)))
            messages.append(ai)
            meta = ai.usage_metadata or {}
            usage["input"] += int(meta.get("input_tokens", 0) or 0)
            usage["output"] += int(meta.get("output_tokens", 0) or 0)

            calls = list(ai.tool_calls)
            for bad in ai.invalid_tool_calls:
                log.debug("[agent] invalid tool call %s: %s", bad.get("name"), bad.get("error"))
                messages.append(
                    ToolMessage(
                        f"Error: could not parse arguments for tool {bad.get('name')!r}: "
                        f"{bad.get('error') or 'invalid JSON'}",
                        tool_call_id=bad.get("id") or "invalid",
                        status="error",
                    )
                )
            if not calls:
                if not ai.invalid_tool_calls:
                    messages.append(HumanMessage(NUDGE_MESSAGE))
                continue
            try:
                for call in calls:
                    messages.append(self._execute(call, handlers))
            except _Finished as done:
                finished = done.args_
                break

        files = sorted(tools.files_touched)
        self._log_usage(nid, n, max_n, steps, usage, started)
        if finished is None:
            outcome: Literal["changed", "unfixable", "no_change"] = (
                "changed" if files else "no_change"
            )
            explanation = f"Agent stopped after {max_steps} steps without finishing; " + (
                "its edits are kept for verification." if files else "no changes were made."
            )
            log.info("[fix] %s attempt %d/%d: %s", nid, n, max_n, explanation)
        else:
            outcome = finished.outcome
            explanation = finished.explanation.strip()
            if outcome == "changed" and not files:
                outcome = "no_change"
            first = explanation.splitlines()[0] if explanation else ""
            log.info(
                "[fix] %s attempt %d/%d: agent finished (%s) %s", nid, n, max_n, outcome, first
            )
        return FixAttempt(
            node_id=nid,
            attempt=n,
            outcome=outcome,
            explanation=explanation,
            files_changed=files,
        )

    # ---- tool execution -----------------------------------------------------------------

    def _handlers(
        self, tools: RepoTools, request: FixRequest
    ) -> dict[str, Callable[[BaseModel], str]]:
        def run_test(args: BaseModel) -> str:
            assert isinstance(args, RunTestArgs)
            if args.node_id not in (None, "", request.node_id):
                return f"Error: only the target test can be run ({request.node_id})"
            if request.run_test is None:
                return "Error: running tests is not available in this run"
            result = request.run_test(request.node_id)
            details = result.details
            if len(details) > RUN_TEST_DETAILS_MAX_CHARS:
                details = details[-RUN_TEST_DETAILS_MAX_CHARS:]
            return (
                f"status: {result.status.value}\n"
                f"message: {result.message}\n"
                f"details (truncated): {details}"
            )

        def finish(args: BaseModel) -> str:
            assert isinstance(args, FinishArgs)
            raise _Finished(args)

        return {
            "list_files": lambda a: tools.list_files(**a.model_dump()),
            "read_file": lambda a: tools.read_file(**a.model_dump()),
            "search_code": lambda a: tools.search_code(**a.model_dump()),
            "edit_file": lambda a: tools.edit_file(**a.model_dump()),
            "create_file": lambda a: tools.create_file(**a.model_dump()),
            "run_test": run_test,
            "finish": finish,
        }

    def _execute(self, call: dict[str, Any], handlers: dict[str, Callable[[BaseModel], str]]):
        name = call.get("name") or ""
        call_id = call.get("id") or name or "call"
        raw_args = call.get("args") or {}
        log.debug("[agent] tool %s %s", name, _short(raw_args))
        spec = _TOOL_SPECS.get(name)
        if spec is None:
            content = f"Error: unknown tool {name!r}; available: {', '.join(_TOOL_SPECS)}"
            return ToolMessage(content, tool_call_id=call_id, status="error")
        try:
            args = spec[0].model_validate(raw_args)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}" for e in exc.errors()
            )
            content = f"Error: invalid arguments for {name}: {errors}"
            return ToolMessage(content, tool_call_id=call_id, status="error")
        try:
            content = handlers[name](args)
        except _Finished:
            raise
        except Exception as exc:  # tools never raise to the model
            log.debug("[agent] tool %s raised: %s", name, exc, exc_info=True)
            content = f"Error: {name} failed: {exc}"
        log.debug("[agent] tool %s -> %d chars", name, len(content))
        status = "error" if content.startswith("Error:") else "success"
        return ToolMessage(content, tool_call_id=call_id, status=status)

    @staticmethod
    def _log_usage(
        nid: str, n: int, max_n: int, steps: int, usage: dict[str, int], started: float
    ) -> None:
        log.info(
            "[fix] %s attempt %d/%d: %d step(s), %d input / %d output tokens, %.1fs",
            nid,
            n,
            max_n,
            steps,
            usage["input"],
            usage["output"],
            time.monotonic() - started,
        )
