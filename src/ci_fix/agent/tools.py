"""Repository tools for the fixer agent: list, read, search, edit and create files.

Pure Python (no LLM). Every path is repo-relative and confined to the checkout: absolute
paths, anything that resolves outside the repo (also via symlinks) and anything inside
``.git`` are refused. Tools never raise to the model; problems come back as ``Error: ...``
strings.
"""

from __future__ import annotations

import fnmatch
import functools
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from ci_fix.logging_setup import get_logger
from ci_fix.tools.git import GitError, run_git

log = get_logger(__name__)

MAX_LIST_ENTRIES = 300
MAX_SEARCH_MATCHES = 100
MAX_FILE_BYTES = 1_000_000
MAX_MATCH_LINE_CHARS = 300
EDIT_CONTEXT_LINES = 3


class ToolError(Exception):
    """A tool problem reported to the model as an ``Error: ...`` string."""


def _never_raise(method: Callable[..., str]) -> Callable[..., str]:
    """Turn any unexpected exception into an ``Error: ...`` string for the model."""

    @functools.wraps(method)
    def wrapper(*args: object, **kwargs: object) -> str:
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            log.debug("[agent] %s raised: %s", method.__name__, exc, exc_info=True)
            return f"Error: {method.__name__} failed: {exc}"

    return wrapper


class RepoTools:
    """File tools restricted to ``repo_path``; records every file it edits or creates."""

    def __init__(self, repo_path: Path, read_max_lines: int) -> None:
        self.repo_path = Path(repo_path)
        self.root = self.repo_path.resolve()
        self.read_max_lines = read_max_lines
        self.files_touched: set[str] = set()

    # ---- path handling ----------------------------------------------------------------------

    def _resolve(self, path: str) -> tuple[Path, str]:
        """Return (absolute resolved path, repo-relative posix path) or raise ToolError."""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path must be a non-empty repo-relative path")
        raw = path.strip()
        if "\0" in raw:
            raise ToolError("path must not contain NUL bytes")
        if PurePosixPath(raw).is_absolute() or Path(raw).is_absolute() or raw.startswith("~"):
            raise ToolError(f"absolute paths are not allowed: {raw!r}; use a repo-relative path")
        try:
            target = (self.root / raw).resolve()
        except (OSError, ValueError, RuntimeError) as exc:
            raise ToolError(f"invalid path {raw!r}: {exc}") from None
        if target != self.root and not target.is_relative_to(self.root):
            raise ToolError(f"path is outside the repository: {raw!r}")
        rel = target.relative_to(self.root)
        if ".git" in rel.parts:
            raise ToolError(f"access to .git is not allowed: {raw!r}")
        return target, rel.as_posix()

    def _git(self, *args: str) -> str:
        return run_git(args, cwd=self.root)

    @staticmethod
    def _error(exc: Exception) -> str:
        return f"Error: {exc}"

    # ---- tools ------------------------------------------------------------------------------

    @_never_raise
    def list_files(self, path: str = ".", pattern: str | None = None) -> str:
        """Tracked and untracked (not ignored) files under ``path``, optionally fnmatch-filtered."""
        try:
            target, rel = self._resolve(path)
            if not target.exists():
                raise ToolError(f"no such path: {path!r}")
            spec = rel if rel != "." else "."
            out = self._git(
                "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", spec
            )
        except (ToolError, GitError) as exc:
            return self._error(exc)
        files = sorted({f for f in out.split("\0") if f and (self.root / f).exists()})
        if pattern:
            files = [
                f
                for f in files
                if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(PurePosixPath(f).name, pattern)
            ]
        if not files:
            return "No files found."
        shown = files[:MAX_LIST_ENTRIES]
        text = "\n".join(shown)
        if len(files) > len(shown):
            text += (
                f"\n[... truncated: showing {len(shown)} of {len(files)} files; "
                "narrow the path or pattern]"
            )
        return text

    def _read_text(self, target: Path, path: str) -> str:
        if not target.is_file():
            raise ToolError(f"no such file: {path!r}")
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(f"file too large to read ({size} bytes): {path!r}")
        data = target.read_bytes()
        if b"\0" in data[:8192]:
            raise ToolError(f"binary file: {path!r}")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"file is not valid UTF-8 text: {path!r}") from None

    @_never_raise
    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Numbered lines ``start_line..end_line`` (1-based, inclusive), capped per call."""
        try:
            target, rel = self._resolve(path)
            text = self._read_text(target, path)
            lines = text.splitlines()
            total = len(lines)
            if total == 0:
                return f"{rel}: (empty file)"
            start = max(1, int(start_line))
            if start > total:
                raise ToolError(f"start_line {start} is past the end of {rel} ({total} lines)")
            last = start + self.read_max_lines - 1
            end = min(total, last if end_line is None else min(int(end_line), last))
            if end < start:
                raise ToolError(f"end_line {end_line} is before start_line {start}")
        except (ToolError, OSError, ValueError, TypeError) as exc:
            return self._error(exc)
        width = max(5, len(str(end)))
        body = "\n".join(f"{n:>{width}}| {lines[n - 1]}" for n in range(start, end + 1))
        header = f"{rel} (lines {start}-{end} of {total})"
        if end < total:
            body += f"\n[... {total - end} more lines; continue with start_line={end + 1}]"
        return f"{header}\n{body}"

    @_never_raise
    def search_code(self, pattern: str, path: str = ".", regex: bool = False) -> str:
        """``git grep -n`` over tracked and untracked files (fixed string unless ``regex``)."""
        try:
            if not pattern:
                raise ToolError("pattern must not be empty")
            _, rel = self._resolve(path)
            mode = "-E" if regex else "-F"
            out = self._git("grep", "-n", "-I", "--untracked", mode, "-e", pattern, "--", rel)
        except ToolError as exc:
            return self._error(exc)
        except GitError as exc:
            if "(exit 1)" in str(exc):  # git grep: no matches
                return "No matches."
            return self._error(exc)
        matches = [m for m in out.splitlines() if m]
        if not matches:
            return "No matches."
        shown = [
            m if len(m) <= MAX_MATCH_LINE_CHARS else m[:MAX_MATCH_LINE_CHARS] + " [...]"
            for m in matches[:MAX_SEARCH_MATCHES]
        ]
        text = "\n".join(shown)
        if len(matches) > len(shown):
            text += (
                f"\n[... truncated: showing {len(shown)} of {len(matches)} matches; "
                "narrow the pattern or path]"
            )
        return text

    @_never_raise
    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Replace the single exact occurrence of ``old_str`` with ``new_str``."""
        try:
            target, rel = self._resolve(path)
            text = self._read_text(target, path)
            if not old_str:
                raise ToolError("old_str must not be empty")
            count = text.count(old_str)
            if count == 0:
                raise ToolError(f"old_str not found in {rel}")
            if count > 1:
                raise ToolError(
                    f"old_str found {count} times in {rel}; include more surrounding "
                    "context so it matches exactly once"
                )
            index = text.index(old_str)
            new_text = text[:index] + new_str + text[index + len(old_str) :]
            target.write_text(new_text, encoding="utf-8")
        except (ToolError, OSError) as exc:
            return self._error(exc)
        self.files_touched.add(rel)
        log.debug("[agent] edited %s", rel)
        first = text.count("\n", 0, index) + 1
        last = first + new_str.count("\n")
        return f"Edited {rel}.\n" + self._context(new_text, first, last)

    @_never_raise
    def create_file(self, path: str, content: str) -> str:
        """Create a new file (parents included); refuses to overwrite an existing path."""
        try:
            target, rel = self._resolve(path)
            if rel == "." or target.exists() or target.is_symlink():
                raise ToolError(f"{path!r} already exists; use edit_file to change it")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except (ToolError, OSError) as exc:
            return self._error(exc)
        self.files_touched.add(rel)
        log.debug("[agent] created %s", rel)
        return f"Created {rel} ({len(content.splitlines())} lines)."

    @staticmethod
    def _context(text: str, first: int, last: int) -> str:
        lines = text.splitlines()
        lo = max(1, first - EDIT_CONTEXT_LINES)
        hi = min(len(lines), last + EDIT_CONTEXT_LINES)
        width = max(5, len(str(hi)))
        return "\n".join(f"{n:>{width}}| {lines[n - 1]}" for n in range(lo, hi + 1))
