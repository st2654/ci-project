"""Deliver the accepted fixes: squash commit, push, open/update the fix PR, comment.

Runs as the ``deliver`` graph node, after ``finalize`` and before the workspace is removed.

* Nothing FIXED → no commit, branch, push or PR. With ``push`` and ``comment_on_pr`` the
  original PR gets a comment saying why (unfixable / not found reasons).
* Otherwise the checkpoint commits are squashed onto the PR head (``git reset --soft``) into
  one commit with the built message and the configured identity (hooks off, no signing);
  the checkpoints stay reachable as ``refs/ci-fix/checkpoints``.
* ``push = false`` (dry run) stops here: the commit, message and PR body are in the result.
* Same-repo PR → force-push ``<branch_prefix><N>`` to ``origin`` (the tool's own branch).
  Fork PR → push the same branch name to the fork only if ``maintainer_can_modify`` and the
  fork's clone URL is known; otherwise a warning and no push/PR/comment.
* Guards before any push: the fix branch must not be the PR's head or base branch or the
  target repo's default branch, and an existing remote branch is force-updated only if its
  tip is a ci-fix commit (author email = ``commit_author_email``, subject ``ci-fix:...``).
* The PR head is re-read right before the push; if it moved, the push still happens but a
  warning is added and the PR body starts with a "re-run ci-fix" line.
* The fix PR targets the original PR's branch (in the fork for fork PRs). An open fix PR
  from the same branch is updated (title + body) instead of opening a second one.
* The comment on the original PR carries ``COMMENT_MARKER``; a re-run edits that comment.
  A failing comment only adds a warning: the fix PR itself is already delivered.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ci_fix.config import Settings
from ci_fix.logging_setup import get_logger
from ci_fix.models import OutcomeStatus
from ci_fix.report import (
    COMMENT_MARKER,
    ReportData,
    build_fix_pr_comment,
    build_no_fix_comment,
    build_pr_title,
    cap_text,
)
from ci_fix.tools.git import GitError, GitRepo
from ci_fix.tools.github import GitHubClient, GitHubError, RepoRef
from ci_fix.workspace import PreparedRepo

log = get_logger(__name__)

FORK_NO_PUSH_WARNING = (
    "fork PR: cannot push (maintainer edits not allowed); returning the diff only"
)
# The local checkpoint commits stay reachable here after the squash (keep_workspace).
CHECKPOINTS_REF = "refs/ci-fix/checkpoints"
NOTHING_TO_COMMIT_WARNING = "tests were fixed but there was nothing to commit"


class DeliveryOutcome(BaseModel):
    commit_sha: str | None = None
    pr_body: str = ""  # the body actually sent (may carry a head-moved line)
    pushed: bool = False
    pr_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


def push_target(prepared: PreparedRepo) -> tuple[str, RepoRef] | None:
    """``(remote name or URL, repo the fix PR goes to)``; None = a fork we may not push to."""
    pr = prepared.pr
    if not pr.is_fork:
        return "origin", prepared.repo
    if not (pr.maintainer_can_modify and pr.head_clone_url and pr.head_repo_full_name):
        return None
    owner, _, name = pr.head_repo_full_name.partition("/")
    return pr.head_clone_url, RepoRef(owner=owner, name=name)


def head_moved_line(old_sha: str, head_ref: str, new_sha: str) -> str:
    return (
        f"⚠ verified against `{old_sha[:12]}`; `{head_ref}` has since moved to "
        f"`{new_sha[:12]}` — re-run ci-fix"
    )


def _check_branch_is_ours(
    repo: GitRepo,
    remote: str,
    branch: str,
    target_repo: RepoRef,
    protected: set[str],
    settings: Settings,
    token: str | None,
) -> None:
    """Refuse to (force-)push to a branch ci-fix does not own (``GitHubError``)."""
    if branch in protected:
        raise GitHubError(
            f"refusing to push: fix branch {branch} is the PR's head/base branch or the "
            f"default branch of {target_repo.full_name}"
        )
    try:
        if not repo.remote_branch_exists(remote, branch, token):
            return
        tip = repo.fetch_remote_branch(remote, branch, token)
        email, subject = repo.commit_author_and_subject(tip)
    except GitError as exc:
        msg = f"cannot push {branch} to {target_repo.full_name}: reading the remote failed: {exc}"
        if any(s in str(exc).lower() for s in ("403", "permission", "denied", "authentication")):
            msg += f"\nhint: the token needs Contents: write on {target_repo.full_name}"
        raise GitError(msg) from None
    if email != settings.commit_author_email or not subject.startswith("ci-fix:"):
        raise GitHubError(
            f"branch {branch} exists on {target_repo.full_name} and was not created by "
            f"ci-fix (tip {tip[:12]} by {email}); refusing to overwrite it"
        )
    log.info(
        "[deliver] branch %s already exists on %s; force-updating it",
        branch,
        target_repo.full_name,
    )


def _comment(
    github: GitHubClient, prepared: PreparedRepo, body: str, outcome: DeliveryOutcome
) -> None:
    """Create or edit (marker) the ci-fix comment on the original PR; errors → warning."""
    repo, number = prepared.repo, prepared.pr.number
    try:
        existing = github.find_comment(repo, number, COMMENT_MARKER)
        if existing is not None:
            github.edit_comment(repo, number, existing, body)
            log.info("[deliver] updated the ci-fix comment on #%d", number)
        else:
            github.create_comment(repo, number, body)
            log.info("[deliver] commented on #%d", number)
    except GitHubError as exc:
        warning = f"could not comment on #{number}: {exc}"
        log.warning("[deliver] %s", warning)
        outcome.warnings.append(warning)


def deliver(
    repo: GitRepo,
    prepared: PreparedRepo,
    settings: Settings,
    github: GitHubClient,
    data: ReportData,
    commit_message: str,
    pr_body: str,
) -> DeliveryOutcome:
    """Commit, push, open/update the fix PR and comment, as configured (see module doc).

    Push and PR failures raise ``GitError``/``GitHubError`` (with a token-scope hint).
    """
    outcome = DeliveryOutcome(pr_body=pr_body)
    pr = prepared.pr
    if not any(t.status == OutcomeStatus.FIXED for t in data.tests):
        log.info("[deliver] nothing fixed: no commit, push or PR")
        if settings.push and settings.comment_on_pr:
            _comment(github, prepared, build_no_fix_comment(data), outcome)
        return outcome

    sha = repo.squash(
        prepared.pr_head_sha,
        commit_message,
        settings.commit_author_name,
        settings.commit_author_email,
        keep_ref=CHECKPOINTS_REF,
    )
    if sha is None:
        log.warning("[deliver] %s", NOTHING_TO_COMMIT_WARNING)
        outcome.warnings.append(NOTHING_TO_COMMIT_WARNING)
        return outcome
    outcome.commit_sha = sha
    log.info("[deliver] squashed the accepted fixes into commit %s", sha[:12])
    if not settings.push:
        log.info("[deliver] dry run: not pushing")
        return outcome

    target = push_target(prepared)
    if target is None:
        log.warning("[deliver] %s", FORK_NO_PUSH_WARNING)
        outcome.warnings.append(FORK_NO_PUSH_WARNING)
        return outcome
    remote, target_repo = target
    branch = prepared.branch
    token = settings.github_token.get_secret_value() if settings.github_token else None
    protected = {pr.head_ref, pr.base_ref, github.get_default_branch(target_repo)}
    _check_branch_is_ours(repo, remote, branch, target_repo, protected, settings, token)

    current = github.get_pull_request(prepared.repo, pr.number)
    if current.head_sha != prepared.pr_head_sha:
        line = head_moved_line(prepared.pr_head_sha, pr.head_ref, current.head_sha)
        warning = (
            f"PR head moved during the run: verified against {prepared.pr_head_sha[:12]}, "
            f"{pr.head_ref} is now at {current.head_sha[:12]} — re-run ci-fix"
        )
        log.warning("[deliver] %s", warning)
        outcome.warnings.append(warning)
        pr_body = cap_text(f"{line}\n\n{pr_body}")
    outcome.pr_body = pr_body
    repo.push(remote, branch, branch, token, force=True, label=target_repo.full_name)
    outcome.pushed = True
    log.info("[deliver] pushed branch %s to %s", branch, target_repo.full_name)

    title = build_pr_title(pr.number, pr.title)
    existing = github.find_open_pr(
        target_repo, head=f"{target_repo.owner}:{branch}", base=pr.head_ref
    )
    if existing is not None:
        ref = github.edit_pull(target_repo, existing.number, title, pr_body)
        log.info("[deliver] updated existing fix PR %s", ref.html_url)
    else:
        ref = github.create_pull(target_repo, title, pr_body, head=branch, base=pr.head_ref)
        log.info("[deliver] opened fix PR %s", ref.html_url)
    outcome.pr_url = ref.html_url

    if settings.comment_on_pr:
        _comment(github, prepared, build_fix_pr_comment(ref.html_url, data), outcome)
    return outcome
