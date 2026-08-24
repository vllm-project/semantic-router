"""Pure policy rules for community issue and pull-request lifecycle state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NEEDS_ACCEPTANCE = "needs-acceptance"
ACCEPTED = "accepted"
READY_FOR_DEV = "ready-for-dev"
IN_PROGRESS = "in-progress"
HELP_WANTED = "help wanted"
GOOD_FIRST_ISSUE = "good first issue"
RELEASE_BLOCKER = "release-blocker"
BACKLOG = "backlog"
CLOSE_CANDIDATE = "close-candidate"
STALE = "stale"

WORKGROUP_LABELS = (
    "wg/mom-routing",
    "wg/router-models-inference-runtime",
    "wg/data-plane-networking",
    "wg/enterprise-environment",
    "wg/agentic-context",
    "wg/developer-experience-ecosystem",
    "wg/evaluation-quality",
)
MAINTAINER_OWNER = "owner/maintainers"
OWNER_LABELS = (*WORKGROUP_LABELS, MAINTAINER_OWNER)

WORKGROUP_OPTIONS = {
    "MoM & Routing": "wg/mom-routing",
    "Router Models & Inference Runtime": "wg/router-models-inference-runtime",
    "Data Plane & Networking": "wg/data-plane-networking",
    "Enterprise & Environment": "wg/enterprise-environment",
    "Agentic & Context": "wg/agentic-context",
    "Developer Experience & Ecosystem": "wg/developer-experience-ecosystem",
    "Evaluation & Quality": "wg/evaluation-quality",
}

PR_STATE_LABELS = (
    "pr/needs-review",
    "pr/needs-author",
    "pr/needs-rebase",
    "pr/blocked",
    "pr/merge-ready",
    "pr/close-candidate",
)
PR_CLOSE_CANDIDATE = "pr/close-candidate"

PRIORITY_LABELS = ("priority/P0", "priority/P1", "priority/P2")
PROTECTED_ISSUE_LABELS = {
    ACCEPTED,
    READY_FOR_DEV,
    HELP_WANTED,
    GOOD_FIRST_ISSUE,
    RELEASE_BLOCKER,
    BACKLOG,
    CLOSE_CANDIDATE,
}
MAINTAINER_PERMISSIONS = {"admin", "maintain", "write"}
API_PAGE_SIZE = 100

RELATED_PATTERN = re.compile(
    r"\brelated\b\s*:?\s*(?:(?:https?://github\.com/)?"
    r"(?P<repo>[\w.-]+/[\w.-]+)(?:/issues/|#)|#)(?P<number>\d+)",
    re.IGNORECASE,
)
TITLE_PREFIX_PATTERN = re.compile(
    r"^\[(?P<category>[^\[\]\r\n]+)\]\s+(?P<summary>\S.*)$"
)


def title_format_error(title: str | None) -> str | None:
    """Return a concise error when a public work-item title is not normalized."""

    match = TITLE_PREFIX_PATTERN.match(title or "")
    if (
        not match
        or not match.group("category").strip()
        or match.group("summary").startswith("[")
    ):
        return (
            "Title must begin with exactly one bracketed category and a summary, "
            "for example `[Feature] Add standalone serving`; do not use `feat:` "
            "or stacked prefixes such as `[Router][Docs]`."
        )
    return None


def label_names(item: dict[str, Any]) -> set[str]:
    return {
        label["name"] if isinstance(label, dict) else str(label)
        for label in item.get("labels", [])
    }


def parse_form_field(body: str | None, heading: str) -> str | None:
    if not body:
        return None
    pattern = re.compile(
        rf"^###\s+{re.escape(heading)}\s*$\s*^([^\n]+)",
        re.MULTILINE,
    )
    match = pattern.search(body)
    if not match:
        return None
    value = match.group(1).strip()
    return value if value and value != "_No response_" else None


def proposed_workgroup(body: str | None) -> str | None:
    value = parse_form_field(body, "Proposed Workgroup")
    return WORKGROUP_OPTIONS.get(value or "")


def extract_related_issue_numbers(
    title: str | None,
    body: str | None,
    *,
    repository: str | None = None,
) -> set[int]:
    text = f"{title or ''}\n{body or ''}"
    return {
        int(match.group("number"))
        for match in RELATED_PATTERN.finditer(text)
        if repository is None
        or match.group("repo") is None
        or match.group("repo").casefold() == repository.casefold()
    }


@dataclass
class IssuePlan:
    add_labels: set[str] = field(default_factory=set)
    remove_labels: set[str] = field(default_factory=set)
    remove_assignees: set[str] = field(default_factory=set)
    clear_milestone: bool = False
    comments: list[tuple[str, str]] = field(default_factory=list)

    def add_comment(self, code: str, message: str) -> None:
        self.comments.append((code, message))


@dataclass(frozen=True)
class IssueAcceptanceEvaluation:
    valid: bool
    error: str | None = None
    owner_label: str | None = None


def evaluate_issue_acceptance(
    issue: dict[str, Any],
    *,
    actor_can_manage: bool,
) -> IssueAcceptanceEvaluation:
    """Validate the explicit ``/accept`` transition for one issue."""

    if not actor_can_manage:
        return IssueAcceptanceEvaluation(
            False,
            "`/accept` requires repository write, maintain, or admin permission.",
        )
    title_error = title_format_error(issue.get("title"))
    if title_error:
        return IssueAcceptanceEvaluation(False, title_error)
    owners = label_names(issue).intersection(OWNER_LABELS)
    if len(owners) != 1:
        return IssueAcceptanceEvaluation(
            False,
            "`/accept` requires exactly one recognized owner label: one `wg/*` "
            "label for project work or `owner/maintainers` for repository governance.",
        )
    return IssueAcceptanceEvaluation(True, owner_label=next(iter(owners)))


def guard_protected_label(
    plan: IssuePlan,
    labels: set[str],
    *,
    event_action: str,
    event_label: str | None,
    actor_can_manage: bool,
) -> None:
    protected_labels = PROTECTED_ISSUE_LABELS.union(OWNER_LABELS, PRIORITY_LABELS)
    if actor_can_manage or event_label not in protected_labels:
        return
    if event_action == "labeled":
        plan.remove_labels.add(event_label)
        labels.discard(event_label)
        action = "removed"
    elif event_action == "unlabeled":
        plan.add_labels.add(event_label)
        labels.add(event_label)
        action = "restored"
    else:
        return
    plan.add_comment(
        "protected-label",
        f"`{event_label}` is Maintainer-controlled and was {action}. "
        "Please request the transition in a comment.",
    )


def normalize_proposed_workgroup(
    plan: IssuePlan,
    labels: set[str],
    issue: dict[str, Any],
    *,
    accepted: bool,
) -> set[str]:
    workgroups = labels.intersection(WORKGROUP_LABELS)
    form_workgroup = proposed_workgroup(issue.get("body"))
    if accepted or not form_workgroup or workgroups == {form_workgroup}:
        return workgroups
    plan.remove_labels.update(workgroups - {form_workgroup})
    plan.add_labels.add(form_workgroup)
    labels.difference_update(workgroups)
    labels.add(form_workgroup)
    return {form_workgroup}


def normalize_unaccepted_issue(
    plan: IssuePlan,
    labels: set[str],
    issue: dict[str, Any],
    assignees: set[str],
) -> None:
    plan.add_labels.add(NEEDS_ACCEPTANCE)
    labels.add(NEEDS_ACCEPTANCE)
    plan.remove_labels.update(
        {
            READY_FOR_DEV,
            IN_PROGRESS,
            HELP_WANTED,
            GOOD_FIRST_ISSUE,
            RELEASE_BLOCKER,
            BACKLOG,
        }
    )
    plan.remove_labels.update(labels.intersection(PRIORITY_LABELS))
    plan.clear_milestone = issue.get("milestone") is not None
    if not assignees:
        return
    plan.remove_assignees.update(assignees)
    plan.add_comment(
        "assignment-before-acceptance",
        "Work cannot be assigned before Maintainer acceptance. The assignment "
        "was removed; use the Workgroup review to make the issue actionable.",
    )


def normalize_delivery_state(
    plan: IssuePlan,
    labels: set[str],
    *,
    accepted: bool,
    workgroups: set[str],
    assignees: set[str],
) -> None:
    if accepted and assignees:
        plan.add_labels.add(IN_PROGRESS)
        plan.remove_labels.update({READY_FOR_DEV, HELP_WANTED, GOOD_FIRST_ISSUE})
    elif accepted and not assignees and IN_PROGRESS in labels:
        plan.remove_labels.add(IN_PROGRESS)

    contributor_ready = (
        accepted
        and len(workgroups) == 1
        and READY_FOR_DEV in (labels | plan.add_labels)
        and not assignees
    )
    invalid_contributor_labels = labels.intersection({HELP_WANTED, GOOD_FIRST_ISSUE})
    if not contributor_ready and invalid_contributor_labels:
        plan.remove_labels.update(invalid_contributor_labels)
        plan.add_comment(
            "contributor-labels",
            "`help wanted` and `good first issue` require `accepted`, "
            "`ready-for-dev`, exactly one Workgroup owner, and no active assignee.",
        )

    if READY_FOR_DEV in labels and (not accepted or len(workgroups) != 1 or assignees):
        plan.remove_labels.add(READY_FOR_DEV)
    if IN_PROGRESS in labels and (not accepted or not assignees):
        plan.remove_labels.add(IN_PROGRESS)


def plan_issue(
    issue: dict[str, Any],
    *,
    event_action: str,
    event_label: str | None,
    actor_can_manage: bool,
) -> IssuePlan:
    """Return idempotent mutations for the current issue state."""

    plan = IssuePlan()
    labels = label_names(issue)
    assignees = {
        assignee.get("login", "")
        for assignee in issue.get("assignees", [])
        if assignee.get("login")
    }
    guard_protected_label(
        plan,
        labels,
        event_action=event_action,
        event_label=event_label,
        actor_can_manage=actor_can_manage,
    )

    accepted = ACCEPTED in labels
    workgroups = normalize_proposed_workgroup(plan, labels, issue, accepted=accepted)
    owners = labels.intersection(OWNER_LABELS)
    if accepted and len(owners) != 1:
        plan.remove_labels.add(ACCEPTED)
        plan.add_labels.add(NEEDS_ACCEPTANCE)
        labels.discard(ACCEPTED)
        labels.add(NEEDS_ACCEPTANCE)
        accepted = False
        plan.add_comment(
            "accepted-owner",
            "`accepted` requires exactly one recognized owner: one `wg/*` label "
            "for project work or `owner/maintainers` for repository governance. "
            "The issue has been returned to `needs-acceptance` for Maintainer triage.",
        )

    if accepted:
        plan.remove_labels.update({NEEDS_ACCEPTANCE, STALE})
        labels.discard(NEEDS_ACCEPTANCE)
    else:
        normalize_unaccepted_issue(plan, labels, issue, assignees)
    normalize_delivery_state(
        plan,
        labels,
        accepted=accepted,
        workgroups=workgroups,
        assignees=assignees,
    )
    plan.add_labels.difference_update(plan.remove_labels)
    return plan


@dataclass(frozen=True)
class PullRequestEvaluation:
    valid: bool
    errors: tuple[str, ...]
    state_label: str
    owner_label: str | None = None
    milestone_number: int | None = None
    release_blocker: bool = False


def select_pull_request_state(
    *,
    close_candidate: bool,
    draft: bool,
    valid: bool,
    review_decision: str | None,
    merge_state: str | None,
    checks_state: str | None,
) -> str:
    if close_candidate:
        return PR_CLOSE_CANDIDATE
    if draft:
        return "pr/needs-author"
    if not valid or checks_state in {"ERROR", "FAILURE"}:
        return "pr/blocked"
    if review_decision == "CHANGES_REQUESTED":
        return "pr/needs-author"
    if merge_state in {"BEHIND", "DIRTY"}:
        return "pr/needs-rebase"
    if review_decision != "APPROVED":
        return "pr/needs-review"
    if checks_state == "SUCCESS" and merge_state in {"CLEAN", "HAS_HOOKS"}:
        return "pr/merge-ready"
    return "pr/blocked"


def evaluate_pull_request(
    pull_request: dict[str, Any],
    linked_issues: list[dict[str, Any]],
    *,
    validation_succeeded: bool = True,
    review_decision: str | None = None,
    merge_state: str | None = None,
    checks_state: str | None = None,
) -> PullRequestEvaluation:
    user = pull_request.get("user") or {}
    errors: list[str] = []
    is_bot = user.get("type") == "Bot"
    draft = bool(pull_request.get("draft"))
    accepted_issues = (
        []
        if is_bot
        else [issue for issue in linked_issues if ACCEPTED in label_names(issue)]
    )
    requires_admission = not draft and not is_bot
    if requires_admission and not linked_issues:
        errors.append("Link a tracking issue with `Closes #123` or `Related #123`.")
    elif requires_admission and not accepted_issues:
        errors.append("At least one linked issue must carry the `accepted` label.")

    owner_labels = {
        label
        for issue in accepted_issues
        for label in label_names(issue)
        if label in OWNER_LABELS
    }
    if accepted_issues and len(owner_labels) != 1:
        errors.append(
            "Accepted linked work must resolve to exactly one recognized owner label."
        )

    milestone_numbers = {
        int(issue["milestone"]["number"])
        for issue in accepted_issues
        if issue.get("milestone") and issue["milestone"].get("number") is not None
    }
    if len(milestone_numbers) > 1:
        errors.append("Linked accepted issues disagree on the release milestone.")
    if requires_admission and not validation_succeeded:
        errors.append("The Community acceptance check did not complete successfully.")

    valid = not errors
    state_label = select_pull_request_state(
        close_candidate=PR_CLOSE_CANDIDATE in label_names(pull_request),
        draft=draft,
        valid=valid,
        review_decision=review_decision,
        merge_state=merge_state,
        checks_state=checks_state,
    )
    return PullRequestEvaluation(
        valid,
        tuple(errors),
        state_label,
        owner_label=(
            "wg/evaluation-quality"
            if is_bot
            else next(iter(owner_labels)) if len(owner_labels) == 1 else None
        ),
        milestone_number=(
            next(iter(milestone_numbers)) if len(milestone_numbers) == 1 else None
        ),
        release_blocker=any(
            RELEASE_BLOCKER in label_names(issue) for issue in accepted_issues
        ),
    )
