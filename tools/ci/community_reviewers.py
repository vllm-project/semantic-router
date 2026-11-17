"""Choose one Working Group Lead to review an accepted pull request.

Each roster lists the group's Leads who are Maintainers or Committers with
write access: approvals from other accounts do not count toward the two the
merge queue requires. Update a roster when a Lead joins or leaves a group or
their repository access changes.
"""

from __future__ import annotations

from community_lifecycle_policy import WORKGROUP_LABELS

WORKGROUP_REVIEWERS: dict[str, tuple[str, ...]] = {
    "wg/mom-routing": (
        "1fanwang",
        "abhinav-m22",
        "subin9",
        "WUKUNTAI-0211",
        "Xunzhuo",
    ),
    "wg/router-models-inference-runtime": (
        "1fanwang",
        "adaamko",
        "Bevisy",
        "Peterren",
        "PharosEast",
        "shraderdm",
        "subin9",
        "theohsiung",
        "TwilightTechie",
        "WUKUNTAI-0211",
        "Xunzhuo",
        "yaojiejia",
    ),
    "wg/data-plane-networking": ("1fanwang", "abhinav-m22", "Bevisy"),
    "wg/enterprise-environment": ("AayushSaini101",),
    "wg/agentic-context": (
        "1fanwang",
        "abhinav-m22",
        "Bevisy",
        "Peterren",
        "yaojiejia",
    ),
    "wg/developer-experience-ecosystem": (
        "1fanwang",
        "AayushSaini101",
        "abhinav-m22",
        "wilsonwu",
        "yaojiejia",
        "Zi-Yi-Ming",
    ),
    "wg/evaluation-quality": ("FAUST-BENCHOU", "Zi-Yi-Ming"),
}

# .github/CODEOWNERS requests these Maintainers on every pull request.
DEFAULT_REVIEWERS = frozenset({"rootfs", "Xunzhuo"})


def workgroup_reviewers(owner_label: str | None, author: str) -> tuple[str, ...]:
    """Return the owning group's Leads who may review, never the author.

    The repository-wide reviewers are already requested, so they are returned
    only when no other Lead remains.
    """

    if owner_label not in WORKGROUP_LABELS:
        return ()
    leads = tuple(
        login
        for login in WORKGROUP_REVIEWERS[owner_label]
        if login.casefold() != author.casefold()
    )
    defaults = {login.casefold() for login in DEFAULT_REVIEWERS}
    others = tuple(login for login in leads if login.casefold() not in defaults)
    return others or leads


def desired_workgroup_reviewer(
    owner_label: str | None,
    pull_number: int,
    author: str,
) -> str | None:
    """Rotate by PR number so every eligible Lead receives an equal share."""

    reviewers = workgroup_reviewers(owner_label, author)
    if not reviewers:
        return None
    return reviewers[pull_number % len(reviewers)]
