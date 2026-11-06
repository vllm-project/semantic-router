"""Choose one Workgroup reviewer for an accepted pull request.

The roster is the overlap of public Working Group leads and active Committers
or Maintainers. Keep it aligned with website/src/data/workGroups.ts and
website/src/data/teamMembers.tsx when those rosters change.
"""

from __future__ import annotations

from community_lifecycle_policy import WORKGROUP_LABELS

# Deterministic rotation spreads requests without needing a mutable cursor.
WORKGROUP_REVIEWERS: dict[str, tuple[str, ...]] = {
    "wg/mom-routing": ("Xunzhuo", "yaojiejia", "ramkrishs"),
    "wg/router-models-inference-runtime": (
        "WUKUNTAI-0211",
        "theohsiung",
        "subin9",
        "ramkrishs",
        "adaamko",
    ),
    "wg/data-plane-networking": (
        "drivebyer",
        "Xunzhuo",
        "abhinav-m22",
        "1fanwang",
        "Bevisy",
    ),
    "wg/enterprise-environment": (
        "AayushSaini101",
        "akshayv",
        "abhinav-m22",
    ),
    "wg/agentic-context": (
        "Xunzhuo",
        "AayushSaini101",
        "abhinav-m22",
        "yaojiejia",
        "1fanwang",
        "Bevisy",
    ),
    "wg/developer-experience-ecosystem": (
        "AayushSaini101",
        "wilsonwu",
        "abhinav-m22",
    ),
    "wg/evaluation-quality": ("Xunzhuo", "FAUST-BENCHOU"),
}

DEFAULT_REVIEWERS = frozenset({"rootfs", "Xunzhuo"})


def desired_workgroup_reviewer(
    owner_label: str | None,
    pull_number: int,
    author: str,
) -> str | None:
    """Select one stable eligible reviewer, excluding the PR author."""

    if owner_label not in WORKGROUP_LABELS:
        return None
    candidates = WORKGROUP_REVIEWERS[owner_label]
    offset = pull_number % len(candidates)
    ordered = candidates[offset:] + candidates[:offset]
    excluded = {author.casefold(), *(login.casefold() for login in DEFAULT_REVIEWERS)}
    for candidate in ordered:
        if candidate.casefold() not in excluded:
            return candidate
    for candidate in ordered:
        if candidate.casefold() != author.casefold():
            return candidate
    return None
