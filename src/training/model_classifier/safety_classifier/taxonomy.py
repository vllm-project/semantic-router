"""Explicit source-to-output taxonomy contract for safety reconstruction.

The source taxonomies are intentionally kept separate from the stable model
labels.  New source categories must be reviewed and added here; callers never
silently discard a category they do not understand.
"""

from __future__ import annotations

from dataclasses import dataclass

TAXONOMY_VERSION = "legacy-9-v1"

LEVEL1_LABELS = ("safe", "unsafe")
LEVEL1_LABEL_TO_ID = {label: label_id for label_id, label in enumerate(LEVEL1_LABELS)}
LEVEL1_ID_TO_LABEL = {label_id: label for label, label_id in LEVEL1_LABEL_TO_ID.items()}

LEVEL2_LABELS = (
    "S1_violent_crimes",
    "S2_nonviolent_crimes",
    "S3_sex_crimes",
    "S5_weapons_cbrne",
    "S6_self_harm",
    "S7_hate",
    "S8_specialized_advice",
    "S9_privacy",
    "S13_misinformation",
)
LEVEL2_LABEL_TO_ID = {label: label_id for label_id, label in enumerate(LEVEL2_LABELS)}
LEVEL2_ID_TO_LABEL = {label_id: label for label, label_id in LEVEL2_LABEL_TO_ID.items()}


AEGIS_CATEGORY_TO_LABEL = {
    "Violence": "S1_violent_crimes",
    "Threat": "S1_violent_crimes",
    "Criminal Planning/Confessions": "S1_violent_crimes",
    "Controlled/Regulated Substances": "S2_nonviolent_crimes",
    "Fraud/Deception": "S2_nonviolent_crimes",
    "Illegal Activity": "S2_nonviolent_crimes",
    "Malware": "S2_nonviolent_crimes",
    "Sexual": "S3_sex_crimes",
    "Sexual (minor)": "S3_sex_crimes",
    "Guns and Illegal Weapons": "S5_weapons_cbrne",
    "Suicide and Self Harm": "S6_self_harm",
    "Hate/Identity Hate": "S7_hate",
    "Harassment": "S7_hate",
    "Unauthorized Advice": "S8_specialized_advice",
    "PII/Privacy": "S9_privacy",
    "Political/Misinformation/Conspiracy": "S13_misinformation",
}

# These AEGIS categories are known, but deliberately outside the legacy
# nine-class output contract.  Keeping the list explicit distinguishes an
# intentional exclusion from an unreviewed source-taxonomy change.
EXCLUDED_AEGIS_CATEGORIES = frozenset(
    {
        "Needs Caution",
        "Profanity",
        "Other",
        "Immoral/Unethical",
        "Copyright/Trademark/Plagiarism",
        "High Risk Gov Decision Making",
        "Manipulation",
    }
)

SYNTH_CATEGORY_TO_LABEL = {
    "S2_non_violent_crimes": "S2_nonviolent_crimes",
    "S6_specialized_advice": "S8_specialized_advice",
    "S7_privacy": "S9_privacy",
    "S9_indiscriminate_weapons": "S5_weapons_cbrne",
    "S11_suicide_self_harm": "S6_self_harm",
    "S13_elections": "S13_misinformation",
}


class UnknownCategoryError(ValueError):
    """Raised when a source dataset introduces an unreviewed category."""


@dataclass(frozen=True)
class TaxonomyMapping:
    """The deterministic legacy mapping for one AEGIS category string."""

    raw_categories: tuple[str, ...]
    mapped_targets: tuple[str, ...]
    selected_label: str | None

    @property
    def is_multilabel(self) -> bool:
        """Whether the source row declares more than one raw category."""
        return len(self.raw_categories) > 1

    @property
    def is_multitarget(self) -> bool:
        """Whether source categories map to multiple legacy output labels."""
        return len(self.mapped_targets) > 1


def parse_aegis_categories(value: str) -> tuple[str, ...]:
    """Parse the comma-delimited AEGIS field while preserving source order."""
    if not isinstance(value, str):
        raise TypeError("AEGIS violated_categories must be a string")
    return tuple(category.strip() for category in value.split(",") if category.strip())


def map_aegis_categories(value: str) -> TaxonomyMapping:
    """Map AEGIS categories, selecting the first mapped target in source order."""
    raw_categories = parse_aegis_categories(value)
    known_categories = set(AEGIS_CATEGORY_TO_LABEL) | EXCLUDED_AEGIS_CATEGORIES
    unknown = tuple(
        category for category in raw_categories if category not in known_categories
    )
    if unknown:
        joined = ", ".join(dict.fromkeys(unknown))
        raise UnknownCategoryError(f"Unknown AEGIS category: {joined}")

    mapped_targets: list[str] = []
    for category in raw_categories:
        target = AEGIS_CATEGORY_TO_LABEL.get(category)
        if target is not None and target not in mapped_targets:
            mapped_targets.append(target)

    targets = tuple(mapped_targets)
    return TaxonomyMapping(
        raw_categories=raw_categories,
        mapped_targets=targets,
        selected_label=targets[0] if targets else None,
    )


def map_synth_category(category: str) -> str:
    """Map one synthetic-data category or fail on taxonomy drift."""
    if not isinstance(category, str):
        raise TypeError("Synthetic category must be a string")
    try:
        return SYNTH_CATEGORY_TO_LABEL[category]
    except KeyError as exc:
        raise UnknownCategoryError(f"Unknown synthetic category: {category}") from exc
