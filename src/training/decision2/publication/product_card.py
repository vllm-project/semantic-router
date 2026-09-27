"""Render a Decision 2.0 product card from scored release evidence.

This renderer deliberately does not open raw predictions or invent metrics. A
separate release audit must validate the supplied JevArena/JevBench table and
the downloadable package before its output can be published.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MODEL_ID = re.compile(r"llm-semantic-router/DEV2\.0-(?:0\.6|0\.8|2|4|9|27)B\Z")
SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
ASSET = re.compile(r"assets/[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:png|svg)\Z")
FIGURES = (
    "jevarena-rank.svg",
    "jevarena-task-matrix.svg",
    "jevbench-public-rank.svg",
)


@dataclass(frozen=True)
class ProductCard:
    model_id: str
    banner: str
    tagline: str
    measured_summary: str
    use_cases: tuple[str, ...]
    direct_weight_source: str
    source_revision: str
    loaded_parameters: int
    method: str
    limitations: tuple[str, ...]
    evidence_table: str
    evaluation_scope: str
    teacher_assisted: bool = False


def _text(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(mark in value for mark in ("\x00", "\r", "<script", "</script"))
    ):
        raise ValueError(f"{label} must be nonempty public text")
    return value.strip()


def _bullets(items: tuple[str, ...], label: str) -> str:
    if not isinstance(items, tuple) or not items:
        raise ValueError(f"{label} needs at least one item")
    return "\n".join(f"- {_text(item, label)}" for item in items)


def render(card: ProductCard) -> str:
    """Create a readable public README with accurate direct provenance.

    The supplied table must be generated from a matched v3/public231 panel;
    package/evidence verification is performed by the publication pipeline.
    """
    if not MODEL_ID.fullmatch(card.model_id):
        raise ValueError("Unexpected Decision 2.0 model ID")
    if not ASSET.fullmatch(card.banner):
        raise ValueError("Banner must be a local release asset")
    if not SOURCE_ID.fullmatch(card.direct_weight_source) or not SHA.fullmatch(
        card.source_revision
    ):
        raise ValueError("Direct weight source needs an exact repository revision")
    if type(card.loaded_parameters) is not int or card.loaded_parameters < 1:
        raise ValueError("Loaded parameter count must be measured")
    table = _text(card.evidence_table, "evidence table")
    if (
        "| JevArena rank | Model | Actual parameters |" not in table
        or "JevBench v1.2 public" not in table
        or "8,147" not in table
    ):
        raise ValueError("A scored v3 and public231 table is required")
    name = card.model_id.rsplit("/", 1)[1]
    tagline = _text(card.tagline, "tagline")
    measured_summary = _text(card.measured_summary, "measured summary")
    method = _text(card.method, "method")
    evaluation_scope = _text(card.evaluation_scope, "evaluation scope")
    if len(card.use_cases) != 3:
        raise ValueError("Choice, Noul and Score each need one use case")
    uses = "\n".join(
        f"| {kind} | {_text(purpose, 'use case')} | {result} |"
        for kind, purpose, result in zip(
            ("Choice", "Noul", "Score"),
            card.use_cases,
            (
                "Selected option and probabilities",
                "Probability of yes",
                "Expected level and probabilities",
            ),
            strict=True,
        )
    )
    limits = _bullets(card.limitations, "limitation")
    teacher = (
        "Teacher-assisted training was used; teacher identities and data terms "
        "are documented in the evaluation and attribution materials.\n\n"
        if card.teacher_assisted
        else ""
    )
    chart_labels = {
        "jevarena-rank.svg": "JevArena rank",
        "jevarena-task-matrix.svg": "JevArena model-by-task matrix",
        "jevbench-public-rank.svg": "JevBench v1.2 public 231 rank",
    }
    chart_lines = "\n\n".join(
        f"![{chart_labels[figure]}](assets/{figure})" for figure in FIGURES
    )
    return f"""---
license: apache-2.0
base_model: {card.direct_weight_source}
tags:
  - decision-model
  - classification
---

![{name} owl banner]({card.banner})

# {name}

{tagline}

Give it a state and the questions your application needs answered. Supply the
options or rubric at runtime; get structured answers and probabilities back.

## Measured decisions

{measured_summary}

The comparison uses the same **8,147-item JevArena v3** panel for every model.
The **231 public JevBench v1.2** questions are shown separately.

{table}

{chart_lines}

{evaluation_scope} Ranks cover only the models shown here.
[Full results and methods](evaluation/EVALUATION.md) ·
[Evaluation manifest](evaluation/manifest.json)

## Three ways to decide

| Decision | Use it to | Returns |
| --- | --- | --- |
{uses}

## Download and decide

Download the full model repository:

```bash
hf download {card.model_id} --local-dir {name}
```

Run a local System One request on a BF16-capable GPU:

```python
import os
import sys

sys.path.insert(0, os.path.abspath("{name}"))
from decision2 import Decision2

model = Decision2.from_pretrained("{name}", device="cuda:0")
result = model.system_one(
    state="The order arrived damaged yesterday. The customer has a receipt.",
    questions={{
        "route": {{
            "type": "choice",
            "instructions": "Who should handle this request?",
            "criteria": {{"returns": "Refund or replacement", "support": "Product help"}},
        }},
        "evidence": {{
            "type": "noul",
            "instructions": "Does the state say the customer has a receipt?",
        }},
        "urgency": {{
            "type": "score",
            "instructions": "Rate handling urgency from the stated facts.",
            "criteria": ["Routine", "Soon", "Immediate"],
        }},
    }},
)
print(result["answers"])
```

This repository is a model download, not a hosted API.

## Architecture

{method}

It loads **{card.loaded_parameters:,} parameters**, including the decision head.
The starting weights are
[{card.direct_weight_source}](https://huggingface.co/{card.direct_weight_source});
their exact revision is in the [evaluation manifest](evaluation/manifest.json).
{teacher}

### Limits

{limits}

The model uses the supplied evidence; it does not retrieve missing facts.
[License](LICENSE) · [Attributions](ATTRIBUTIONS.md)
"""
