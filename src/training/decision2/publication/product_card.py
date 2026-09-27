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
    method = _text(card.method, "method")
    evaluation_scope = _text(card.evaluation_scope, "evaluation scope")
    uses = _bullets(card.use_cases, "use case")
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

## Where it helps

{uses}

## Try Choice, Noul and Score

Download the model and run its included native decision runtime on a BF16-capable GPU:

```bash
hf download {card.model_id} --local-dir {name}
```

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

Choice returns a selected candidate and probabilities. Noul returns a probability
for a yes/no proposition. Score returns an expected ordered level and its
distribution. Answers are conditional on the supplied state and criteria; the
model does not retrieve missing evidence.

## Measured results

The table compares models evaluated with the same JevArena v3 protocol and a
separate rerun of the **JevBench v1.2 public 231-question panel**. Its raw
accuracy is not the upstream multi-axis score or closed benchmark. JevArena
uses 8,147 original items; its
headline combines typed decision accuracy and the median of 15 human-labeled
transfer tasks. Invalid and missing answers count as failures.

{evaluation_scope}

{table}

{chart_lines}

Ranks apply only to the displayed same-panel roster. See
[evaluation methods and hashes](evaluation/EVALUATION.md) and the
[public manifest](evaluation/manifest.json) for panel versions and scope.

## How it was made

{method}

The direct starting weights are
[{card.direct_weight_source}](https://huggingface.co/{card.direct_weight_source})
at revision `{card.source_revision}`. The native inference package loads
**{card.loaded_parameters:,} parameters**, including its decision head.
{teacher}See [attributions](ATTRIBUTIONS.md) and [license](LICENSE).

## Limits

{limits}
"""
