# Sample Data

Trimmed datasets for the quickstart flow live here. They mirror the schema consumed by
`bench/vllm_semantic_router_bench` so we can swap them in for the heavy upstream corpora.

## Available slices
- `mmlu-sample.jsonl` – 10 representative questions spanning multiple disciplines.
- `arc-sample.jsonl` – 10 ARC-style science questions with multiple-choice options.

Each line is a JSON object with the following fields:
- `question_id`: stable identifier used in reports.
- `category`: benchmark category or sub-domain.
- `question`: user prompt text.
- `options`: either a list (MMLU) or mapping (ARC) of answer choices.
- `answer`: canonical correct answer (letter).
- `cot_content` *(optional)*: short reasoning snippet for CoT prompts.

Files intentionally stay under 20 KB so they can ship with the repository.

## Integration plan
`quick-eval.sh` will soon surface a `QUICKSTART_SAMPLE_ROOT` override. When present,
the benchmark runner will load these JSONL files instead of fetching full datasets
from Hugging Face. Until that wire-up lands, the JSONL fixtures exist here to unblock
concurrent work (documentation, UI previews, and benchmarking harness updates).
