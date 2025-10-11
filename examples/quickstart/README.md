# Semantic Router Quickstart

> ⚠️ This is an initial skeleton for the 10-minute quickstart workflow. Content will be expanded in follow-up tasks.

## Goal
Provide a single command path (`make quickstart`) that prepares the router, runs a small evaluation, and surfaces a concise report so that new users can validate the system within 10 minutes.

## Structure
- `quickstart.sh` – orchestrates dependency checks, model downloads, and service startup.
- `quick-eval.sh` – executes a minimal benchmark run and captures results.
- `config-quickstart.yaml` – opinionated defaults for running the router locally.
- `sample-data/` – trimmed datasets used for fast evaluation.
- `templates/` – report and config templates shared by quickstart scripts.

## Next Steps
1. Teach the benchmark loader to honor `QUICKSTART_SAMPLE_ROOT` so local JSONL slices are used offline.
2. Add a Makefile target (`quickstart`) that chains router bootstrap and quick evaluation.
3. Create CI smoke tests that run the 10-minute flow with the trimmed datasets.

## Quick Evaluation
Run the standalone evaluator once the router is healthy. A typical flow looks like:

```bash
./examples/quickstart/quickstart.sh &   # starts router (Ctrl+C to stop)
./examples/quickstart/quick-eval.sh --dataset mmlu --samples 5 --mode router
```

The evaluation script will place raw artifacts under `examples/quickstart/results/<timestamp>/raw` and derive:
- `quickstart-summary.csv` – compact metrics table for spreadsheets or dashboards.
- `quickstart-report.md` – Markdown summary suitable for PRs or runbooks.

Key flags:
- `--mode router|vllm|both` to toggle which side runs.
- `--samples` to tune runtime vs. statistical confidence.
- `--output-dir` for custom destinations (defaults to timestamped folder).
- All settings also respect `QUICKSTART_*` environment overrides.

## Local Sample Data
The `sample-data/` directory now includes trimmed JSONL slices for quick runs:
- `mmlu-sample.jsonl` – 10 multi-category academic questions.
- `arc-sample.jsonl` – 10 middle-school science questions with ARC-style options.

Each record follows the same schema that the benchmark loader expects (`question_id`, `category`, `question`, `options`, `answer`, optional `cot_content`). Sizes stay under 10 KB per file so the quickstart remains lightweight.

**Integration hook**: upcoming work will extend `bench/vllm_semantic_router_bench` to read from these JSONL files whenever `QUICKSTART_SAMPLE_ROOT` is set (falling back to Hugging Face datasets otherwise). Keep the files committed and deterministic so that the automated 10-minute flow can depend on them once the loader change lands.


