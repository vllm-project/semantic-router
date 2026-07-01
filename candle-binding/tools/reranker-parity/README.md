<!-- markdownlint-disable -->
# Cross-encoder reranker parity oracle

`verify_parity.py` scores the same `(query, document)` cases used by the Go test
(`candle-binding/cross_encoder_live_test.go`) with HuggingFace `transformers`, and
is the ground truth for the candle `BertCrossEncoder` implementation.

It is a developer tool, **not** part of CI: it needs `torch` + `transformers`,
which the Go/Rust binding tests deliberately avoid. Instead, run it once to
(re)generate the committed golden file, and the env-gated Go test
`TestCrossEncoderRerankMatchesTransformersGolden` asserts the candle scores match
that golden within tolerance, with no Python at test time.

## Usage

```bash
pip install -r requirements.txt

# print a human-readable parity table
python verify_parity.py

# (re)generate the committed golden file the Go parity test reads
python verify_parity.py --emit-golden ../../testdata/reranker_parity_golden.json
```

Regenerate the golden file whenever the canonical model or the test cases change,
and keep the cases in sync with `rerankCases` in `cross_encoder_live_test.go`.
