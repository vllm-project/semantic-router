# RAG relevance comparison

Compare retrieval order, the existing cross-encoder, and the opt-in SystemOne
scorer on identical candidates. This driver measures ranking quality and
management API latency. It does not measure answer quality or full RAG latency.
No real-model quality result is included with this driver.

Start the Router through the repository's local-image `vllm-sr serve` flow.
Configure two reachable recipes with an enabled vectorstore RAG rerank plugin:
`cross-encoder` binds `rag.reranker` with `relevance_scores.v1`, while `decision`
binds it with `decision.v1`. Pin both model revisions and record the image,
commit, hardware, deployment input budgets and candidate generation procedure.
The diagnostic endpoint calls those prepared bindings without repeating search.

Supply a JSONL file with one query per line. Candidate order must be the original
retrieval order. IDs must be unique within a query; relevance is an integer from
0 (irrelevant) to 3 (directly relevant). For example, this is a format example,
not a quality dataset:

```json
{"id":"library-hours","query":"When is the library open?","candidates":[{"id":"hours","text":"The library opens on weekdays.","relevance":3},{"id":"parking","text":"Parking is free on Sundays.","relevance":0}]}
```

Use held-out labeled queries, including hard negatives and long documents.
Do not tune the question on the reported evaluation set. Keep the same candidate
pool and `top_k` for all three methods. The first slice accepts up to 32 candidates.

```bash
python3 bench/rag_relevance/evaluate.py \
  --url http://localhost:8080 \
  --dataset /path/to/held-out-candidates.jsonl \
  --cross-encoder-recipe cross-encoder --decision-recipe decision \
  --top-k 3 --out /tmp/rag-relevance-results.json
```

If management API authentication is enabled, set `ROUTER_API_TOKEN` in the
process environment. The output contains scores and binding metadata but no
query or document text, authorization header or endpoint URL. Review model
artifact identifiers before publishing the report.

The report includes the dataset digest, per-query rankings, nDCG@k, and p50/p95
latency. Queries with no relevant candidate are counted but excluded from mean
nDCG. Any failed or incomplete inference aborts the run instead of silently
excluding difficult queries. Latency includes transport and any runtime cache
hits. Run against freshly started deployments for a cold-cache comparison;
report repeated runs separately. Reported input tokens are not a monetary cost
estimate or a count of model forwards.

```bash
python3 -m unittest discover -s bench/rag_relevance -p 'test_*.py'
```
