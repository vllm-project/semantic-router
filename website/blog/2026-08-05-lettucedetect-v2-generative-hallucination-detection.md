---
slug: lettucedetect-v2-generative-hallucination-detection
title: "LettuceDetect v2 in Semantic Router: Generative Hallucination Detection as a vLLM Endpoint"
description: Span-level hallucination detection for code, tool output, and documents — served by vLLM, running natively in the Semantic Router response pipeline.
authors: [adaamko, boweihe, Xunzhuo, rootfs]
tags: [hallucination, lettucedetect, halugate, vllm, detection, semantic-router]
image: /img/blog/lettucedetect-v2-detection.png
---

Semantic Router can now verify grounded responses with a **generative span detector served by vLLM**. The new `endpoint` detector backend runs [LettuceDetect v2](https://github.com/KRLabsOrg/LettuceDetect) against every fact-checkable answer: unsupported spans are located to the character, typed against a hallucination taxonomy, and explained — in one call, before the response reaches the user.

The models come out of a joint paper between KR Labs and the Semantic Router team, *Beyond Document Grounding: Span-Level Hallucination Detection over Code, Tool Output, and Documents* ([arXiv:2607.00895](https://arxiv.org/abs/2607.00895)). This post walks through the paper — the benchmark, the taxonomy, the models, and what they score — and then through the integration that puts the detector into the serving stack.

![LettuceDetect v2 flagging contract hallucinations through Semantic Router](/img/blog/lettucedetect-v2-detection.png)

<!-- truncate -->

## Why a new detector

HaluGate established the shape of hallucination mitigation in Semantic Router: a sentinel decides whether a request needs fact-checking, a detector locates unsupported spans in the response, and an explainer classifies why each span is problematic. That architecture works — but its original token-classifier detector is recall-bound (the HaluGate post itself noted that nearly half of hallucinations were missed), and like most detectors it was trained on natural-language document evidence.

Real grounded systems increasingly answer from **structured inputs**: source code, developer-tool output, markdown, tables. In code and tool output, a single unsupported substring — a wrong field, a fabricated method name, a misreported value — can change program behavior while leaving the rest of the answer correct. A verifier should point to the unsupported substring, not just reject the answer. Prose-trained detectors are weakest exactly there, and as the paper shows, so are large zero-shot LLM judges.

## The benchmark: one span-level task across code, tools, and documents

There was no shared span-level benchmark that treats generated code, tool observations, and structured documents under the same verification task as classic document RAG. The paper builds one: **74,285 newly constructed span-labeled examples** across five new sources, plus converted examples from RAGTruth and the 14-language PsiloQA. The full training split contains 145,250 examples, with 6,171 validation and 10,698 test examples, split by grounding source so the test set only uses unseen repositories, papers, and articles.

| Source | Grounding | Built from |
|--------|-----------|------------|
| Code | repository files, gold fix | SWE-bench coding-agent traces |
| Tool output | verbose tool observations | Squeez (query + observation + gold lines) |
| ACL | retrieved paper chunks | ACL-Verbatim |
| README | project documentation | popular GitHub repositories |
| Wikipedia | markdown articles | open-wikipedia-markdown |
| RAGTruth / PsiloQA | documents | incorporated benchmarks |

The labeling method is what makes exact character offsets possible at this scale. Every example starts from a **grounded correct answer**; an injector model (Gemma 4 31B for code, Qwen 3.6 35B for tool output and markdown) proposes a small, localized hallucination as a structured edit, and the exact character span is recovered from the edit itself rather than from a noisy diff. The span stays narrow by construction: in `torch.cuda.set_active_device(gpu)`, only `set_active_device` is labeled — not the surrounding correct call. The code test split is additionally validated with evidence-based review.

Every span is typed. Three top-level categories carry the "what kind of wrong is this" decision:

- **Contradiction** — wrong logic, values, fields, or conditions in an otherwise plausible answer
- **Unsupported addition** — extra behavior or claims that were neither requested nor evidenced
- **Fabricated reference** — invented methods, attributes, keyword arguments, sections, or identifiers

Thirteen subcategories describe the surface element affected (`entity`, `temporal`, `numerical`, `value`, `relational`, `identifier`, `section`, `attribute`, `claim`, `behavior`, `elaboration`, `subjective`, `unspecified`), harmonizing the distinctions used by RAGTruth, FAVA, and code-hallucination taxonomies into one scheme that works across prose and code.

## The models

Two detector families are trained on this benchmark, sharing one task formulation and one prompt across all sources:

**[lettucedect-v2-qwen-2b](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b)** is a Qwen3.5-2B generative detector fine-tuned to return the hallucinated spans as structured JSON — each quoted verbatim from the answer, typed with category and subcategory, with optional per-span explanations. It is trained and evaluated with a **32,768-token maximum sequence length**, so a single prediction can include the user request, repository evidence or retrieved documents, tool output, and the answer to check. Fine-tuning is LoRA (rank 32, α=64) in bf16, learning rate 2×10⁻⁴ with a linear schedule, two epochs at effective batch size 32. Returned strings are matched back into the answer to recover character offsets.

**[lettucedect-v2-mmbert-base](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base)** follows the LettuceDetect token-classification architecture on a 307M-parameter mmBERT-base backbone: context and request tokens are masked from the loss, answer tokens are labeled supported/unsupported, and contiguous positives decode into character spans. It trains on the same split with an 8,192-token context, and a label-conditioned [taxonomy head](https://huggingface.co/KRLabsOrg/lettucedect-v2-taxonomy-head) can type its binary spans in a second stage. The encoder is much cheaper at inference — the right choice when throughput dominates.

## What they score

Evaluation is character-overlap span precision/recall/F1, example-level F1 (an answer is flagged if at least one span is predicted), and span IoU. For typed detection, a predicted span only receives credit when its category also matches. Span-F1 per source, against a fine-tuned LFM2.5-8B sibling and the zero-shot gpt-oss-120b judge:

![Span-F1 per source: the fine-tuned 2B detector vs a fine-tuned 8B sibling and a 120B zero-shot judge](/img/blog/lettucedetect-v2-results.png)

| Source (span-F1) | Qwen-2B | mmBERT-base | LFM-8B | gpt-oss-120b (judge) |
|------------------|---------|-------------|--------|----------------------|
| Unified test | **0.689** | 0.642 | 0.650 | — |
| Code-agent | **0.602** | 0.508 | 0.507 | 0.177 |
| Tool output | **0.719** | 0.588 | 0.692 | 0.331 |
| README | **0.866** | 0.751 | 0.804 | 0.666 |

The code-agent column explains why the benchmark had to exist. Zero-shot judges — including 550B-class models — over-flag correct newly-written code as "unsupported" because it is not literally in the context: with a generic prompt, Nemotron-3-Ultra marks clean patch code as fabricated, and a task-aware prompt only lifts its precision from 0.11 to 0.13. Prior span detectors fare no better there (LettuceDetect-large reaches 0.17). Detecting structured-input hallucinations is a learned skill, not a bigger judge.

Typed detection is harder than binary: the generative detector reaches 0.585 category-gated span-F1 (0.468 subcategory-gated) against its 0.689 binary score, and it beats the encoder-plus-taxonomy-head cascade end to end (0.585 vs 0.461 category-gated). On established natural-language benchmarks the 2B model stays competitive rather than specialized: **81.8 RAGTruth example-F1** (close to RAG-HAT's 83.9) and the best reported English PsiloQA IoU (**0.724**), with multilingual coverage from PsiloQA's 14 languages.

## The integration: a detector that is just another vLLM model

The `endpoint` backend (implemented by community contributor [@pranavthakur0-0](https://github.com/pranavthakur0-0) in [#2526](https://github.com/vllm-project/semantic-router/pull/2526)) lets the hallucination detector run behind any OpenAI-compatible server — the detector itself is served by vLLM, next to the models it verifies. No new serving infrastructure:

```yaml
hallucination_mitigation:
  enabled: true
  detector:
    backend: endpoint            # default remains: candle (in-process)
    endpoint: http://127.0.0.1:8077/v1
    model_id: KRLabsOrg/lettucedect-v2-qwen-2b
    include_explanation: true
```

The pipeline around it is unchanged. At request time, the fact-check sentinel gates which requests need verification at all, and the router captures the grounding context — tool results or RAG passages — from the request. At response time, instead of running the in-process token classifier and then a separate NLI explainer, the router makes **one structured call** to the endpoint: the model returns typed spans as strict-schema JSON (vLLM structured outputs guarantee parseability), the router recovers exact character offsets, and the configured action fires — a `x-vsr-response-warnings: hallucination` header for downstream systems to act on, or the annotated answer itself.

![Live demo: request routed, response intercepted, spans flagged and typed](/img/blog/lettucedetect-v2-demo.gif)

```text
$ curl http://router:8801/v1/chat/completions -d '{ "messages": [
    {"role": "user", "content": "What are the termination terms of our agreement?"},
    {"role": "tool", "content": "Section 8.2: ... thirty (30) days written notice.
     ... EUR 48,000, payable quarterly in advance."}]}'

HTTP/1.1 200 OK
x-vsr-response-warnings: hallucination

[Hallucination Warning] Detailed analysis:
  [contradiction/numerical] "sixty (60) days" — the contract says thirty (30) days
  [contradiction/value] "monthly" — the contract says quarterly
```

Failure semantics matter as much as the happy path: transport errors, non-2xx responses, and malformed detector output propagate as detector errors rather than silently becoming a clean verdict, and an unreachable endpoint degrades gracefully instead of blocking traffic.

One implementation detail is easy to get silently wrong: these models are supervised-fine-tuned on a **frozen prompt** — the exact system prompt and evidence serialization seen in training. Any drift degrades detection without producing a single error. The integration reproduces both byte-for-byte, down to paragraph breaks, and the exact prompts (including the explanation variant) are documented on the [model card](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b).

## On Semantic Router's own benchmark

The reproducible harness in `bench/hallucination/` evaluates detectors outside the router stack — encoders in-process, generative detectors through an OpenAI-compatible endpoint. On all 10,000 HaluEval QA samples, example-level:

| Detector | Params | Precision | Recall | F1 | p50 |
|----------|--------|-----------|--------|-----|-----|
| lettucedect-v2-qwen-2b (vLLM-served) | 2B | 0.962 | 0.769 | **0.855** | ~116 ms |
| haldetect-combined, best swept threshold | 149M | 0.999 | 0.502 | 0.668 | ~25 ms |

The two models serve different operating points: the encoder pipeline is ~4.5× faster at near-perfect precision; the generative detector catches roughly 1.5× the hallucinations and types each span. Which one a deployment runs is now a config decision.

## Try it

The detector is one `vllm serve` away:

```bash
vllm serve KRLabsOrg/lettucedect-v2-qwen-2b --port 8077 --max-model-len 131072
```

Point the router at it with the config above, send a request that carries grounding context (a `tool`-role message or RAG passages), and watch the `x-vsr-response-warnings` header. To compare detectors on your own data before deploying, the standalone harness accepts any custom JSONL with context/question/answer fields:

```bash
python3 -m bench.hallucination.evaluate_detectors \
    --detector llm:KRLabsOrg/lettucedect-v2-qwen-2b \
    --base-url http://localhost:8077/v1 \
    --dataset your_data.jsonl
```

The benchmark, the models, and the pipeline around them are all open — contributions to any of the three are welcome.

## Links

- Models & dataset: [huggingface.co/KRLabsOrg](https://huggingface.co/KRLabsOrg)
- Paper: [arXiv:2607.00895](https://arxiv.org/abs/2607.00895)
- LettuceDetect: [github.com/KRLabsOrg/LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)
- Benchmark harness: `bench/hallucination/` in this repository
