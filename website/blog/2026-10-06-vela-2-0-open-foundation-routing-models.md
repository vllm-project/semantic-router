---
slug: vela-2-0-open-foundation-routing-models
title: "Vela 2.0: Towards Open Foundation Routing Models"
description: Four open routing models (0.3B, 0.8B, 4B, 9B) that answer the router's safety, PII, hallucination and routing questions as typed questions in one request, with span and set answers on top of the SystemOne decision format.
authors: [adaamko, Xunzhuo]
tags: [vela, routing-models, signals, pii, hallucination, safety, semantic-router]
image: /img/blog/vela2-hero.jpg
---

Semantic Router now has one model family for the small decisions it makes on every request. **Vela 2.0** answers safety, prompt-attack, domain, modality, PII and hallucination checks as typed questions in one request, and returns labelled character spans next to the usual Choice, yes/no and Score answers. It comes in four sizes: a 307M encoder that runs on CPU, and 0.8B, 4B and 9B decoders fine-tuned from the released Decision 2.0 models.

This post walks through the architecture, the training recipe and the results against the Vela 1.0 specialists, LettuceDetect and the GLiNER family, including where the new models are weaker. All four are released under Apache-2.0.

![Vela 2.0: Towards Open Foundation Routing Models](/img/blog/vela2-hero.jpg)

<!-- truncate -->

## Why one model

Before the router picks a model it asks several small questions about the request: is it harmful or a prompt attack, which domain and output modality it needs, which spans are personal data, and which claims of a generated answer its context does not support. In Vela 1.0 each question had its own classifier: Safety, Hazard, Guard, Domain, Modality, FactCheck, Feedback, PII and Halu, each with a fixed head and a closed label set. A new signal meant a new dataset, a new model and a new deployment, and none of them could answer a question it was not trained with.

Typed decision models remove the fixed label set: the SystemOne format of the Jev API and the open Decision models answers a question written at request time as a *Choice* among supplied options, a yes/no probability (*Noul*) or a *Score* over ordered levels. Two gaps remained for the router. PII entities and unsupported claims are spans of text, hazard categories are multi-label sets, and a grounded hallucination check needs the context and the answer as separate inputs; none of these fits the three types. And generic decision models are weak on the router's own tasks: GLiNER2.5-Decide reaches a macro AUC of 0.704 on 14 router safety sets, too low to gate traffic on.

## The family

| Model | Initialised from | Parameters | Input | Safety, macro AUC (14 sets) |
|-------|------------------|-----------:|------:|----------------------------:|
| [Vela-2.0-0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B) | Decision-1.0-Kai-0.6B (Choice trunk), encoder | 307M | 8,192 | 0.871 |
| [Vela-2.0-0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B) | Decision-2.0-Eos-0.8B | 756M | 16,384 | 0.875 |
| [Vela-2.0-4B](https://huggingface.co/vllm-sr/Vela-2.0-4B) | Decision-2.0-Nox-4B | 4.2B | 16,384 | **0.921** |
| [Vela-2.0-9B](https://huggingface.co/vllm-sr/Vela-2.0-9B) | Decision-2.0-Lux-9B | 7.9B | 16,384 | **0.921** |

Every member takes the same request: a state (plain text, or `request`, `context` and `answer` parts) and named questions of five types. Choice (2–255 options), Noul and Score keep SystemOne's request and response shape. *Span* questions return `{label, start, end, text, probability}` with code-point offsets into the part they are asked over, and *set* questions return any number of labels with one probability each. To our knowledge, Vela 2.0 is the first model in the System One decision-model family (Choice / Noul / Score) to also answer span questions.

![Four sizes, one family: safety macro AUC over 14 public sets](/img/blog/vela2-family.jpg)

## How a request is read

The 0.3B encoder reads one token sequence: a schema block of typed questions behind marker tokens, then the typed parts. The decoders start from Decision 2.0. A Decision 2.0 model uses a Qwen3.5 text backbone as an encoder and never generates text; it asks one question per sequence, so every question re-reads the context. In Vela 2.0 the state is encoded once, and every question is attached as its own block with an attention mask that lets the block see the state and itself, and nothing else.

[![The state is read once; each question block sees only the state and itself](/img/blog/vela2-one-read.png)](/img/blog/vela2-one-read.png)

Adding a question therefore does not change the other answers; in FP32 the isolation is exact. The encoder, where all questions share one sequence, lacks this property: in one check a single answer moved from 0.771 asked alone to 0.461 bundled.

### Spans: word × label

Spans were the hard part in a decoder. Under a causal mask a word's state sees only the words to its left, which is the wrong view for span boundaries. So the target text (up to 2,048 tokens) is repeated after the label block, and words are read from that second copy, whose states have seen the whole target and the labels. Longer targets are read in windows of up to 1,800 tokens with a stride of 1,536, and word logits are averaged over windows.

The label side needed a fix too. Our first decoder span head read each label at a single marker token, and the label vectors came out almost identical (pairwise cosine 0.997), so it learned whether a word is an entity but not which one. The released span head reads each label as the mean hidden state of its whole label block, name and description, plus a learned slot embedding, and scores every word × label pair with a bilinear + MLP scorer in FP32. On dev, the repeated target and the new span head together raise short PII F1 from 0.828 to 0.959, and long-document PII F1 from about 0.06 to 0.94.

[![Span head v2 and the two-head dispatch](/img/blog/vela2-span-heads.png)](/img/blog/vela2-span-heads.png)

### Two span heads

The span head above is trained on PII, hallucination and toxic labels only. Asked for an arbitrary entity type it finds little (zero-shot NER dev F1 0.113 at 0.8B, 0.219 at 9B). The decoders therefore carry a second, *broad* span head of the same form with its own weights, trained last with the backbone, the decision heads and the router span head frozen. A fixed rule (in the figure above) sends PII, hallucination and toxic questions to the router head and every other label set to the broad head, and the response names the head that answered in `span_heads`.

Because the router path is untouched, every router output is bitwise identical with and without the broad head; we checked this on 700 rows per model. The 0.3B ships without a broad head; trained the same way, it learned some open NER (31.0 F1) but no evidence extraction (0.1 word-F1 on ACL-Verbatim).

## Training

[![Three training stages for the decoders](/img/blog/vela2-training.png)](/img/blog/vela2-training.png)

All four members share one data recipe. Every source is written as parts, typed questions and answers, and labels are resampled at every draw following GLiNER2's recipe: options are anonymised, dropped and paraphrased, so the model cannot memorise a label set.

| Data | Sources | Share of stage-1 steps |
|------|---------|-----------------------:|
| Long-document PII, 17 types | generated, train-only pool of 10,457 documents | 27.8% |
| Synthetic router-style decisions | 304,593 rows, 63,043 labels, 21 languages, generated with Qwen3-30B-A3B-Instruct-2507 | 23.1% |
| General decisions | SNLI, BoolQ, ARC, MASSIVE, TyDi QA and other Decision 1.0 source families, in our own wording | 12.5% |
| Safety and prompt attacks | AEGIS 2.0, PolyGuardMix, Nemotron-Safety-Guard v3, LLMail-Inject, Salad-Data | 12.2% |
| Hallucination spans | LettuceDetect prose and code data, including the RAGTruth train split | 11.3% |
| Decision 2.0 replay | Decision 2.0 training rows, with a KL term to the frozen base | 7.5% |
| Router tasks | Global-MMLU, Aya, DiffusionDB, WildFeedback, router PII recipe, Presidio replay | 3.8% |
| Small in-house set | | ~2% |

A synthetic row is kept only when a blind re-label agrees with it (71% do). Hallucination rows whose context the 8,192-token encoder window would cut (9,470 of 144,943) are dropped, since a cut context teaches the model to flag claims whose support was removed. Training rows are deduplicated against every evaluation set.

The decoders train in three stages. Stage 1 fine-tunes the whole Decision 2.0 base for 4,000 steps on the recipe above, with a KL term to the frozen base on the replay rows so part of its general decisions is kept. Stage 2 trains only the router span head, and stage 3 only the broad span head (90% open extraction: NER, relations, entity mentions and extractive evidence from SQuAD 2.0, HotpotQA and Natural Questions train data; 10% router span replay). The 0.3B encoder trains in one stage, 101,000 steps from Kai's Choice trunk, about 6.9 hours on one AMD MI325X.

Every size was trained with three seeds. Seeds, checkpoints, temperatures and thresholds are chosen on dev rows only, test numbers are compiled after the choice, and every difference below is a paired bootstrap over identical rows.

## What they score

### Against the Vela 1.0 specialists

Each comparison runs on the specialist's own test rows, with the router-pinned revisions re-scored per row.

| Task | Vela 1.0 | Vela 2.0 0.3B | Vela 2.0 9B | Δ 9B vs Vela 1.0 [95% CI] |
|------|---------:|--------------:|------------:|--------------------------:|
| Prompt attacks, unseen families (AUC) | 0.792 (Guard) | 0.882 | **0.989** | +19.8 [+15.4, +23.9] |
| Multilingual HateCheck (AUC) | 0.646 (Safety) | 0.662 | **0.855** | +20.9 [+20.3, +21.5] |
| RTP-LX request harm (AUC) | 0.761 (Safety) | 0.728 | **0.801** | +4.0 [+3.4, +4.5] |
| PII, short texts (F1) | 0.976 | **0.995** | 0.985 | +0.9 [+0.1, +1.9] |
| PII, 8K-token documents (F1) | 0.908 | 0.894 | **0.940** | +3.2 [−1.1, +7.0] |
| Hallucination, 10,698 examples (example-F1) | 0.875 | 0.848 | **0.885** | +1.0 [+0.4, +1.7] |
| Domain (macro-F1) | 0.831 | 0.825 | **0.844** | +1.3 [−0.3, +2.8] |

The 9B is ahead of or level with the specialist on every router task we can measure, with the largest gains where the specialists generalise worst: unseen prompt-attack families and multilingual hate speech. The long-document PII buckets have 30 documents each, so their intervals are wide. Modality, fact-check and feedback go from 0.889, 0.501 and 0.425 macro-F1 to 1.000, 1.000 and 0.939 at 9B, but their test sets are weak, so we treat them as sanity checks.

### Safety against a generic decision model

Over the 14 safety sets (RTP-LX, HateCheck, XSTest, CultureGuard, AEGIS 2.0, PolyGuard, Do-Not-Answer and prompt-attack sets), the macro AUC is 0.871 / 0.875 / 0.921 / 0.921 for the four sizes against 0.704 for GLiNER2.5-Decide. The paired intervals put the 0.8B and the 9B ahead of Decide on all 14 sets. Vela was trained on these families and Decide was not, so this shows what routing-specific training buys, not zero-shot transfer.

### Hallucination on RAGTruth

On the same 2,700 RAGTruth test rows, with both LettuceDetect v2 checkpoints re-scored through their own evaluation input, Vela 2.0 9B reaches 0.774 example-F1, ahead of the LettuceDetect v2 mmBERT encoder (0.743; +3.1 [+0.5, +5.4]) and behind the 2B generative detector (0.817; −4.3 [−6.3, −2.3]). When hallucination detection is the only job, the generative detector is the better choice.

### Open extraction with the broad head

![The broad span head on user-named labels (Vela 2.0 4B)](/img/blog/vela2-your-labels.jpg)

All models below are scored by one harness, which reproduces the GLiFormer-base card on all seven NER sets and the GLiNER-large-v2.5 average within 0.2 points.

| Model | Zero-shot NER, 7 sets (F1) | ACL-Verbatim evidence (word-F1) | ms per question, 3K-token context |
|-------|---------------------------:|--------------------------------:|----------------------------------:|
| Vela 2.0 9B, broad head | 43.5 | **24.5** | 2,288 |
| Vela 2.0 4B, broad head | 40.7 | 24.4 | 1,499 |
| Vela 2.0 0.8B, broad head | 28.9 | 23.6 | 477 |
| GLiFormer-large | **63.6** | 7.0 | 930 |
| GLiNER-large-v2.5 | 61.3 | 4.6 | 496 |
| GLiNER2.5-Decide | 51.5 | 2.3 | 415 |
| GLiNER2.5-small | 42.5 | 4.6 | **197** |

Evidence extraction is where the broad head is strong. ACL-Verbatim was held out of training; all three sizes reach 23.6–24.5 word-F1 on it, while no GLiNER-family model exceeds 7.0 (a ModernBERT highlighter trained on ACL-Verbatim itself reaches 53.4). Open NER is the weak side: the 9B is level with GLiNER2.5-small and 20.1 points below GLiFormer-large. Latency is the mean over 100 questions on one A40.

### Cost

A router-sized request, seven questions including PII and hallucination spans, takes about 0.09 s on an A40 for the 0.3B, 0.13 s for the 0.8B, 0.40–0.49 s for the 4B and 0.60–0.71 s for the 9B. On a two-thread CPU the 0.3B takes 1.9–3.4 s and the 0.8B 38–51 s. The decoders need about 3, 17 and 32 GB of GPU memory in FP32. On CPU the encoder is the only practical member; on a GPU the 9B is the most accurate on most tasks.

## Where it gives ground

General decisions are below the Decision 2.0 bases. On the Jev Decision Index (edition 0.2.1, 38 benchmarks) the 0.8B scores 16.01 against 20.14 for Eos-0.8B, the 4B 31.63 against 42.55 for Nox-4B, and the 9B 41.09 against 46.23 for Lux-9B. The 4B loses most on intent classification (BANKING77 macro-F1 0.48, CLINC150 0.16). On Decide's own benchmark, fast-decisions, the 9B is level with Decide (62.5 vs 62.9) and the 0.3B is 13.5 points behind. Vela 2.0 is specialised for routing; Decision 2.0 remains the better general decision model at each size.

Open NER is behind dedicated models, and the broad head is slower than the GLiNER-family encoders (0.48–2.3 s against 0.2–0.9 s per question). Smaller gaps against the per-task references: toxic spans for the 4B (−7.2 char-F1 points against an mmBERT span head trained on the same data), long-document PII for the 0.8B (0.603 F1 at 8K tokens), and RTP-LX request harm for the 0.3B (−3.2 AUC).

## Try it

```bash
pip install torch "transformers>=5.17" safetensors tokenizers numpy
```

```python
from transformers import AutoModel

m = AutoModel.from_pretrained("vllm-sr/Vela-2.0-9B", trust_remote_code=True).to("cuda")
PII_LABELS = m.vela2_engine.cal["pii_schema"]["labels"]   # the 17 trained PII types

result = m.system_one(
    state={"request": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol for an adult?",
           "source": "For adults, the maximum dose of paracetamol is 4 grams in 24 hours, taken as 500 mg to 1 g every 4 to 6 hours.",
           "answer": "Adults can take up to 6 grams of paracetamol in 24 hours, in doses of 500 mg to 1 g every 4 to 6 hours."},
    questions={
        "pii": {"type": "span", "instructions": "Which spans are personal information?",
                "criteria": PII_LABELS, "over": "request"},
        "halu": {"type": "span", "instructions": "Which spans of the answer are not supported by the context?",
                 "criteria": {"unsupported": "a claim not supported by the context"}},
        "domain": {"type": "choice", "instructions": "Which subject area is this request about?", "over": "request",
                   "criteria": {"health": "medicine, clinical practice, nutrition, ageing or sexual health",
                                "math": "arithmetic, algebra, geometry, statistics or other mathematics",
                                "other": "a subject that fits none of the listed areas"}}})
```

Two span questions and a Choice, one call. The recorded output from the release (trimmed):

```json
{
  "answers": {
    "pii": {"type": "noul", "noul": 1.0},
    "halu": {"type": "noul", "noul": 0.996},
    "domain": {"type": "choice", "choice": "health", "confidence": 0.977}
  },
  "spans": {
    "pii": [{"label": "PERSON", "start": 8, "end": 17, "text": "Tom Baker", "probability": 0.999},
            {"label": "EMAIL_ADDRESS", "start": 19, "end": 40, "text": "tom.baker@example.com", "probability": 1.0}],
    "halu": [{"label": "unsupported", "start": 22, "end": 29, "text": "6 grams", "probability": 0.982}]
  },
  "span_heads": {"pii": "router", "halu": "router"}
}
```

Other label sets go to the broad head: asking "Which spans of the context answer the request?" over a product description returns `March 2024` (0.997) and `499 euros` (0.865) for "When did the X200 go on sale, and at what price?". A set question (`"type": "set"`) returns every label above its threshold, for example both `billing` and `shipping` for "My card was charged twice and the parcel never arrived."

Each model also ships `vela2_serve.py`, a small FastAPI server for the same request on `POST /v1/systemone`, so the `typesafe-sdk` client works against it with `base_url` pointed at the server:

```bash
pip install fastapi uvicorn
python vela2_serve.py --model . --device cuda --port 8001
```

The 0.3B has the same interface, runs on CPU, and also exports to ONNX.

## Links

- Models: [Vela 2.0 collection](https://huggingface.co/collections/vllm-sr/vela-20-towards-open-foundation-routing-models-6abfd3ba17c08e6d36a7e8c6) ([0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B), [0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B), [4B](https://huggingface.co/vllm-sr/Vela-2.0-4B), [9B](https://huggingface.co/vllm-sr/Vela-2.0-9B))
- Paper: *Vela 2.0: Towards Open Foundation Routing Models*, forthcoming
- Demo Space: forthcoming
- Decision 2.0 bases: [Eos-0.8B](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B), [Nox-4B](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B), [Lux-9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B)

Vela 2.0 is led by KR Labs and vLLM Semantic Router. The decoders build on the Decision 2.0 models, and we thank the Decision 2.0 team for releasing them openly; the label resampling follows GLiNER2's recipe.
