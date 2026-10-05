---
slug: vela-2-0-open-foundation-routing-models
title: "Vela 2.0: Towards Open Foundation Routing Models"
description: "Vela 2.0 adds span answers to the decision-model format: four open models (0.3B to 9B) that answer every routing signal, from safety and domain to PII and unsupported claims, in one request."
authors: [adaamko, Xunzhuo]
tags: [vela, routing-models, signals, pii, hallucination, safety, semantic-router]
image: /img/blog/vela2-hero.jpg
---

import { ArticleChartGallery, ArticleFigure, ArticleMetrics, ArticleVideo } from '@site/src/components/ArticleMedia';

<ArticleVideo
  src="/videos/vela-2-0/vela-2-0-launch.mp4"
  poster="/img/blog/vela2-hero.jpg"
  title="Vela 2.0: Towards Open Foundation Routing Models"
  landscape
>
  Still one model. Still one request. Now with spans. Sound on.
</ArticleVideo>

**A router asks many small questions before it picks a model. Vela 2.0 answers all of them in one request.**

Is the request harmful? Which domain is it about? Which words are personal data? Which claims in the answer does the context not support? Vela 2.0 answers every one of these as a typed question, with Choice, yes/no and Score answers, plus labelled character spans for any label you name. It comes in four sizes: a 307M encoder that runs on CPU, and 0.8B, 4B and 9B decoders built on the open Decision 2.0 models.

[**Explore the four models →**](https://huggingface.co/collections/vllm-sr/vela-20-towards-open-foundation-routing-models-6abfd3ba17c08e6d36a7e8c6) · [**Read about Decision →**](/blog/decision-models)

<!-- truncate -->

## Why decision models

Vela 1.0 gave every routing signal its own model: Safety, Hazard, Guard, Domain, Modality, FactCheck, Feedback, PII and Halu, each with a fixed head and a closed label set. That works until the router needs a new signal. Then it needs a new dataset, a new model and a new deployment, and none of the existing models can help.

[Decision models](/blog/decision-models) remove the fixed label set. The application writes the question and its options at request time, and the model returns a typed answer: a **Choice** among the options, a **Noul** (yes/no) probability or a **Score** over ordered levels. One model, any number of questions, and the labels live in the request rather than in the weights.

Decision 2.0 (Eos-0.8B, Nox-4B, Lux-9B) is the current generation of that idea. Each model reads its input with a Qwen3.5 text backbone and never generates text: it scores the options it is given. For the router, two pieces were still missing:

- **Spans.** PII entities and unsupported claims are pieces of text, not options. A routing model has to say *where* they are.
- **Routing accuracy.** A general decision model is not trained on the router's own signals. GLiNER2.5-Decide, for example, reaches a macro AUC of 0.704 on 14 public safety sets, too low to gate traffic on.

Vela 2.0 adds both. To our knowledge, it is the first model in the Choice / Noul / Score decision-model family that also answers span questions.

## One read, every question

Every Vela 2.0 model takes the same request: a *state* (plain text, or named parts such as `request`, `context` and `answer`) and any number of named questions of five types. Choice, Noul and Score keep the decision format unchanged. **Span** questions return `{label, start, end, text, probability}` with offsets into the part they are asked over, and **set** questions return any number of labels, each with its own probability.

In the decoders the state is read once. Every question is attached as its own block, with an attention mask that lets the block see the state and itself and nothing else.

<ArticleFigure
  src="/img/blog/vela2-one-read.png"
  width={2880} height={1680}
  alt="The state is encoded once; each question block attends to the state and to itself only."
>
  The state is read once. Each question block sees only the state and itself, so adding a question never changes another answer; in FP32 the isolation is exact.
</ArticleFigure>

This is what makes one model practical for a router. The questions a deployment asks can change from request to request, and each answer stays the same whatever else is asked alongside it.

## Spans inside a decision model

Spans are the hard part in a decoder. Under a causal mask, a word's hidden state sees only the words to its left, which is the wrong view for deciding where an entity starts and ends. Vela 2.0 repeats the target text after the label block (up to 2,048 tokens) and reads every word from that second copy, whose states have already seen the whole target and every label. Longer targets are read in overlapping windows of up to 1,800 tokens, and word scores are averaged across windows.

The labels need a full reading too. Each label is represented by the mean hidden state of its whole label block (its name and its description) plus a learned slot embedding. A bilinear + MLP scorer then rates every word × label pair in FP32, and neighbouring words above threshold become one span. Because the label is read from its description, a span question can use labels the model has never seen.

<ArticleFigure
  src="/img/blog/vela2-span-heads.png"
  width={2880} height={1800}
  alt="Span head v2: label vectors from the label block, word vectors from the repeated target, a word-by-label score grid, and the rule that sends each span question to the router or the broad head."
>
  Left: the word × label span head. Right: the rule that picks a span head for each question.
</ArticleFigure>

On development data, the repeated target and the label-block reading together raise short-text PII F1 from 0.828 to 0.959, and PII F1 on long documents from about 0.06 to 0.94.

## Two span heads, chosen per question

The decoders carry two span heads of the same form with separate weights:

- the **router head**, trained on PII, unsupported claims and toxic spans, the signals a router acts on;
- the **broad head**, trained on open extraction: named entities of any type, relations, entity mentions and extractive evidence for a question.

The broad head trains last, with the backbone, the decision heads and the router head frozen. Every router output is therefore bitwise identical with and without it; we checked this on 700 rows per model. The application does not pick a head. A fixed rule sends PII, hallucination and toxic questions to the router head and every other label set to the broad head, and the response names the head that answered. An explicit `head` field overrides the rule.

The result is one model that is a router signal model and an open extractor at once, with no change to its routing behaviour. The 0.3B encoder ships with the router head.

## Where Vela 2.0 sits

Vela 2.0 sits between two lines of work: span extractors that read labels at request time, and decision models that answer typed questions.

| | Typed decisions (Choice / Noul / Score) | Spans for labels named in the request | Grounded check (context vs answer) | Trained on router safety, PII and hallucination |
| --- | --- | --- | --- | --- |
| GLiNER | — | entities | — | — |
| GLiNER2 | classification | entities and structured fields | — | — |
| LettuceDetect | — | fixed label (unsupported) | yes | hallucination only |
| Decision 2.0 | yes | — | — | — |
| **Vela 2.0** | **yes** | **any label, two heads** | **yes** | **yes** |

GLiNER showed that a span extractor can read its labels from the request instead of a fixed output layer, and GLiNER2 extended that to classification and structured extraction in one schema. Vela 2.0 follows GLiNER2's label resampling during training. What it adds is the decision format around the spans: one request carries safety, domain and policy questions next to the span questions, answered in one forward pass of a model that also handles grounded inputs.

## Results

Every comparison below is on identical rows, scored by one harness, with seeds, checkpoints and thresholds chosen on development data only.

<ArticleMetrics items={[
  { label: 'Prompt attacks', value: '0.989', before: '0.792', baseline: 'Vela 1.0 Guard', measure: 'AUC · unseen attack families · 9B', source: 'https://huggingface.co/vllm-sr/Vela-2.0-9B' },
  { label: 'Multilingual hate speech', value: '0.855', before: '0.646', baseline: 'Vela 1.0 Safety', measure: 'AUC · Multilingual HateCheck · 9B', source: 'https://huggingface.co/vllm-sr/Vela-2.0-9B' },
  { label: 'Extractive evidence', value: '24.5', before: '7.0', baseline: 'best GLiNER-family model', measure: 'word-F1 · ACL-Verbatim, held out · 9B', source: 'https://huggingface.co/vllm-sr/Vela-2.0-9B' },
]} />

<ArticleChartGallery label="Choose a Vela 2.0 result" charts={[
  { label: 'Router signals', src: '/img/blog/vela-2-0/router-tasks.png', width: 2448, height: 1496, alt: 'Vela 2.0 9B against each Vela 1.0 specialist on the specialist test rows: prompt attacks 0.792 to 0.989, HateCheck 0.646 to 0.855, RTP-LX 0.761 to 0.801, long-document PII 0.908 to 0.940, hallucination 0.875 to 0.885, domain 0.831 to 0.844, short PII 0.976 to 0.985.', children: 'One Vela 2.0 9B against the Vela 1.0 specialists, each on its own test rows. The largest gains are where specialists generalise worst: unseen prompt-attack families and multilingual hate speech.' },
  { label: 'Safety', src: '/img/blog/vela-2-0/safety-family.png', width: 2448, height: 1496, alt: 'Macro AUC over 14 public safety sets: GLiNER2.5-Decide 0.704, Vela 2.0 0.3B 0.871, 0.8B 0.875, 4B 0.921, 9B 0.921.', children: 'Macro AUC over 14 public safety and prompt-attack sets. Vela 2.0 is trained on these signal families and Decide is not: this is what routing-specific training adds to a decision model.' },
  { label: 'Evidence', src: '/img/blog/vela-2-0/evidence.png', width: 2448, height: 1496, alt: 'ACL-Verbatim word-F1: Vela 2.0 9B 24.5, 4B 24.4, 0.8B 23.6, GLiFormer-large 7.0, GLiNER-large-v2.5 4.6, GLiNER2.5-small 4.6, GLiNER2.5-Decide 2.3.', children: 'The broad head finds the exact words that answer a question, on a set held out of training, at every decoder size.' },
  { label: 'Latency', src: '/img/blog/vela-2-0/latency.png', width: 2448, height: 1496, alt: 'Seconds per router request with seven questions on one A40: 0.3B 0.09, 0.8B 0.13, 4B 0.40 to 0.49, 9B 0.60 to 0.71.', children: 'One router request, seven questions including PII and hallucination spans, on one A40.' },
]} />

**Router signals.** One 9B model is ahead of or level with every Vela 1.0 specialist on the specialist's own test rows, and clearly ahead where specialists generalise worst: prompt attacks from unseen families (0.792 → 0.989 AUC, paired 95% CI +15.4 to +23.9 points) and multilingual hate speech (0.646 → 0.855). It also leads on RTP-LX request harm (0.761 → 0.801), PII in 8K-token documents (0.908 → 0.940 F1), hallucination spans (0.875 → 0.885 example-F1) and domain (0.831 → 0.844 macro-F1). The 0.3B encoder reaches 0.995 F1 on short-text PII, ahead of the Vela 1.0 PII model (0.976), and runs on CPU.

**Safety across the family.** Over 14 public safety and prompt-attack sets the four sizes reach a macro AUC of 0.871, 0.875, 0.921 and 0.921, against 0.704 for GLiNER2.5-Decide.

**Hallucination.** On the 2,700 RAGTruth test rows, the 9B reaches 0.774 example-F1, ahead of the LettuceDetect v2 encoder re-scored on the same rows (0.743, +3.1 points, paired 95% CI [+0.5, +5.4]), while answering every other routing question in the same call.

**Open extraction.** ACL-Verbatim asks for the exact sentences in a paper that support an answer, and it was held out of training. The broad head reaches 23.6 to 24.5 word-F1 at all three decoder sizes; no GLiNER-family model exceeds 7.0.

**Cost.** A full router request takes 0.09 s on an A40 for the 0.3B, 0.13 s for the 0.8B, 0.40–0.49 s for the 4B and 0.60–0.71 s for the 9B. That is one call for every signal, where Vela 1.0 needed one model per signal.

## Four sizes, one interface

<ArticleFigure
  src="/img/blog/vela2-family.jpg"
  width={1920} height={1080}
  alt="The four Vela 2.0 sizes with their safety macro AUC over 14 public sets."
/>

| Model | Initialised from | Parameters | Input | Span heads |
| --- | --- | ---: | ---: | --- |
| [Vela-2.0-0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B) | Decision-1.0-Kai-0.6B, encoder | 307M | 8,192 tokens | router |
| [Vela-2.0-0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B) | Decision-2.0-Eos-0.8B | 756M | 16,384 tokens | router + broad |
| [Vela-2.0-4B](https://huggingface.co/vllm-sr/Vela-2.0-4B) | Decision-2.0-Nox-4B | 4.2B | 16,384 tokens | router + broad |
| [Vela-2.0-9B](https://huggingface.co/vllm-sr/Vela-2.0-9B) | Decision-2.0-Lux-9B | 7.9B | 16,384 tokens | router + broad |

The request and response are the same at every size, so a deployment can start with the 0.3B on CPU and move to a decoder without changing a line of client code. The 0.3B also exports to ONNX.

## Training

<ArticleFigure
  src="/img/blog/vela2-training.png"
  width={2880} height={1640}
  alt="Three decoder training stages: a full fine-tune on the routing recipe, then the router span head, then the broad span head, each later stage on a frozen model."
>
  The decoders train in three stages from the released Decision 2.0 models. Each later stage freezes everything trained before it.
</ArticleFigure>

All four sizes share one data recipe. Every source is written as parts, typed questions and answers, and labels are resampled at every draw: options are anonymised, dropped and paraphrased, so the model learns to read labels rather than memorise them. Stage 1 fine-tunes the whole Decision 2.0 base for 4,000 steps, with a KL term to the frozen base on replayed Decision 2.0 rows to keep its general decisions. Stage 2 trains the router span head, and stage 3 the broad span head (90% open extraction: named entities, relations, entity mentions and extractive evidence from SQuAD 2.0, HotpotQA and Natural Questions train data; 10% router span replay). The 0.3B encoder trains in one stage of 101,000 steps from Kai's Choice trunk, about 6.9 hours on one AMD MI325X.

<details>
<summary>Stage-1 data mix</summary>

| Data | Sources | Share of steps |
| --- | --- | ---: |
| Long-document PII, 17 types | generated, train-only pool of 10,457 documents | 27.8% |
| Synthetic router-style decisions | 304,593 rows, 63,043 labels, 21 languages, generated with Qwen3-30B-A3B-Instruct-2507 | 23.1% |
| General decisions | SNLI, BoolQ, ARC, MASSIVE, TyDi QA and other Decision 1.0 source families, in our own wording | 12.5% |
| Safety and prompt attacks | AEGIS 2.0, PolyGuardMix, Nemotron-Safety-Guard v3, LLMail-Inject, Salad-Data | 12.2% |
| Hallucination spans | LettuceDetect prose and code data, including the RAGTruth train split | 11.3% |
| Decision 2.0 replay | Decision 2.0 training rows, with a KL term to the frozen base | 7.5% |
| Router tasks | Global-MMLU, Aya, DiffusionDB, WildFeedback, router PII recipe, Presidio replay | 3.8% |
| Small in-house set | | ~2% |

A synthetic row is kept only when a blind re-label agrees with it (71% do). Training rows are deduplicated against every evaluation set. Every size was trained with three seeds, and the released seed was chosen on development rows only.

</details>

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

Two span questions and a Choice, one call. The recorded output (trimmed):

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

Any other label set goes to the broad head. Asking "Which spans of the context answer the request?" over a product description returns `March 2024` (0.997) and `499 euros` (0.865) for "When did the X200 go on sale, and at what price?". A set question returns every label above its threshold, for example both `billing` and `shipping` for "My card was charged twice and the parcel never arrived."

Each model also ships `vela2_serve.py`, a small FastAPI server for the same request on `POST /v1/systemone`:

```bash
pip install fastapi uvicorn
python vela2_serve.py --model . --device cuda --port 8001
```

## Towards open foundation routing models

Vela 2.0 is a step from a set of routing classifiers towards one routing model: a model that reads a request once, answers whatever the deployment asks, points at the exact words behind its answers, and takes new labels without retraining. The next step is to bring it into vLLM Semantic Router as a native signal backend, so that one Vela 2.0 call replaces the per-signal classifiers on the request path.

- Models: [Vela 2.0 collection](https://huggingface.co/collections/vllm-sr/vela-20-towards-open-foundation-routing-models-6abfd3ba17c08e6d36a7e8c6) ([0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B), [0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B), [4B](https://huggingface.co/vllm-sr/Vela-2.0-4B), [9B](https://huggingface.co/vllm-sr/Vela-2.0-9B))
- Decision 2.0 bases: [Eos-0.8B](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B), [Nox-4B](https://huggingface.co/vllm-sr/Decision-2.0-Nox-4B), [Lux-9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B)
- Paper: *Vela 2.0: Towards Open Foundation Routing Models*, forthcoming

We thank the Decision 2.0 team for releasing the base models openly; the label resampling follows GLiNER2's recipe.
