---
slug: vela-3-0-name-it-vela-finds-it
title: "Vela 3.0: name it, Vela finds it"
description: "Six open routing models, 0.3B to 27B, that answer the router's decisions and return the spans you name in one call. A 380M UltraBERT encoder with the highest public Decision Index of any model up to 500M parameters, zero-shot entities at GLiNER level, and from 4B classification and relations at or above prompted LLMs."
authors: [adaamko, Xunzhuo]
tags: [vela, routing-models, zero-shot, ner, relation-extraction, classification, semantic-router]
image: /img/blog/vela-3-0/launch-poster.jpg
---

import { ArticleChartGallery, ArticleFigure, ArticleMetrics, ArticleVideo } from '@site/src/components/ArticleMedia';

<ArticleVideo
  src="/videos/vela-3-0/vela-3-0-launch.mp4"
  poster="/img/blog/vela-3-0/launch-poster.jpg"
  title="Vela 3.0: name it, Vela finds it"
  landscape
>
  Six open models. Decisions and spans in one call. Sound on.
</ArticleVideo>

**Name it. Vela finds it. Decisions and the spans you name, in one call.**

Vela 3.0 is a family of six open models for the vLLM Semantic Router, from a 0.3B encoder that reads images to a 27B model. Each one reads a request once and returns the router's decisions and the exact spans you name: personal data, unsupported claims, evidence, entities and relations. Its default router, **vela3-fast**, is a 380M [UltraBERT](https://huggingface.co/dmis-lab/UltraBERT-380M) encoder with the highest public Decision Index of any model up to 500M parameters: **50.36**. Across the family, entities are at GLiNER level, and from 4B up classification and relations are at or above every model the GLiFormer paper reports, prompted Gemma-4-31B, GPT-OSS-120B and GPT-5-mini included.

Vela 3.0 is the next step from [Vela 2.0](/blog/vela-2-0-open-foundation-routing-models): the same `system_one` interface, scaled at once in backbone, decisions, sizes, modality, languages and tasks.

[**Explore the six models →**](https://huggingface.co/collections/vllm-sr/vela-30-6ad1c430eb1718eb4754edcd) · [**Read about Vela 2.0 →**](/blog/vela-2-0-open-foundation-routing-models)

<!-- truncate -->

## Highlights

- **Decisions at 380M.** vela3-fast, a 380M UltraBERT encoder, has the highest public Decision Index of any model up to 500M parameters: **50.36** (our run of the unmodified kit, edition 0.3, public index). On the board's public column (reported), 72 entries larger than 1B score below it, 19 of them above 8B: Jevfire (27.8B) reads 49.8 and decisio 0.4.0 (Gemma 4 12B) 49.72. The 100 entries with a published size that score higher start at 873M (StartLux-Decision v1.1, 53.79).
- **Images at 0.3B.** vela3-fast-vision points at what you name on BoxSet-test v2 at 44.1 mean IoU, level with a 27B d3-turbo development build (44.2) at 1/89 the size.
- **Classification.** vela3-mini (80.2), vela3-flash (80.8) and vela3-turbo (81.4) are at or above every model the GLiFormer paper reports on its 13-set suite. The best of those is GPT-5-mini at 79.8 (reported).
- **Relations.** vela3-turbo scores **35.9** strict F1 end to end and vela3-mini 30.1, under the GLiFormer paper's Table 4 protocol. The best reported number there is 25.7, for a prompted Qwen3.5-9B.
- **Entities.** vela3-lite scores **64.4** on OOD-7 zero-shot NER at 0.8B. The best GLiNER-family number reported on it is 62.76.
- **Routing.** Against the Vela 2.0 model of the same size, the family is better on **105 of 144** router-signal test files and worse on 10. Every model wins more than it loses.

## Choose your model

| Model | Size | Base | Best for |
| --- | ---: | --- | --- |
| [vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) | 0.4B | UltraBERT-380M encoder | The default router: decisions and spans in one encoder pass, English, up to 32K tokens |
| [vela3-fast-vision](https://huggingface.co/vllm-sr/vela3-fast-vision) | 0.3B | UltraVBERT encoders | Images on CPU: image decisions, region pointing, OCR check, image hallucination spans |
| [vela3-lite](https://huggingface.co/vllm-sr/vela3-lite) | 0.8B | d3-lite | Small GPUs, low latency, multilingual |
| [vela3-mini](https://huggingface.co/vllm-sr/vela3-mini) | 4B | d3-mini | A balance of accuracy and speed |
| [vela3-flash](https://huggingface.co/vllm-sr/vela3-flash) | 9B | d3-flash | High-accuracy routing |
| [vela3-turbo](https://huggingface.co/vllm-sr/vela3-turbo) | 27B | d3-turbo v1.0 | The reference model |

## Quickstart

Start with vela3-fast, on GPU or CPU:

```bash
pip install "transformers==5.17.0" torch safetensors tokenizers huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("vllm-sr/vela3-fast")
sys.path.insert(0, path)
from vela3_encoder import Vela3Encoder

m = Vela3Encoder.load(path)  # CUDA if available, else CPU
out = m.system_one(
    "Steve Jobs founded Apple in Cupertino. My card was charged twice.",
    {"intent": {"type": "choice", "instructions": "What does the customer want?",
                "criteria": {"refund": "Money back for the duplicate charge", "cancel": "Cancel the order"}},
     "urgent": {"type": "noul", "instructions": "The customer needs a fast response."},
     "ents": {"type": "span", "instructions": "Which spans are named entities of these types?",
              "criteria": {"person": None, "organization": None}},
     "founder": {"type": "span", "instructions": "Which spans are the founder of Apple?", "criteria": {"founder": None}}})
# out["answers"]["intent"]: choice + probabilities; ["urgent"]: p(yes); ["ents"] / ["founder"]["spans"]: label, start, end, text, score
```

You pass the text as the state, and the questions you want answered as a dictionary. Each question is a choice, a yes / no, a score, or a span over labels you name. The decoder sizes load with `AutoModel.from_pretrained(..., trust_remote_code=True)` and take the same questions ([below](#try-it)).

## From Vela 2.0 to Vela 3.0

| | Vela 2.0 | Vela 3.0 |
| --- | --- | --- |
| **Encoder** | 307M ModernBERT encoder, 8,192 tokens | 380M UltraBERT encoder, 32K tokens, with its own decision backbone distilled from d3 |
| **Decisions** | The hybrids fine-tune the whole Decision 2.0 base; the 0.3B starts from Decision-1.0-Kai's Choice trunk | vela3-fast: 50.36 public Decision Index, the highest of any model up to 500M; the d3 sizes keep their base frozen and answer ordinary decisions bit for bit as the base (561 of 561 identical) |
| **Sizes** | Four models, 0.3B to 9B | Six models, 0.3B to 27B, four of them on d3 decoders |
| **Modality** | Text | Text and images; vela3-fast-vision points at what you name at 44.1 mean IoU at 0.3B |
| **Languages** | Router training data with synthetic rows in 21 languages | vela3-lite: MASSIVE intent accuracy of 76.4 to 83.7 in German, French, Spanish, Italian, Polish, Chinese and Japanese, within about 9 points of English |
| **Tasks** | Router spans, plus a broad span head for open extraction | Dedicated adapters for entities, relations and evidence: OOD-7 64.4 at 0.8B, relations 35.9 at 27B, QASPER evidence 45.3 at 4B (Vela 2.0 4B: 31.5, same harness) |
| **Unsupported claims** | 0.848 example-F1 at 0.3B (LettuceDetect v2 test set) | 0.889 at 0.4B (vela3-fast), 0.900 at 27B |
| **Router signals** | One model for every router signal | Better than the same-size Vela 2.0 on 105 of 144 test files, worse on 10 |

## One request, every question

A router asks two kinds of question about every request. A decision asks which way: which domain, is it a jailbreak, does the answer need a fact check, which model should take it. An extraction asks what exactly: which words are personal data, which claims of the answer the context does not support, which sentence is the evidence. Vela answers both in one call.

<ArticleFigure
  src="/img/blog/vela-3-0/architecture/vela3-fast.svg"
  width={1600} height={900}
  alt="vela3-fast: a system_one request with a state and named questions goes into a UltraBERT-380M encoder. A frozen decision backbone, distilled from d3, answers ordinary choice, yes-no and score questions with adapters off. Gated LoRA adapters for spans, evidence, entities, relations, router signals, and domain, fact-check and feedback switch on only for their own questions. The outputs are decisions with a probability per option and typed spans with label, offsets and score."
  diagram
>
  vela3-fast: one call, one encoder pass per question, no generated text. The decision backbone is frozen; each adapter switches on only for its own kind of question.
</ArticleFigure>

**Spans as decisions.** The labels you name are presented to the model as options. A span head scores every word against every label, and neighbouring words that agree join into a span with character offsets and a score. A span question therefore costs about as much as a decision: on vela3-fast, 18 ms at 200 tokens against 14 ms for four decisions.

**A frozen backbone with routed adapters.** Once the decision backbone is trained, it is frozen. Every other job gets its own small LoRA adapter on top, and a router in the package switches each adapter on only for its own kind of question. Because the backbone never moves, a new adapter cannot change an existing answer.

## Results

Vela 3.0 numbers are our runs through each released package, one request at a time, with the shipped thresholds. Other models' numbers are as their authors report them on the same test set and metric, marked "reported".

<ArticleMetrics items={[
  { label: 'Decisions up to 500M', value: '50.36', before: '47.49', baseline: 'GLiClass v4 Edge (0.4B), reported', measure: 'Decision Index public index · vela3-fast (0.4B)', source: 'https://huggingface.co/vllm-sr/vela3-fast' },
  { label: 'Relations, end to end', value: '35.9', before: '25.7', baseline: 'Qwen3.5-9B prompted, reported', measure: 'strict F1 · GLiFormer Table 4 protocol · vela3-turbo', source: 'https://huggingface.co/vllm-sr/vela3-turbo' },
  { label: 'Zero-shot classification', value: '81.4', before: '79.8', baseline: 'GPT-5-mini, reported', measure: 'macro-F1 · 13 sets, label names only · vela3-turbo', source: 'https://huggingface.co/vllm-sr/vela3-turbo' },
  { label: 'Zero-shot entities', value: '64.4', before: '62.76', baseline: 'best GLiNER-family model, reported', measure: 'F1 · OOD-7, label names only · vela3-lite (0.8B)', source: 'https://huggingface.co/vllm-sr/vela3-lite' },
]} />

<ArticleChartGallery label="Choose a Vela 3.0 result" charts={[
  { label: 'Decisions, all sizes', src: '/img/blog/vela-3-0/di-scatter-all.png', width: 3200, height: 1800, alt: 'Decision Index public index against model size for vela3-fast and every board entry with a published size. vela3-fast at 380M reads 50.36 (our run). 72 entries larger than 1B score lower, among them Jevfire 27.8B at 49.8 and decisio 0.4.0 (Gemma 4 12B) at 49.72 (reported). The smallest entry above is StartLux-Decision v1.1 (873M) at 53.79.', children: 'vela3-fast against every Decision Index board entry with a published size. Board entries: reported, 0.4.1 public column. vela3-fast: our run of the unmodified kit, edition 0.3, not a board entry.' },
  { label: 'Decisions up to 500M', src: '/img/blog/vela-3-0/decisions.png', width: 3200, height: 1800, alt: 'Decision Index public index, the top models up to 500M parameters: vela3-fast 50.36 (our run), Amber 0.4B 49.74, GLiClass v4 Edge 47.49 and Crivo-400M 39.79 (reported).', children: 'The highest public Decision Index of any model up to 500M parameters.' },
  { label: 'Entities, classification, relations', src: '/img/blog/vela-3-0/hero.png', width: 3200, height: 1800, alt: 'Zero-shot entities, classification and relations against model size: Vela 3.0 sizes against the GLiNER and GLiFormer family, prompted LLMs and NER-tuned generative models, all reported.', children: 'Label names only. From 4B, at or above every model the GLiFormer paper reports on classification and relations.' },
  { label: 'Router signals', src: '/img/blog/vela-3-0/router.png', width: 3200, height: 1800, alt: 'Router signal test files where each Vela 3.0 size is better, level or worse than the same-size Vela 2.0: fast 13/6/5, fast-vision 12/9/3, lite 23/1/0, mini 18/5/1, flash 19/4/1, turbo 20/4/0.', children: 'Each size against the same-size Vela 2.0 on the public router signal suite, paired 95% bootstrap CI.' },
  { label: 'Speed', src: '/img/blog/vela-3-0/speed.png', width: 3200, height: 1800, alt: 'Median latency at 200 tokens on one AMD Instinct MI325X: a span question takes 18 ms on vela3-fast, 29 ms on vela3-lite, 37 ms on vela3-mini and 40 ms on vela3-flash; four router decisions take 14 ms on vela3-fast and 22 ms on vela3-lite.', children: 'Medians at 200 tokens, batch 1, one AMD Instinct MI325X.' },
]} />

### Decisions

The [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) scores decision models on 37 public benchmarks plus private tests. vela3-fast has the highest public index of any model up to 500M parameters: **50.36**. Next on the board's public column (reported) come Amber 0.4B (49.74), GLiClass v4 Edge (47.49) and Crivo-400M (39.79). The largest leads over GLiClass v4 Edge are on Tools & Automation (+15.6) and Arts & Human Taste (+4.5).

Size is where the result stands out. Across every board entry with a published size, 72 models larger than 1B score below vela3-fast, 19 of them above 8B, including Jevfire (27.8B, 49.8) and decisio 0.4.0 (Gemma 4 12B, 49.72). The 100 entries that score higher start at 873M, with StartLux-Decision v1.1 at 53.79.

vela3-fast-vision answers the same text decisions with a 0.3B model that also reads images. It scores 36.7 on the public index and 22.79 on the vision board's public part.

Both are our runs of the unmodified kit at edition 0.3, one read per package, and neither is a board entry. We do not claim a Full (private-test) score. The board's 0.4.1 public column scores a rebuilt GSM8K, so the comparison is close but not exact. As a check, GLiClass v4 Edge reads 47.46 in our kit against its 47.49 on the board.

### Entities, classification and relations

The model gets only the label names. For relations, CoNLL04 is zero-shot, while DocRED, FewRel and CrossRE have public training splits in the mixture.

| Model | OOD-7 NER (F1) | 13-set classification (macro-F1) | Relations, Table 4 protocol (strict F1) |
| --- | ---: | ---: | ---: |
| **vela3-fast** (0.4B) | 62.3 | 74.4 | 17.4 |
| **vela3-lite** (0.8B) | **64.4** | 73.3 | 19.7 |
| **vela3-mini** (4B) | 63.3 (65.7) | 80.2 | **30.1** |
| **vela3-flash** (9B) | 62.0 (65.4) | **80.8** | **27.5** |
| **vela3-turbo** (27B) | **64.6** | **81.4** | **35.9** |
| GLiFormer-base (264M), reported | 60.40 | 72.36 | 18.94 |
| GLiFormer-large (576M), reported | 59.05 | 75.03 | 21.33 |
| gliner-multitask-large-v0.5 (440M), reported | 62.76 | – | – |
| NuNER Zero-span (449M), reported | 61.96 | – | – |
| GLiNER-L v1 (300M), reported | 60.9 | – | – |
| GLiNER2-large (486M) / GLiNER2 (205M), reported | – | 66.19 | 17.8 |
| GLiNER-Relex (size not given), reported | – | – | 25.6 |
| Qwen3.5-9B, reported (prompted for relations) | – | 71.4 | 25.7 |
| Gemma-4-31B-IT, reported (prompted for relations) | – | 79.69 | 25.08 |
| GPT-OSS-120B (5.1B active), reported (prompted for relations) | – | 77.76 | 23.99 |
| GPT-5-mini (size not given), reported (prompted for relations) | – | 79.79 | 22.1 |

Bold: at least 0.5 point above every other model listed for that benchmark. In brackets: the score when the model first writes one definition per label you named (vela3-mini and -flash; cached per label set). Sources: GLiNER Tab 1, GLiNER-multitask Tab 2, GLiFormer paper Tables 3 to 5.

**Named entities (OOD-7).** Seven test sets from domains the model was not trained on: five CrossNER domains, plus MIT Movie and MIT Restaurant. vela3-lite (64.4) and vela3-turbo (64.6) are above every GLiNER and GLiFormer number reported on it, and vela3-lite also passes GNER-T5-large (63.5 at 780M, reported). With their own label definitions, vela3-mini and -flash reach 65.7 and 65.4. vela3-fast (62.3) is level with NuNER Zero (61.96), and on the 26 NER sets of the GLiFormer paper it scores 57.4, above GLiFormer-large's 50.91 (reported, its card). Generative NER models fine-tuned for the task from 3B up, GNER-T5-xl and -xxl and B2NER, report higher OOD-7 scores; they are on the chart.

**Classification (13 sets).** The GLiFormer paper's suite of public sets for sentiment, topic, intent, emotion and spam, scored as mean macro-F1 from label names only. vela3-flash (80.8) and vela3-turbo (81.4) are above every model the paper reports, GPT-5-mini (79.8) and Gemma-4-31B-IT (79.7) included, and vela3-mini (80.2) is 0.4 above GPT-5-mini. vela3-fast (74.4) and vela3-lite (73.3) are above GLiFormer-base (72.36), Qwen3.5-9B (71.4) and GLiClass (71.01).

**Relations.** End to end, the model must find both entities and their relation, scored as strict F1 over whole (head, relation, tail) triples and averaged over CoNLL04, DocRED, FewRel and CrossRE, as in the GLiFormer paper's Table 4. vela3-mini (30.1), vela3-flash (27.5) and vela3-turbo (35.9) are above every model reported there, prompted LLMs included. vela3-lite (19.7) is above GLiFormer-base (18.94) and GLiNER2 (17.8), and vela3-fast (17.4) is level with GLiNER2. CoNLL04 is zero-shot for every Vela 3.0 size, while DocRED, FewRel and CrossRE have public training splits in its mixture. With the head entity given, CoNLL04 measures relations alone: vela3-turbo finds the related entities at 51.7 F1, vela3-mini at 42.5 and vela3-fast at 26.1, while GLiFormer-base scores 8.6 on the same rows in our harness.

### Unsupported claims and evidence

- **Hallucination spans.** On the 10,698-answer LettuceDetect v2 test set, vela3-fast flags unsupported answers at 0.889 example-F1 and vela3-turbo at 0.900. LettuceDetect v2 on mmBERT-base scores 0.869 (reported). On RAGTruth, vela3-flash scores 0.800, above LettuceDetect-large v1 (0.792, reported).
- **Evidence.** vela3-mini finds the evidence sentences on QASPER zero-shot at 45.3 word-F1; Vela 2.0 4B scores 31.5 in the same harness. vela3-fast scores 46.5 on ACL-Verbatim, where the benchmark's train split is in its training data (in-domain).

### Router signals

The [public router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite) is the vLLM Semantic Router's own test of its request signals: harmful requests, jailbreaks, hazard categories, personal data, unsupported answers, fact-check need, tool need, output modality, domain and user feedback. We ran each size against the Vela 2.0 model of its size, on the same rows and through each model's package, and scored it with the dataset's own code. A test file counts as a win or a loss when the paired 95% bootstrap confidence interval of the difference excludes zero.

| Model | vs | Better | Level | Worse |
| --- | --- | ---: | ---: | ---: |
| vela3-fast | Vela 2.0 0.3B | 13 | 6 | 5 |
| vela3-fast-vision | Vela 2.0 0.3B | 12 | 9 | 3 |
| vela3-lite | Vela 2.0 0.8B | 23 | 1 | 0 |
| vela3-mini | Vela 2.0 4B | 18 | 5 | 1 |
| vela3-flash | Vela 2.0 9B | 19 | 4 | 1 |
| vela3-turbo | Vela 2.0 9B | 20 | 4 | 0 |

Over the six models, 105 of 144 test files are better, 29 level and 10 worse, and two sizes lose none.

### Images: vela3-fast-vision

vela3-fast-vision is the family's image model, at 0.3B parameters and small enough for a CPU.

- **Region pointing.** On BoxSet-test v2 it scores 44.1 mean IoU. A 27B d3-turbo development build scores 44.2 on the same set with grid scoring, so vela3-fast-vision is level with it at 1/89 the size. Its accuracy at IoU 0.5 is 42.1, against 40.7 for that build.
- **OCR check.** Given a scanned document and its OCR text, it marks the parts of the OCR text that the image does not support. On our synthetic test set of 800 rendered documents it scores 86.8 char-F1, catching 504 of 632 corrupted fields with 34 false flags on 11,981 clean ones.
- **Image hallucination spans.** On M-HalDetect validation it marks the words of an image description that the image does not support at 0.518 char-F1 (evaluation only).
- **Text, too.** It flags unsupported answers at 0.892 example-F1 on the LettuceDetect v2 test set, above LettuceDetect v2 on mmBERT-base (0.869, reported), and it answers the router's text signals in the same package.

### Multilingual

vela3-lite inherits its base model's languages. On the MASSIVE test split, for three scenarios never used in training, its intent accuracy is 78.8 in German, 79.3 in French, 79.1 in Spanish, 78.8 in Italian, 76.4 in Polish, 83.7 in Chinese and 82.9 in Japanese, within about 9 points of English (85.1), with English label names. Hungarian (66.8) and Arabic (55.3) are partial. vela3-fast is English only.

### Speed

Medians at 200 tokens, one request at a time, on one AMD Instinct MI325X. A span question over any labels you name takes 18 ms on vela3-fast, 29 ms on vela3-lite, 37 ms on vela3-mini and 40 ms on vela3-flash. Four router decisions take 14 ms on vela3-fast and 22 ms on vela3-lite. vela3-fast-vision answers a question about a 1024 px image in 29–31 ms on the GPU and in about 1.3 s on 16 CPU threads. Each card has the 1K- and 3K-token columns.

## Six models, one interface

<ArticleChartGallery label="Choose a Vela 3.0 architecture" charts={[
  { label: 'vela3-fast (0.4B)', src: '/img/blog/vela-3-0/architecture/vela3-fast.svg', width: 1600, height: 900, diagram: true, alt: 'vela3-fast: a UltraBERT-380M encoder with a frozen decision backbone distilled from d3 and gated LoRA adapters for spans, evidence, entities, relations and router signals.', children: 'The encoder: a frozen decision backbone, distilled from d3, and gated adapters, one per job.' },
  { label: 'd3 sizes (0.8B to 27B)', src: '/img/blog/vela-3-0/architecture/vela3-d3.svg', width: 1600, height: 900, diagram: true, alt: 'vela3-lite, -mini, -flash and -turbo: a frozen d3 decoder. A question router sends ordinary choice, yes-no and score questions to the base with adapters off, router signals to decision LoRAs, and span questions to a span LoRA and span head.', children: 'The decoder sizes: a frozen d3 base at a pinned revision, with gated adapters. Ordinary questions run with every adapter off.' },
  { label: 'vela3-fast-vision (0.3B)', src: '/img/blog/vela-3-0/architecture/vela3-vision.svg', width: 1600, height: 900, diagram: true, alt: 'vela3-fast-vision: three UltraVBERT backbones, each answering one kind of question: a decision backbone for image and text decisions, a decision-trained backbone with a box adapter for region pointing, and the original frozen backbone with adapters for image hallucination spans, the OCR check and text.', children: 'The image model: three UltraVBERT backbones, one question each, so each question costs one 0.3B pass.' },
]} />

The six models share the question format. vela3-fast and vela3-fast-vision are encoders: they answer in one forward pass and write no text. vela3-lite, -mini, -flash and -turbo wrap a Decision 3.0 (d3) model, unchanged, at a pinned revision: d3-lite, d3-mini, d3-flash and d3-turbo v1.0.

**The d3 sizes.** The base model stays frozen. An ordinary choice, yes / no or score question goes to the base with every adapter off, so the answer is the base's own, bit for bit: each card's check finds 561 of 561 identical. Router signals switch on their decision adapters, and span questions switch on a span adapter and span head. vela3-mini and -flash can also write a one-line definition for each label you only named, and they cache it per label set. The adapters add 154 MB to vela3-lite and 5 GB to vela3-turbo.

**vela3-fast-vision.** Rather than ask one set of weights to do everything, it keeps three [UltraVBERT](https://huggingface.co/athrael-soju/UltraVBERT) backbones, each a SigLIP2-B vision encoder feeding an UltraBERT-150M text encoder at 1024 px, and sends each question to one of them: a decision backbone for image and text decisions, a second decision-trained backbone with the region-pointing adapter, and the original frozen backbone with the adapters for image hallucination spans, the OCR check and text.

## How we trained it

### Teaching an encoder to decide: distilling d3

vela3-fast starts from [UltraBERT-380M](https://huggingface.co/dmis-lab/UltraBERT-380M), a modern long-context English encoder. An encoder has no decision skill of its own, so we distil it from d3. The training rows are decision questions in the same format the model answers at inference: a state, a question and its options. The target for each row is d3's whole answer distribution, not just its top answer, which carries how sure the teacher is and which wrong options are close. The first stage makes about two passes over d3's 1.31M-row soft-label decision mixture.

Then we add three kinds of data, each disclosed on the card:

- **Public train splits** of the benchmark sources, written as decisions in the benchmark's own layouts: WinoGrande, HellaSwag, the New Yorker caption contest, Habermas, iSarcasmEval, ANLI, ACOS, NLI4CT, Humicroedit, VAST and RAGTruth. Train splits only.
- **Synthetic decision tasks** in the Decision Index's task framings: home-appliance control from the kit's own generator with fresh seeds; chord recognition on programmatic progressions; causal questions from our own structural causal models, answered by exact enumeration; and programmatic reasoning, rule-application, select-all and chess-position tasks from a pool of 3M unique rows.
- **Breadth pools**: knowledge multiple-choice, reading, label-set-perturbed classification, retrieval relevance and tool-use decisions, labelled by a teacher panel (d3 27B, d3-turbo v1.0 and d3-flash). A row is kept only when at least two of the three teachers agree with the gold label.

Where a row has a gold label, the target is half gold and half teacher. A row is dropped when the teacher is confident in a different answer, because that usually means the generator made a mistake. Every rendered request is checked against every Decision Index item, by exact match and by 13-gram near duplicates, and dropped on a hit.

### Checkpoint soups

Most recipes train three seeds from the same initialisation, and we average their weights. The final vela3-fast decision backbone is a uniform soup of four such students, at 30k to 60k steps each: three are themselves 3-seed soups (the train-split recipe, and the breadth recipe with two teacher-panel variants), and the fourth is a run with 1M programmatic rows. Because every student starts from the same distilled checkpoint, averaging is valid, and it costs nothing at inference.

### Adapters and router signals

With the decision backbone frozen, each adapter trains for its own job: spans for personal data and unsupported claims, evidence over long documents with the whole paper in one 32K window, entities, relations, and router signals with dedicated adapters for domain, fact-check and feedback. Router signals are trained supervised, as in Vela 2.0, on the public router datasets Vela 2.0 used: AEGIS 2.0, PolyGuardMix and Nemotron-Safety-Guard v3 for safety, Global-MMLU for domain, WildFeedback for feedback, DiffusionDB for modality and MASSIVE for intents. Domain and feedback also train on the router signal suite's own train split, with its test rows and anything sharing text with them kept out.

vela3-fast-vision's decision backbone is distilled with the vela3-fast text recipe plus d3 soft targets on images, including d3-mini's answer distributions on RVL-CDIP train-split images; rows that ask whether a response is supported keep up to 4,096 tokens of state, so the response is never cut off. Its region-pointing adapter is trained on box data from the Decision 3.0 team, the image hallucination adapter on span data built on COCO images, and the OCR check on synthetic rendered documents only.

## How we measure

- **Public test sets only,** run through each released package with `system_one`, one request at a time, with the shipped thresholds. The OCR check is the one exception: its test set is synthetic, and every text that uses it says so.
- **Other models** are as their authors report them on the same test set and metric, marked "reported" with the table they come from. Where we scored another model ourselves, the row says so: GLiFormer-base on CoNLL04 with the head given, and Vela 2.0 on the router suite.
- **In-domain** marks a benchmark whose train split is in that model's training data. Its test split is always held out.
- **Selection.** Candidate backbones were compared on our own development sets. The few that cleared a fixed bar got one pre-registered read each on the Decision Index public index, with its items split into two halves by item group. We picked on one half and checked the pick on the other. {/* Adam: PUBLIC_NUMBERS §9 flag 9 (selection protocol) is open; delete this bullet if you prefer not to discuss selection. */}
- **Zero-shot** means the model gets only label names, for NER, classification and relations (CoNLL04). Router signals are trained.
- **Sizes** are reported parameter counts. For the d3 sizes this is the base model; the adapters add 154 MB to 5 GB.

<details>
<summary>The benchmarks</summary>

- OOD-7: CrossNER AI, literature, music, politics and science, plus MIT Movie and MIT Restaurant (strict entity F1);
- the GLiFormer paper's 13 classification sets (mean macro-F1);
- relations under its Table 4 protocol: CoNLL04, DocRED, FewRel and CrossRE (strict triplet F1);
- the LettuceDetect v2 test set and RAGTruth (example-F1);
- QASPER and ACL-Verbatim (word-F1);
- BoxSet-test v2 (mean IoU);
- M-HalDetect validation (char-F1);
- MASSIVE (accuracy);
- the Decision Index (public index);
- the public router signal suite (AUC or accuracy, with the dataset's own scoring).

Each model's card has the per-set tables, the Vela 2.0 comparison on the same rows and the full source list.

</details>

## Try it

The decoder sizes load through `transformers`. On a GPU, ask vela3-mini for entities, unsupported claims, PII and relations with the same call:

```python
# pip install "transformers==5.17.0" torch torchvision pillow opencv-python-headless safetensors accelerate
from transformers import AutoModel

m = AutoModel.from_pretrained("vllm-sr/vela3-mini", trust_remote_code=True)  # downloads d3-mini at the pinned revision

# Name any labels; the spans come back with offsets and scores.
r = m.system_one(
    state="Geoffrey Hinton presented capsule networks at NeurIPS in Montreal.",
    questions={"ents": {"type": "span", "criteria": ["researcher", "algorithm", "conference", "location"]}})
# researcher "Geoffrey Hinton", algorithm "capsule networks", conference "NeurIPS", location "Montreal"

# Unsupported claims in an answer, against a context or an image.
r = m.system_one(
    state={"context": "The Eiffel Tower is 330 metres tall and was completed in 1889.",
           "answer": "The Eiffel Tower is 330 metres tall and was completed in 1925 by Gustave Eiffel's son."},
    questions={"halu": {"type": "span", "criteria": {"unsupported": None}}})
# flags "1925 by Gustave Eiffel's son"; add images=["photo.jpg"] to check an answer against a photo

# PII, relations and evidence use the same call.
r = m.system_one(
    state="Steve Jobs founded Apple in Cupertino together with Steve Wozniak.",
    questions={"pii": {"type": "span", "criteria": ["PERSON"]},
               "rel": {"type": "span", "instructions": "Which spans are the founder of Apple?", "criteria": {"founder": None}}})
```

Swap the repo name to change the size; the questions stay the same. vela3-fast-vision loads from its snapshot like vela3-fast and takes images with `images=[...]`; see [its card](https://huggingface.co/vllm-sr/vela3-fast-vision).

{/* Studio: the Space vllm-sr/vela3-studio is private as of 25 Oct; it goes public on release day. */}
The [Vela 3.0 Studio](https://huggingface.co/spaces/vllm-sr/vela3-studio) runs the examples with hover-to-source, live on vela3-fast-vision.

## In vLLM Semantic Router

Vela 2.0 plugs into the router as a `model_runtime` deployment: one deployment answers the built-in signals and the decision and span questions a recipe defines, in one call (see `config/fragments/signal/decision/vela2-labels-spans.yaml` and [Choose a model](/docs/next/model-runtime/choose-a-model)). Vela 3.0 keeps the `system_one` question format, but the router's model runtime does not load the Vela 3.0 packages yet. For now they run through their own packages, as shown above.

{/* TODO(Adam): Vela 3.0 is not a model-id swap today. The runtime's vela2 family detects only Vela 2.0 members (src/model-runtime/vllm_srun/families/vela2/family.py, member_of), and there is no vela3 family; the d3 bases (d3-lite, d3-mini, d3-flash, d3-turbo) are in the decision3 registry, but the Vela 3.0 adapters and the UltraBERT encoder are not. Replace this paragraph with the config fragment and the e2e profile once a vela3 family lands. */}

## Towards open foundation routing models

{/* DRAFT for Adam: the paragraph below. */}
Vela 3.0 shows that a 380M encoder can carry decisions, router signals and open extraction in one pass, and that the larger sizes can add spans and signals to a decision model while keeping its own answers. Next we want these models to be the router's native signal backend, and to keep widening what one request can ask: more languages in the small sizes, more modalities, and new adapters that leave every existing answer untouched.

- Models: [Vela 3.0 collection](https://huggingface.co/collections/vllm-sr/vela-30-6ad1c430eb1718eb4754edcd) ([vela3-fast](https://huggingface.co/vllm-sr/vela3-fast), [vela3-fast-vision](https://huggingface.co/vllm-sr/vela3-fast-vision), [vela3-lite](https://huggingface.co/vllm-sr/vela3-lite), [vela3-mini](https://huggingface.co/vllm-sr/vela3-mini), [vela3-flash](https://huggingface.co/vllm-sr/vela3-flash), [vela3-turbo](https://huggingface.co/vllm-sr/vela3-turbo))
- Bases: Decision 3.0 (d3-lite, d3-mini, d3-flash, d3-turbo v1.0); [UltraBERT-380M](https://huggingface.co/dmis-lab/UltraBERT-380M); [UltraVBERT](https://huggingface.co/athrael-soju/UltraVBERT)
- Router suite: [vllm-sr/router-signal-suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
- Studio: [vllm-sr/vela3-studio](https://huggingface.co/spaces/vllm-sr/vela3-studio)
- Previous release: [Vela 2.0: Open Foundation Routing Models](/blog/vela-2-0-open-foundation-routing-models)

**License.** vela3-fast and the four d3 sizes (vela3-lite, -mini, -flash and -turbo) are Apache-2.0. vela3-fast-vision is MIT, the licence of its UltraVBERT base, and ships the Apache-2.0 text of its UltraBERT-150M and SigLIP2 components. Each model card links its notices and license scope.

**Hardware.** vela3-fast was trained on AMD Instinct MI325X GPUs, and every GPU latency in this post was measured on one AMD Instinct MI325X.
