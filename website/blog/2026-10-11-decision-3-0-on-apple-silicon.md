---
slug: decision-3-0-on-apple-silicon
title: "Decision 3.0 on Apple Silicon: Text, Image and Video Decisions on a Mac"
description: "d3-lite, d3-nano and d3-mini converted to Core ML: triage 1,000 pull requests at 0.12 s each, drive a real website, watch live video and play Super Mario Bros., all on one Mac."
authors: [Alex-Wengg]
tags: [decision, core-ml, apple-silicon, multimodal, demos, games, semantic-router]
image: /img/blog/decision-3-0-coreml/pr-triage-poster.jpg
---

import { ArticleVideo } from '@site/src/components/ArticleMedia';

<ArticleVideo
  src="/videos/decision-3-0-coreml/d3-pr-triage.mp4"
  poster="/img/blog/decision-3-0-coreml/pr-triage-poster.jpg"
  title="d3-lite triages pull requests on Core ML"
  landscape
>
  d3-lite sorts 1,000 sample pull requests into the vLLM-SR workgroups, on an M5 Pro. Right: GPU load and the per-PR log.
</ArticleVideo>

**Decision 3.0 runs on a Mac.**

[Decision 3.0](https://huggingface.co/collections/vllm-sr/decision-30) brought text, image and video decisions to one open model family. We converted the three smaller sizes (**d3-lite 0.8B, d3-nano 2B and d3-mini 4B**) to Core ML and built demos for text, screenshots and live video. Later we added d3-flash 9B, which plays Super Mario Bros. and taught d3-lite to do the same.

A decision made on the Mac has no per-call cost and no network round trip, and the pull request, web page or video never leaves the machine. That fits alongside vLLM-SR: routine decisions can be answered on the device, and the ones that need a bigger model can go to the router.

[**Get the Core ML models →**](https://huggingface.co/FluidInference/d3-coreml) · [**Swift runtime →**](https://github.com/FluidInference/FluidUse) · [**Watch the demo on X →**](https://x.com/Alex_tra_memory/status/2109343807854493724)

<!-- truncate -->

## How the conversion works

Each d3 model is a Qwen3.5 backbone with a small FP32 readout that scores every option at the last prompt token, one forward pass per question. Core ML runs the heavy part, the backbone and the vision tower. The host app runs the light steps around it: a table lookup for the embeddings, arithmetic for the positions, and one small matrix multiply for the readout.

In detail:

- **Text graph on Core ML.** The backbone takes token embeddings as input, so image and video tokens can be spliced in before it runs. Prompts are right-padded to a few fixed lengths. That is exact for d3: the Gated DeltaNet layers are causal and never see the padding, and the full-attention layers mask it out.
- **Vision tower on Core ML.** It runs in fp32 with fixed patch counts and masked padding. For video, a block-diagonal mask keeps attention inside each frame pair, as upstream does.
- **Host side.** Token embeddings, M-RoPE positions and the 255-way readout run in numpy (or Swift) in FP32. Prompts come from upstream's own request preparation, run on a d3 with no weights loaded, so every prompt, timestamp and position matches upstream exactly.
- **8-bit weights.** At fp16, d3-mini's text graph is 6.7 GB per length bucket. Two buckets loaded at once made the GPU page on a 24 GB Mac, at 8–12 s per step. With 8-bit weights both stay resident.

Conversion uses PyTorch 2.7 and coremltools 9.0. We checked the Core ML release against upstream PyTorch (bf16 on the Mac's GPU) on two public benchmarks: the [JevBench](https://github.com/fstandhartinger/jevbench/tree/b6b8fff7e345b98c060ad26c13308860ddc67004) public suite, run through its unchanged System One adapter and scorer as in the [System One Auto](/blog/system-one-auto) post, and the [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) TEST split, scored against its gold labels.

| | d3-lite | d3-nano | d3-mini |
|---|---|---|---|
| **JevBench public** (231 tasks), PyTorch → Core ML | 73.2% → **74.0%** | 75.3% → **75.8%** | 81.8% → **81.8%** |
| **typed-decisions TEST** (2,000 questions), PyTorch → Core ML | 51.6% → **51.6%** | 62.7% → **62.7%** | 73.6% → **73.9%** |
| Median JevBench request, PyTorch → Core ML | 204 → **53 ms** | 173 → **69 ms** | 438 → **169 ms** |
| Download | 3.7 GB | 10.5 GB | 19.6 GB |

**Core ML scores the same as upstream, and each request is 2.5–4× faster.** On a Mac, upstream PyTorch falls back to Transformers' reference code for the DeltaNet layers, since the fast kernels it uses on NVIDIA GPUs aren't available.

All numbers are from one MacBook Pro (M5 Pro, 24 GB), running on the GPU. d3-nano is the middle option: close to d3-lite's speed, and 11 points above it on typed-decisions.

<details>
<summary>Method and limits</summary>

- Both backends ran one request at a time. Request times include a local HTTP round trip.
- JevBench accuracy is exact-label accuracy on the public suite, not the official sealed leaderboard or its composite score. 37 of its tasks are longer than 1,536 tokens (the longest is 3,970), so the release includes a 4,096-token text graph.
- typed-decisions gold labels come from several annotators who often disagree, so accuracy there sits well below JevBench for every model.

</details>

## Demos

d3-lite handles the demo where speed matters; d3-mini handles screenshots and video; d3-flash and a distilled d3-lite play Mario.

**Triage pull requests (d3-lite, [on X](https://x.com/Alex_tra_memory/status/2109343807854493724)).** The video at the top. For each of 1,000 sample pull requests, d3-lite reads the title, the start of the description and the changed files, then answers two questions: which of the seven [vLLM-SR workgroups](/blog/join-vllm-sr-workgroups) owns it, and what kind of change it is. Each PR takes **0.12 s, so 8.3 PRs and 16.5 decisions per second**. On real vLLM-SR pull requests, d3-mini matched the maintainers' workgroup label on **134 of 168** and the change-type tag on **153 of 182**.

<ArticleVideo
  src="/videos/decision-3-0-coreml/d3-browser-pilot.mp4"
  poster="/img/blog/decision-3-0-coreml/browser-pilot-poster.jpg"
  title="d3-mini drives a real website in Chrome"
  landscape
>
  Goal: open a book in the Travel category with a five-star rating. d3-mini picks each click from a screenshot.
</ArticleVideo>

**Drive a real website (d3-mini).** d3-mini gets a goal and a screenshot of [books.toscrape.com](https://books.toscrape.com) with numbered marks on the links. It picks the next click, reads the books on the page, and stops when one meets the goal. It reached **all 17 goals in a 17-goal test, at about 2.2 s per click**. d3 reads values well but compares them less reliably: one pick across a whole page reached 9 of 17. Reading each book card and comparing in code reached all 17.

<ArticleVideo
  src="/videos/decision-3-0-coreml/d3-live-video.mp4"
  poster="/img/blog/decision-3-0-coreml/live-video-poster.jpg"
  title="d3-mini watches a street crossing live"
  landscape
>
  Each time d3-mini is free, it looks at the last 4 seconds and the timeline grows from its answers. Video: diagonal crosswalk at Yonge &amp; Dundas, Toronto (Raysonho, Wikimedia Commons, CC0).
</ArticleVideo>

**Watch live video (d3-mini).** As the clip plays, d3-mini looks at the last 4 seconds (8 frames at its native 2 fps) and answers whether the cars are moving and whether someone is cycling. Each look takes **1.1 s on Core ML, against 2.4 s for PyTorch on the same Mac**, and answers stay about 2 seconds behind the live video.

<ArticleVideo
  src="/videos/decision-3-0-coreml/d3-mario.mp4"
  poster="/img/blog/decision-3-0-coreml/mario-poster.jpg"
  title="A distilled d3-lite plays Super Mario Bros. 1-1 on Core ML"
  landscape
>
  Every move, d3-lite answers three yes/no questions about close-ups just ahead of Mario. Right: its answers and time per decision. Bottom: GPU load and the decision log.
</ArticleVideo>

**Play Super Mario Bros.: a proof of concept.** d3 is a general decision model, not a game agent, and nothing here was particularly built for Mario. We gave it the setup JEV-9B used to clear World 1-1. While Mario is on the ground, the game pauses and d3 answers three plain yes/no questions about close-ups just ahead of him: is there an enemy, a pipe or step, a gap? Mario jumps if any answer is yes, and runs otherwise. The harness reads Mario's position from the emulator's memory and plays fixed button moves.

**d3-flash 9B clears 1-1 out of the box, on Core ML, in 56 decisions at 1.25 s per decision.** It gets no game-specific prompt, examples or training: just the same Choice / Noul interface it uses for routing and policy questions.

Distillation brings this down to d3-lite. As a proof of concept, a LoRA trained only on d3-flash's answers to the same three questions (states from 28 of the 32 levels, 4 held out) **clears 1-1 on Core ML in 65 decisions at 0.31 s per decision**, 4× faster than its teacher. The recipe doesn't depend on which d3 is the teacher, so the same approach could distil the 27B model. Distillation can also go beyond the teacher: when the labels are corrected by searching the emulator for moves that survive, a student clears levels that neither d3-flash nor d3 27B finishes on its own.

## Try it

The Core ML models for all three sizes are in [FluidInference/d3-coreml](https://huggingface.co/FluidInference/d3-coreml), and each folder has a Python host built on coremltools. For Swift apps, `D3Manager` in [FluidUse](https://github.com/FluidInference/FluidUse) downloads a model and answers requests.

Thanks to the vLLM Semantic Router team for early access to Decision 3.0.
