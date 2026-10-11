---
slug: decision-3-0-on-apple-silicon
title: "Decision 3.0 on Apple Silicon: Text, Image and Video Decisions on a Mac"
description: "d3-lite, d3-nano and d3-mini converted to Core ML: triage 1,000 pull requests at 0.12 s each, drive a real website and watch live video, all on one Mac."
authors: [Alex-Wengg]
tags: [decision, core-ml, apple-silicon, multimodal, demos, semantic-router]
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

[Decision 3.0](https://huggingface.co/collections/vllm-sr/decision-30) brought text, image and video decisions to one open model family. We converted the three smaller sizes (**d3-lite 0.8B, d3-nano 2B and d3-mini 4B**) to Core ML and built demos for each input type. The model, the prompts and the answers stay on the device.

[**Get the Core ML models →**](https://huggingface.co/FluidInference/d3-coreml) · [**Swift runtime →**](https://github.com/FluidInference/FluidUse)

<!-- truncate -->

## Same contract, on device

The Core ML release keeps the upstream contract: one forward pass per question, the FP32 readout at the last prompt token, a probability for every option and no generated text.

- **Text graph:** int8 weights with fp16 compute, one graph per prompt-length bucket.
- **Vision tower:** fp32. Video frames go through it one frame pair at a time.
- **Host side:** embeddings, M-RoPE positions and the readout. The Python host builds every prompt with upstream's own runtime (run without weights), so prompts match upstream byte for byte.

Answers that differ from upstream PyTorch (bf16 on MPS) on the same requests:

| | d3-lite | d3-nano | d3-mini |
|---|---|---|---|
| Text (100 questions) | 2 | 0 | 1 |
| Images (receipt checks) | 1 / 24 | 0 / 24 | 0 / 48 |
| Video (4-second clips, 2 fps) | 0 / 28 | 1 / 28 | 0 / 28 |
| Per text question | ~46 ms | ~65 ms | ~160 ms |
| Per 4-second video look | 0.32 s | 0.57 s | 1.1 s |

All numbers are on one MacBook Pro (M5 Pro, 24 GB), running on the GPU.

## Triage pull requests at 8 per second

The demo at the top is about speed. A queue of 1,000 sample pull requests runs down the center. For each one, d3-lite reads the title, the start of the description and the changed files, and answers two questions:

- Which of the seven [vLLM-SR workgroups](/blog/join-vllm-sr-workgroups) owns it?
- What kind of change is it?

The card then flies into its workgroup. Each PR takes **0.12 s, so 8.3 PRs and 16.5 decisions per second**, for the whole run.

The pull requests are generated samples, so the run can last as long as the demo needs. On real vLLM-SR pull requests, d3-mini matched the maintainers' workgroup label on **134 of 168** and the title's change-type tag on **153 of 182**.

## Drive a real website

<ArticleVideo
  src="/videos/decision-3-0-coreml/d3-browser-pilot.mp4"
  poster="/img/blog/decision-3-0-coreml/browser-pilot-poster.jpg"
  title="d3-mini drives a real website in Chrome"
  landscape
>
  Goal: open a book in the Travel category with a five-star rating. d3-mini picks each click from a screenshot of the page.
</ArticleVideo>

d3-mini gets a goal and a screenshot of a live website ([books.toscrape.com](https://books.toscrape.com), built for scraping practice) with numbered marks on the links. It picks the next link to click, reads the books on the page, and stops when a book meets the goal.

On Core ML it reached **17 of 17 goals, at about 2.2 s per click**.

What made the difference: d3 reads values well but compares them less reliably. Asking for one pick across a whole page of books got 9 of 17 goals. Asking it to read each book card (title, rating, price) and doing the comparison in code got 17 of 17.

## Watch live video

<ArticleVideo
  src="/videos/decision-3-0-coreml/d3-live-video.mp4"
  poster="/img/blog/decision-3-0-coreml/live-video-poster.jpg"
  title="d3-mini watches a street crossing live"
  landscape
>
  Each time d3-mini is free, it looks at the last 4 seconds of video and the timeline grows from its answers. Video: diagonal crosswalk at Yonge &amp; Dundas, Toronto (Raysonho, Wikimedia Commons, CC0).
</ArticleVideo>

As the clip plays, d3-mini looks at the last 4 seconds (8 frames at its native 2 fps) and answers two questions: are the cars moving, and is someone cycling? The timeline on the right is built from those answers, as they arrive.

On Core ML each look takes **1.1 s, against 2.4 s for PyTorch on the same Mac**. Answers stay at about 2 seconds behind the live video, with 0 answer changes in 28 looks compared with PyTorch.

## Try it

The Core ML models for all three sizes are in [FluidInference/d3-coreml](https://huggingface.co/FluidInference/d3-coreml). Each folder has a Python host built on coremltools, and the Swift runtime is in [FluidUse](https://github.com/FluidInference/FluidUse).

Thanks to the vLLM Semantic Router team for early access to Decision 3.0.
