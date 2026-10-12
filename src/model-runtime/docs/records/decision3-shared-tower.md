# Decision 3.0 shared vision tower

In an image request, the exact profile runs the vision tower on one copy of
the request's images per question in a forward, as the released runtime's
processor hands them on. The copies' batch changes the features' rounding, so
reading the images once and reusing the features is approximate. Batches of
the approximate profiles (`batching`, `max_speed`) now do that
(`Decision3Model.run_approximate`).

- **Date:** 2026-10-13.
- **Device:** one AMD Instinct MI300X (gfx942) in the router's ROCm image
  `vllm-sr-rocm:latest` built 2026-10-11 (PyTorch 2.12.0, Triton 3.7.0,
  FLA 0.5.2), pinned kernel choices.
- **Model:** `vllm-sr/d3-flash` at the built-in revision `521525ef`.
- **Method:** each request ran through `Decision3Model.run` and then
  `run_approximate` in the same process. A request holds one image and three
  questions: the dataset's multiple-choice question with its options as a
  Choice, "What kind of image is this?" (photo, diagram, map, chart) and "Is
  the image a photograph?" as a Yes / No question.

## Answers

| Data | Requests | Decisions changed | Largest probability difference | Accuracy, exact | Accuracy, approximate |
| --- | --- | --- | --- | --- | --- |
| ScienceQA test, first 300 rows with an image (`derek-thomas/ScienceQA` `f18b0a70`, CC BY-SA 4.0) | 300 | 0 of 900 | 0.038 | 95.67% | 95.67% |
| MMMU validation, first 10 single-image multiple-choice rows of each subject (`MMMU/MMMU` `876ce5cb`, Apache-2.0) | 300 | 5 of 900 (4 dataset questions, 1 image kind) | 0.046 | 64.00% | 63.33% |

Accuracy is the dataset question's answer against the dataset's gold option.

## Latency

`vllm-sr/d3-flash`'s example receipt (968 tokens per question), median of 12
after 3, the tower's features recomputed every time:

| Questions | Exact | Approximate |
| --- | --- | --- |
| 1 | 70.6 ms | 72.2 ms |
| 2 | 130.2 ms | 103.1 ms |
| 3 | 189.7 ms | 133.0 ms |
| 8 | 492.1 ms | 283.8 ms |

On the datasets' smaller images (median 343 and 384.5 tokens per question) the
three-question requests took 55.7 and 64.3 ms exact, 52.6 and 57.4 ms
approximate, at the median.
