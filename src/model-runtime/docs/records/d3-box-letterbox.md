# Decision 3.0: letterboxing an image Box on wide, short images

On a wide, short image d3 reads only a few merged rows across the short side.
Its Box answer then places the short-axis edges wrong at high confidence
(#7011). The family options `box_pad_rows` and `box_pad_min_aspect` letterbox
that image for the Box alone. The resized photo is padded on its short axis
with a neutral grey (238), centred, to `box_pad_rows` merged rows. The Box is
read on that canvas and its edges are mapped back to the photo, clipped to
[0, 1]. The option is off by default.

A Box is letterboxed only when all of these hold:

- it is an image Box (no `at`) over the state's only image and has no image
  of its own;
- the resized image is at least `box_pad_min_aspect`:1 and its short side is
  under `box_pad_rows` merged rows;
- the canvas stays within the image pixel budget;
- the letterboxed Box and the request stay within their token caps (otherwise
  the request is planned as sent).

Every other question of the request keeps the photo as sent.

- **Date:** 2026-11-22.
- **Models:** d3-ultra (revision 3ac9ff15) and d3-turbo (revision 689b3af1),
  `--dtype auto`, approximate numerics, `max_num_seqs` 1,
  `max_num_batched_tokens` 4,096, result cache off.
- **Devices:** one AMD MI300X with the ROCm router image, and one NVIDIA GB10
  (d3-ultra only) with the CUDA router image. The change's `vllm_srun` was
  mounted over each image's package.
- **Options:** `box_pad_rows: 8`, `box_pad_min_aspect: 4`.
- **Data:** BoxSet-dev only; the test split was not scored.

## Images of ordinary proportions do not change

There were 300 BoxSet-dev images as sent, none of them at 4:1. On the
MI300X, every one answered with the same input tokens and the same IoU with
the option on and off, for both models. The decision3 golden image
(64 x 48, read at 7 merged rows) is under 4:1 and is not letterboxed.

The other questions of a request read the photo as sent:
`tests/test_box_padding.py` checks that a Yes / No beside a letterboxed Box
keeps its token IDs.

## Wide and tall crops

BoxSet-dev has no single-image row at 4:1 or more, so the wide, short images
are crops. Each is a crop of a BoxSet-dev image that keeps the whole annotated
box, at 4, 6, 8 or 16:1, wide and tall. It is resized so that its short side
is 96 to 224 px (3 to 7 merged rows), so every crop is letterboxed. There are
312 crops of distinct source images, disjoint from the 300 images above. On
the MI300X:

| Model | Option | Mean IoU | IoU >= 0.5 | Wrong at confidence >= 0.9 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d3-ultra | off | 0.494 | 174 / 312 | 50 | 0.235 | 0.218 |
| d3-ultra | on | 0.664 | 235 / 312 | 10 | 0.077 | 0.082 |
| d3-turbo | off | 0.502 | 184 / 312 | 44 | 0.210 | 0.211 |
| d3-turbo | on | 0.674 | 242 / 312 | 12 | 0.064 | 0.088 |

Against the option off, d3-ultra moved 68 crops from a miss (IoU < 0.5) to a
hit and 7 from a hit to a miss. d3-turbo moved 64 and 6.

The images as sent are the reference: ECE 0.019 (d3-ultra) and 0.020
(d3-turbo). Their confidence >= 0.9 bucket hits 0.989 of the time at a mean
confidence of 0.988. On the letterboxed crops the same bucket hits 0.955
(d3-ultra) and 0.946 (d3-turbo) at 0.987 and 0.988. The ECE gap to the
reference has a bootstrap 95% interval of [0.024, 0.087] and [0.014, 0.078].
Without the option it was [0.165, 0.261] and [0.138, 0.235]. Letterboxing
removes most of the overconfidence on these images, but not all of it.
`box_calibration` was fitted on images as sent and was not refitted. 6:1 is
the worst ratio (ECE 0.137 and 0.103).

## A second panel and the GB10

A separate panel has 72 BoxSet-dev sources: 48 crops at 8 and 16:1
(1,600 px long), 18 crops at 4, 6 and 8:1 (224 or 256 px short side) and six
images as sent under 3:1. On it, 60 sources are letterboxed:

| Device | Model | Option | Mean IoU | IoU >= 0.5 |
| --- | --- | --- | ---: | ---: |
| MI300X | d3-ultra | off / on | 0.556 / 0.728 | 41 / 61 |
| MI300X | d3-turbo | off / on | 0.556 / 0.715 | 43 / 62 |
| GB10 | d3-ultra | off / on | 0.557 / 0.725 | 41 / 59 |

On the MI300X, the 12 sources that are not letterboxed answered identically
with the option on and off. On the GB10, a rerun of the same request moved
one source's IoU by up to 0.057. The two devices differ per source by up to
0.066 IoU with the option on.

## Cost

A letterboxed Box reads more image tokens. The 312 crops took 108 more on
average, with a median of 90 and a maximum of 260, over 364 input tokens per
request without the option. When the request has other questions, the
letterboxed state is a second state, read once more for the Box.

Latency was measured at one request at a time. The MI300X median stayed at
0.14 s. The GB10 median on the 72-source panel went from 3.4 s to 4.5 s.

## Not measured

- Wide, short images that were not cropped. The training rows hold only four
  such images (4.05 to 6.15:1). Their five questions answered as a smoke test,
  not evidence, since the model was trained on them: one miss became a hit,
  and one hit on a target 5.5% of the image wide became a miss at low
  confidence.
- A calibration map for letterboxed Boxes, and the BoxSet test split.
- Other d3 sizes, and the GB10 for d3-turbo.
