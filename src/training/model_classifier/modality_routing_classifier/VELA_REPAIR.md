# Vela Modality repair

`Vela-1.0-Encoder-307M-Modality` reads a **text request** and classifies the requested
output. It is not a vision encoder and does not establish support for image,
audio, or video inputs. Label IDs are unchanged:

| ID | Label | Requested output |
| --- | --- | --- |
| 0 | AR | Text, code, or editable source markup |
| 1 | DIFFUSION | A newly generated image |
| 2 | BOTH | A newly generated image and separate written output |

Describing an existing picture, writing alt text, returning SVG/Mermaid source,
and explaining how to draw something are AR tasks. A reference to an image in
the request is insufficient to infer image generation. Output text displayed
inside a requested image does not by itself require the separate BOTH class.

The new recipe uses CC0 DiffusionDB caption metadata, Apache-2.0 academic requests,
and reviewed Aya requests. It downloads no images. Original output instructions
are written in English, Chinese, Spanish, French, German, and Japanese; gallery
captions retain their source wording and are not claimed to have been translated.
Gallery contributor groups and template families are split before sampling.
Academic and Aya requests provide ordinary AR examples.

Fetch pinned files, hydrate the Aya sidecar, and build Domain data using the
[shared workflow](../sequence_repair/README.md), then run:

```bash
python -m src.training.model_classifier.modality_routing_classifier.prepare_vela_data \
  --sources artifacts/vela/sources --domain-corpus artifacts/vela/domain \
  --annotations artifacts/vela/annotations/aya-reviewed-labels-v1.jsonl \
  --output artifacts/vela/modality-v1

python -m src.training.model_classifier.modality_routing_classifier.extend_contracts \
  --corpus artifacts/vela/modality-v1 --output artifacts/vela/modality-v2
```

The extension adds train-only wording diversity for existing-picture analysis,
editable source output, completed image assets, and explicitly separate image
plus explanation. It recovers captions only from frozen training records and
verifies their source hashes; dev/test requests remain unchanged. These authored
output contracts are not naturally observed user instructions. Report their
quality separately from the human-authored Aya source, whose available reviewed
examples cover AR rather than all three labels.

Initialize a fresh adapter and head on the pinned encoder; use the old Modality
weights only as a baseline. Continue with the shared runner and inspect confusion
between AR, DIFFUSION, and BOTH across languages. Training loss near zero is not a
reason to select a checkpoint if unseen development wording regresses. The
context workflow provides task-bearing head/middle/tail stress and retains source
groups. Freeze the candidate and its actual base lineage before final test
comparison and native/ONNX export validation.
