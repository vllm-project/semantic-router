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
  --corpus artifacts/vela/modality-v1 \
  --registry artifacts/vela/annotations/output-contracts.json \
  --output artifacts/vela/modality-expanded
```

The optional extension requires an explicit registry. `source_templates` maps
each language to the original AR template list, indexed by the source row's
`template_family` suffix. `contracts` maps the same languages to supplied
AR, DIFFUSION, and BOTH template lists. Each template has exactly one `{}`
caption field. A minimal structural example is:

```json
{
  "version": 1,
  "source_templates": {"en": ["Describe: {}"]},
  "contracts": {
    "en": {
      "AR": ["Explain in text: {}"],
      "DIFFUSION": ["Generate an image: {}"],
      "BOTH": ["Generate an image and explain it in text: {}"]
    }
  }
}
```

Use the actual source wrappers and reviewed task instructions for a dataset;
the example is not a training corpus. The builder verifies each recovered
caption's hash, retains its source parent, and appends variants only to train.
Development and test requests retain their content and labels. Registry labels
are supplied judgments, not proof of independent review or natural user
requests. Keep the registry and its hash with the dataset artifacts.

Initialize a fresh complete head on the Vela Base at an immutable revision;
use `--method full --fresh-head` with the shared runner and explicitly provide
`--base-id llm-semantic-router/Vela-1.0-Encoder-307M` and `--base-revision`.
Inspect confusion between AR, DIFFUSION, and BOTH across languages. Training loss near zero is not a
reason to select a checkpoint if unseen development wording regresses. The
context workflow provides task-bearing head/middle/tail stress and retains source
groups. Freeze the candidate and its actual base lineage before final test
comparison and native/ONNX export validation.
