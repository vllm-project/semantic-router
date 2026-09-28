# Multimodal routing image fixtures

These images support the `embedding-signal-image-routing` E2E testcase. The
test pairs every image with all three image-routing rules, checking one
positive match and two cross-category non-matches for each fixture.

## Inventory

| File | Intended positive rule | Source |
| --- | --- | --- |
| `passport_sample.jpg` | `identifier_document_imagery` | AI-generated fictional specimen document |
| `code_screenshot.jpg` | `code_or_terminal_imagery` | AI-generated code editor scene |
| `conference_room.jpg` | `ambient_office_imagery` | AI-generated office scene |

The images were generated with Google Gemini on 2026-05-09. The recorded
workflow used text prompts rather than third-party source assets, and the
fixtures are designed to contain no real identity data or product logos. The
original 1024×1024 PNG outputs were cropped and downscaled to 384×384, then
encoded as JPEG fixtures.

The exact case matrix, expected decisions, and request text live in
[`../embedding_signal_image_cases.json`](../embedding_signal_image_cases.json).
Category-distinct images are required because a generic synthetic pattern does
not provide a stable positive and negative routing contract.

## Source prompts

The prompts are provenance records, not golden model outputs:

- `passport_sample.jpg`: a clearly fictional “SPECIMEN PASSPORT” on a desk,
  with placeholder identity fields and no real national design.
- `code_screenshot.jpg`: a dark editor showing generic Go-like source code,
  line numbers, and no product branding.
- `conference_room.jpg`: an empty meeting room with a table, chairs,
  whiteboard, and no people or company signage.

Regeneration can change embeddings even when the prompt is unchanged. Treat a
replacement as a fixture change: run the multimodal-routing E2E profile and
review all nine positive/cross-category cases.

## Adding or replacing a fixture

- use content that is safe to distribute in this repository and record its
  source here;
- keep the image at 384×384 and below 50 KiB;
- add or update the corresponding cases in
  `embedding_signal_image_cases.json`;
- verify both the intended positive rule and sibling-rule non-matches.
