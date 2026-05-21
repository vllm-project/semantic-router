# Image fixtures for multimodal-routing e2e profile

Each image is a small, license-clean fixture used by the `embedding-signal-image-routing` testcase to verify that the image-modality routing rules in the multimodal-routing profile cosine-match real visual content in their target categories.

## Fixture inventory

| File | Size | Target rule (positive case) | License |
|---|---|---|---|
| `passport_sample.jpg` | 45 KB | `identifier_document_imagery` | Gemini-generated, no upstream copyright |
| `code_screenshot.jpg` | 15 KB | `code_or_terminal_imagery` | Gemini-generated, no upstream copyright |
| `conference_room.jpg` | 30 KB | `ambient_office_imagery` | Gemini-generated, no upstream copyright |

All three images were generated 2026-05-09 via Google Gemini (aistudio.google.com). Generated content has no upstream copyright holder. Original Gemini output (1024x1024 PNG, several MB each) was smart-cropped + downscaled to 384x384 to match SigLIP-base's native input resolution, then re-encoded as JPEG quality 85 for photographic content. Final sizes under 50 KB each.

## Why real images instead of synthetic PNGs

The earlier version of this fixture used a single 32x32 synthetic PNG reused across all test cases. That fixture exercised the pipeline (request-path image extractor + candle-binding image encoder + cosine math + response header propagation) but did NOT exercise rule specificity. A synthetic PNG's embedding lands wherever it happens to land in the shared multimodal space; it could pass any rule's threshold by accident, or fail all of them. A test that "passes" against synthetic input doesn't prove the runtime correctly identifies the right rule for the right image.

The three real images here are visually distinct by design (each prompt describes a different photographic category) and let the testcase assert a 3x3 matrix: each image should FIRE its target rule and should NOT fire the two sibling rules. That's the rule-specificity assertion the synthetic-PNG fixture never made.

## Regeneration prompts

Reproducible via Gemini if regeneration is ever needed.

### passport_sample.jpg (positive for `identifier_document_imagery`)

> A flat lay photograph of a fictional sample passport open on a wooden desk surface. The visible page shows a placeholder photo, the heading "SPECIMEN PASSPORT," and clearly fictitious filler data ("JANE SAMPLE," issue date "01 JAN 2030," country code "ZZ"). Realistic government-document visual style with security-pattern background. Centered composition, soft overhead lighting. No real names, real signatures, real photos, or real personal information. No copyrighted security elements from any real country's passport design.

### code_screenshot.jpg (positive for `code_or_terminal_imagery`)

> A screenshot of programming source code displayed in a dark-mode editor with syntax highlighting. Monospace font with line numbers along the left margin. Keywords like "func," "if," "return" highlighted in blue and green. String literals in orange. Code shows a generic Go-style function definition with curly braces and indentation, about 25-30 lines visible. The editor window has a tab bar at the top and a status bar at the bottom but no specific brand or product logos. Plain dark gray editor chrome.

### conference_room.jpg (positive for `ambient_office_imagery`)

> A photograph of an empty modern conference room interior. A large whiteboard on one wall with simple geometric shapes drawn in colored marker (no readable text or words). A long meeting table in the center with chairs around it. Soft natural daylight coming through floor-to-ceiling windows on one side. Neutral wall colors, minimalist office aesthetic. No people in the frame. No company logos, brand names, or specific identifying signage anywhere in the scene.

## Adding new fixtures

If you add a new image fixture, document it in the inventory table above with its target rule and license posture. Keep individual file sizes under 50 KB; the testdata directory is checked into the repo and bloating it slows clones.
