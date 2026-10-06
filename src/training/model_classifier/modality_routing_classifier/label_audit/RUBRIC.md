# Label-audit rubric: requested output modality

You are auditing labels for a prompt router. For each user message, decide what the assistant's reply must **contain** for the request to be fully answered. Judge each message on its own. You are not told the current label; do not guess it.

## Labels

- **A (AR)**: the reply is text only. Answers, explanations, code, lists, rewrites, translations, summaries, stories, plans, and prompts written for the user to copy.
- **D (DIFFUSION)**: the reply is an image (or several) and essentially nothing else. Image generation or editing, including terse Stable-Diffusion-style prompts (a subject plus style, medium, artist or quality tags, with no verb) and picture-shaped designs (logo, poster, banner, cover, avatar, wallpaper, illustration).
- **B (BOTH)**: the reply must contain text and images together. Either the user explicitly asks for visuals alongside explanation ("...and show a diagram", "with pictures", "illustrated guide", "infographic explaining...", "step by step with images"), or the deliverable is inherently text plus visuals (illustrated tutorial, comic or storyboard with captions, report with charts or figures).

## Rules (apply in order)

1. **No request means A.** If there is nothing to answer (song lyric, fragment, greeting), label A with tag `nor` or `frag`.
2. **Judge the deliverable, not the vocabulary.** Words such as picture, image, photo, draw, diagram do not make a row D or B when the user wants code, math, or how-to help. CSS that positions a picture, matplotlib code, and "how do I create an image with AI?" are all A. Code that draws something is A.
3. **Talking about images is A.** Questions about art, photography, image models or Stable Diffusion; asking for a caption, alt text, description or critique; asking for **a prompt to give an image tool**. Label A with tag `img` or `prm`.
4. **An image as the deliverable is D.** If the user also wants a written explanation, description or story alongside it, label B.
5. **Implicit benefit is a flag, not a label.** For a text-only request where visuals would clearly help (physical how-tos, anatomy, geometry, cooking technique), label A and set `vh`. Never upgrade to B on implicit benefit alone.
6. **Deliverables that usually mix text and visuals** (presentation, slide deck, brochure, flyer, website mockup). Visuals explicitly requested: B, or D if the picture is the whole thing (a poster). Visuals not mentioned: A with `vh` and `dvis`.
7. **Missing context** ("make it bluer", "another one"): judge the literal words. Edit language is D with confidence `L`.
8. **Non-English** rows are judged the same way, with tag `nen`. **Cut text** (marked `[...N chars cut...]`): judge what is visible; the tooling adds the truncation tag itself.

## Confidence

- `H` (default, omit): one label is clearly best.
- `M`: a reasonable reader could pick another label.
- `L`: genuinely ambiguous.

## Output protocol

One line per row: `ID LABEL [flags]`. LABEL is `A`, `D` or `B`. Flags may come in any order and are omitted when they are the default:

- `vh`: visuals would clearly help (rule 5, label A only)
- `M` or `L`: confidence
- tags: `img` about_images, `prm` prompt_writing, `nor` no_request, `frag` fragment or nonsense, `nen` non_english, `dvis` deliverable_visual, `amb` ambiguous

Examples: `12 A` and `13 D` and `14 B M` and `15 A vh dvis` and `16 A L frag`.

Output only the lines, with no commentary.
