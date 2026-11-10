# Ecosystem logo assets

These assets use the logo artwork supplied in the project's Ecosystem
presentation. Original colored symbols, gradients, geometry, and internal
black/white details are preserved. No marks were redrawn and no gradients were
added.

Dark-background lettering variants keep the source geometry while making dark
type readable: Intel, NVIDIA, Nutanix, KR Labs, UIC, NYU, and the project wordmark
retain their colored elements with white lettering. Chicago and Berkeley use
white wordmarks; the monochrome AMD, UBS, AI21, Liquid, and MBZUAI marks also use
white. These are display adaptations, not claims of official dark-mode assets.
DaoCloud's charcoal background was removed while retaining its green symbol and
white wordmark. White areas within colored seals and symbols remain intact.

The `*-light.svg` variants use the original artwork with dark lettering and
colored symbols. DaoCloud's light variant changes only its white lettering to
charcoal, preserving the green symbol and transparent geometry. This is a
display adaptation, not an official light-mode asset. Logos that are already
readable on both backgrounds share the same asset.

Each self-contained SVG embeds a losslessly compressed, cropped PNG. No runtime
filter, remote image, or font resource is required. These are raster-backed SVGs,
not newly drawn vector logos. The NTU seal is exported at 512 × 512 pixels for
display, retaining full-color PNG without palette quantization.
The `vllm-sr-wordmark-dark` and `vllm-sr-wordmark-light` PNG/SVG pairs each contain
the same artwork, including the original yellow and blue V. Both are
2160 × 690 pixels.

`atmosphere.webp` is the generated background plate.

## README logo wall

The repository README shows `ecosystem-light-theme.png` on GitHub's light theme
and `ecosystem-dark-theme.png` on its dark theme, through a `<picture>` element.
Both are transparent PNGs, 2000 pixels wide, rendered from these SVGs by
`website/scripts/build-ecosystem-composite.mjs`. Like `EcosystemSection.tsx`, the
light-theme wall uses each `*-light.svg` variant where one exists and the
dark-theme wall uses the plain SVGs. The script lays the organizations out in the
homepage's order, in rows of five (the fourth row has four), and sizes each logo
to the same optical weight: the same box area, adjusted for ink coverage, within
height, width and upscaling limits.

When the organizations change, update the script's `rows` and the README's alt
text to match `EcosystemSection.tsx`, then render both PNGs from the repository
root:

```sh
docker run --rm --init --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/repo -w /repo \
  mcr.microsoft.com/playwright:v1.57.0-noble sh -c '
    npm install --prefix /tmp/deps --no-save --no-audit --no-fund playwright@1.57.0 sharp@0.34.5 >/dev/null &&
    NODE_PATH=/tmp/deps/node_modules node website/scripts/build-ecosystem-composite.mjs'
```

The captions load Inter from Google Fonts, so the container needs network access.
`--preview DIR` also writes each wall flattened on its GitHub background
(`#ffffff` or `#0d1117`) for review.
