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

`atmosphere.webp` is the generated background plate. `ecosystem.webp` is a static
export of the homepage ecosystem section for the repository README. Keep its
logo order and copy synchronized with `EcosystemGrid.tsx`.
