// Export the pinned upstream Matplotlib SVGs without changing chart geometry.
// Install sharp 0.35.4 and make DejaVu Sans available to fontconfig first.
import { createRequire } from 'node:module'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const sharp = require(process.env.SHARP_MODULE || 'sharp')
const here = dirname(fileURLToPath(import.meta.url))
const names = [
  'omni-nano-pareto-arxiv',
  'omni-nano-pareto-nmsqa',
  'omni-nano-pareto-vehicle-sounds',
  'omni-mini-pareto-sibfleurs',
]

for (const name of names) {
  await sharp(resolve(here, `${name}.svg`), { density: 170 })
    .resize(2448, 1496)
    .png()
    .toFile(resolve(here, '../..', `${name}.png`))
}
