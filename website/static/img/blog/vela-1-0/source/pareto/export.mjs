// Export the pinned upstream Matplotlib SVGs without changing chart geometry.
// Install sharp 0.35.4 and make DejaVu Sans available to fontconfig first.
import { createRequire } from 'node:module'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const sharp = require(process.env.SHARP_MODULE || 'sharp')
const here = dirname(fileURLToPath(import.meta.url))
const selected = [
  'omni-nano-pareto-imdb',
  'omni-nano-pareto-nmsqa',
  'omni-mini-pareto-mridingham',
  'omni-mini-pareto-sibfleurs',
]
const general = [
  'omni-nano-general-english41',
  'omni-mini-general-english41',
  'omni-nano-general-audio19',
  'omni-mini-general-audio19',
]

for (const name of [...general, ...selected]) {
  const [width, height] = general.includes(name) ? [2924, 1700] : [2448, 1496]
  await sharp(resolve(here, `${name}.svg`), { density: 170 })
    .resize(width, height)
    .png()
    .toFile(resolve(here, '../..', `${name}.png`))
}
