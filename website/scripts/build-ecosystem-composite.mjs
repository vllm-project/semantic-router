#!/usr/bin/env node
// Renders the README's ecosystem logo wall as transparent PNGs, one per GitHub theme.
// Runs inside the Playwright container; see website/static/img/ecosystem/README.md.
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const { chromium } = require('playwright')
const sharp = require('sharp')

const assetDir = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'static', 'img', 'ecosystem')

// Same organizations and order as EcosystemSection.tsx, in the README's rows.
const rows = [
  ['amd', 'hugging-face', 'microsoft', 'intel', 'nvidia'],
  ['red-hat', 'ibm', 'liquid', 'daocloud', 'delta'],
  ['mbzuai', 'mcgill', 'kr-labs', 'university-of-chicago', 'uc-berkeley'],
  ['umass-boston', 'uic', 'national-taiwan-university', 'nyu'],
  ['ubs', 'ai21', 'bayer', 'dell', 'nutanix'],
]
const captions = {
  'mbzuai': 'MBZUAI',
  'umass-boston': 'UMass Boston',
  'national-taiwan-university': 'National Taiwan University',
}

const themes = [
  {
    name: 'light',
    background: '#ffffff',
    caption: '#59636e',
    logo: id => existsSync(join(assetDir, `${id}-light.svg`)) ? `${id}-light.svg` : `${id}.svg`,
  },
  {
    name: 'dark',
    background: '#0d1117',
    caption: '#9198a1',
    logo: id => `${id}.svg`,
  },
]

const layout = {
  width: 1000,
  scale: 2,
  rowHeight: 88,
  // Every logo gets the same box area, weighted by its ink coverage to this power.
  area: 1910,
  coverageWeight: 0.75,
  minCoverage: 0.3,
  maxHeight: 52,
  maxCaptionedHeight: 42,
  maxWidth: 150,
  maxUpscale: 1.4,
}

function parseArgs(argv) {
  const options = { out: assetDir, preview: null }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--out' || arg === '--preview') {
      if (!argv[i + 1]) throw new Error(`Missing value for ${arg}`)
      options[arg.slice(2)] = resolve(argv[++i])
    }
    else if (arg === '--help' || arg === '-h') {
      console.log('Usage: node website/scripts/build-ecosystem-composite.mjs [--out DIR] [--preview DIR]')
      process.exit(0)
    }
    else {
      throw new Error(`Unknown argument: ${arg}`)
    }
  }
  return options
}

async function measure(file) {
  const svg = readFileSync(join(assetDir, file))
  const { data, info } = await sharp(svg).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  let alpha = 0
  for (let i = 3; i < data.length; i += 4) alpha += data[i]
  return { svg, width: info.width, height: info.height, coverage: alpha / 255 / (info.width * info.height) }
}

function fit({ width, height, coverage }, captioned) {
  const ratio = width / height
  const h = Math.min(
    Math.sqrt(layout.area / (ratio * Math.max(coverage, layout.minCoverage) ** layout.coverageWeight)),
    captioned ? layout.maxCaptionedHeight : layout.maxHeight,
    layout.maxWidth / ratio,
    (height * layout.maxUpscale) / layout.scale,
  )
  return { width: Math.round(h * ratio), height: Math.round(h), upscale: (h * layout.scale) / height }
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', '\'': '&#39;' }[char]))
}

async function composite(theme) {
  const cells = await Promise.all(rows.map(row => Promise.all(row.map(async (id) => {
    const logo = await measure(theme.logo(id))
    const caption = captions[id]
    const size = fit(logo, Boolean(caption))
    const src = `data:image/svg+xml;base64,${logo.svg.toString('base64')}`
    const figcaption = caption ? `<figcaption>${escapeHtml(caption)}</figcaption>` : ''
    return { id, size, html: `<figure><img src="${src}" width="${size.width}" height="${size.height}" alt="">${figcaption}</figure>` }
  }))))
  const html = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..700&display=swap">
<style>
html, body { margin: 0; background: transparent; }
main { width: ${layout.width}px; }
.row { display: grid; grid-template-columns: repeat(var(--columns), minmax(0, 1fr)); height: ${layout.rowHeight}px; align-items: center; justify-items: center; }
figure { display: flex; flex-direction: column; align-items: center; gap: 7px; margin: 0; }
img { display: block; }
figcaption { color: ${theme.caption}; font: 500 13px/1.2 Inter, sans-serif; white-space: nowrap; }
</style>
</head>
<body><main>${cells.map(row => `<div class="row" style="--columns: ${row.length}">${row.map(cell => cell.html).join('')}</div>`).join('')}</main></body>
</html>`
  return { html, cells: cells.flat() }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const browser = await chromium.launch()
  try {
    const context = await browser.newContext({ viewport: { width: layout.width, height: 800 }, deviceScaleFactor: layout.scale })
    const page = await context.newPage()
    for (const theme of themes) {
      const { html, cells } = await composite(theme)
      await page.setContent(html, { waitUntil: 'networkidle' })
      const fontLoaded = await page.evaluate(async () => {
        await document.fonts.load('500 13px Inter')
        await Promise.all([...document.images].map(image => image.decode()))
        return [...document.fonts].some(face => face.family.replace(/"/g, '') === 'Inter' && face.status === 'loaded')
      })
      if (!fontLoaded) throw new Error('Inter did not load; the captions need network access to Google Fonts')
      const screenshot = await page.locator('main').screenshot({ omitBackground: true })
      const output = join(options.out, `ecosystem-${theme.name}-theme.png`)
      mkdirSync(options.out, { recursive: true })
      await sharp(screenshot).png({ compressionLevel: 9 }).toFile(output)
      if (options.preview) {
        mkdirSync(options.preview, { recursive: true })
        await sharp(output).flatten({ background: theme.background }).png().toFile(join(options.preview, `ecosystem-${theme.name}-theme-on-${theme.background.slice(1)}.png`))
      }
      const sizes = cells.map(({ id, size }) => `${id} ${size.width}x${size.height}${size.upscale > 1 ? ` (upscaled ${size.upscale.toFixed(2)}x)` : ''}`)
      console.log(`${output}\n  ${sizes.join(', ')}`)
    }
  }
  finally {
    await browser.close()
  }
}

main().catch((error) => {
  console.error(error.message)
  process.exitCode = 1
})
