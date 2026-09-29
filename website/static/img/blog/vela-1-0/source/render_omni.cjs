#!/usr/bin/env node
'use strict'

// Render trusted, self-contained SVG files. Requires playwright and sharp.
const fs = require('node:fs')
const path = require('node:path')

function parseArgs(argv) {
  const opts = { title: 'Vela Omni architectures' }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--help' || arg === '-h') {
      console.log('Usage: node render_omni.cjs OUTPUT_DIR [--title TEXT] [--node-modules PATH] [--browser PATH]\nRequires playwright and sharp; NODE_PATH is also supported.')
      process.exit(0)
    }
    if (['--title', '--node-modules', '--browser'].includes(arg)) {
      if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error(`Missing value for ${arg}`)
      opts[arg.slice(2)] = argv[++i]
    }
    else if (!arg.startsWith('-') && !opts.directory) opts.directory = path.resolve(arg)
    else throw new Error(`Unknown argument: ${arg}`)
  }
  if (!opts.directory) throw new Error('OUTPUT_DIR is required; use --help for usage.')
  return opts
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', '\'': '&#39;' }[char]))
}

async function main() {
  const opts = parseArgs(process.argv.slice(2))
  const dependency = name => require(opts['node-modules'] ? path.join(path.resolve(opts['node-modules']), name) : name)
  const { chromium } = dependency('playwright')
  const sharp = dependency('sharp')
  const root = opts.directory
  const files = fs.readdirSync(root).filter(file => ['07-omni-nano.svg', '08-omni-mini.svg'].includes(file) && fs.statSync(path.join(root, file)).isFile()).sort()
  if (!files.length) throw new Error(`No SVG files in ${root}`)
  const browser = await chromium.launch({ headless: true, ...(opts.browser ? { executablePath: path.resolve(opts.browser) } : {}) })
  const checks = []
  try {
    const page = await browser.newPage({ deviceScaleFactor: 2 })
    for (const file of files) {
      const svg = fs.readFileSync(path.join(root, file), 'utf8')
      await page.setContent(`<html><body>${svg}</body></html>`)
      const dimensions = await page.evaluate(() => {
        const svg = document.querySelector('svg')
        if (!svg) throw new Error('Missing SVG root')
        // Fixed px dimensions or a viewBox are required; percentage sizes are ambiguous.
        const dimension = (name) => {
          const raw = svg.getAttribute(name)
          if (raw && !/^\d+(?:\.\d+)?(?:px)?$/.test(raw)) throw new Error(`SVG ${name} must use px or unitless values`)
          return raw ? Number.parseFloat(raw) : svg.viewBox.baseVal[name]
        }
        return { width: dimension('width'), height: dimension('height') }
      })
      const { width, height } = dimensions
      if (![width, height].every(value => Number.isInteger(value) && value > 0)) throw new Error(`${file}: use positive integer px dimensions or an integer viewBox`)
      await page.setViewportSize({ width, height })
      await page.setContent(`<!doctype html><html><head><style>@page{size:${width}px ${height}px;margin:0}html,body{margin:0;width:${width}px;height:${height}px;background:transparent}body>svg{display:block;width:${width}px;height:${height}px}</style></head><body>${svg}</body></html>`)
      await page.evaluate(() => document.fonts.ready)
      const check = await page.evaluate(({ width, height }) => {
        const root = document.querySelector('svg')
        const viewport = root.getBoundingClientRect()
        const bounds = (el, relativeTo) => {
          const b = el.getBBox()
          let matrix = el.getScreenCTM()
          if (relativeTo) matrix = relativeTo.getScreenCTM().inverse().multiply(matrix)
          const corners = [[b.x, b.y], [b.x + b.width, b.y], [b.x, b.y + b.height], [b.x + b.width, b.y + b.height]]
            .map(([x, y]) => new DOMPoint(x, y).matrixTransform(matrix))
          const xs = corners.map(p => p.x - (relativeTo ? 0 : viewport.left))
          const ys = corners.map(p => p.y - (relativeTo ? 0 : viewport.top))
          return { text: el.textContent, x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) }
        }
        const boxes = []
        for (const group of root.querySelectorAll('g[data-box]')) {
          const box = group.dataset.box.trim().split(/[ ,]+/).map(Number)
          if (box.length !== 4 || !box.every(Number.isFinite)) throw new Error('data-box requires x y width height')
          const [x, y, w, h] = box
          for (const el of group.querySelectorAll('text')) {
            if (el.closest('g[data-box]') !== group) continue
            const b = bounds(el, group)
            if (b.w && (b.x < x + 8 || b.y < y + 4 || b.x + b.w > x + w - 8 || b.y + b.h > y + h - 4)) boxes.push({ ...b, container: box })
          }
        }
        const outside = [...root.querySelectorAll('text')].map(el => bounds(el)).filter(b => b.w && (b.x < -0.1 || b.y < -0.1 || b.x + b.w > width + 0.1 || b.y + b.h > height + 0.1))
        return { boxes, outside }
      }, dimensions)
      checks.push({ file, width, height, ...check })
      const stem = path.join(root, path.basename(file, '.svg'))
      await page.screenshot({ path: `${stem}-transparent.png`, omitBackground: true })
      await sharp(`${stem}-transparent.png`).flatten({ background: '#ffffff' }).png().toFile(`${stem}.png`)
      await page.pdf({ path: `${stem}.pdf`, width: `${width}px`, height: `${height}px`, printBackground: true, preferCSSPageSize: true })
    }
  }
  finally {
    await browser.close()
  }
  fs.writeFileSync(path.join(root, 'layout-check.json'), JSON.stringify(checks, null, 2) + '\n')
  const tileW = 700, tileH = 960, pad = 22
  const columns = Math.min(3, files.length), rows = Math.ceil(files.length / columns)
  const tiles = []
  for (const [i, file] of files.entries()) {
    const input = await sharp(path.join(root, file.replace(/\.svg$/, '.png'))).resize(tileW, tileH, { fit: 'contain', background: '#fff' }).flatten({ background: '#fff' }).toBuffer()
    tiles.push({ input, left: pad + (i % columns) * (tileW + pad), top: pad + Math.floor(i / columns) * (tileH + pad) })
  }
  await sharp({ create: { width: columns * tileW + (columns + 1) * pad, height: rows * tileH + (rows + 1) * pad, channels: 3, background: '#e8e8e8' } }).composite(tiles).png().toFile(path.join(root, 'overview.png'))
  const sections = files.map((file) => {
    const stem = file.slice(0, -4), href = suffix => encodeURIComponent(stem + suffix)
    return `<section><h2>${escapeHtml(stem)}</h2><img src="${href('.svg')}" alt="${escapeHtml(stem)}"><p><a href="${href('.svg')}">Editable SVG</a><a href="${href('.pdf')}">Vector PDF</a><a href="${href('.png')}">White PNG</a><a href="${href('-transparent.png')}">Transparent PNG</a></p></section>`
  }).join('\n')
  fs.writeFileSync(path.join(root, 'index.html'), `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escapeHtml(opts.title)}</title><style>body{font:16px system-ui;background:#eee;color:#111;margin:0}main{max-width:1400px;margin:40px auto;padding:20px}h1{font-size:27px;font-weight:500}h2{font-size:18px;font-weight:500}section{background:white;padding:24px;margin:28px 0}img{display:block;max-width:100%;max-height:1500px;margin:auto}a{color:#333;margin-right:20px}</style><main><h1>${escapeHtml(opts.title)}</h1><p><a href="architecture-atlas.pdf">PDF atlas (after verification)</a></p>${sections}</main></html>\n`)
  const issues = checks.filter(check => check.boxes.length || check.outside.length)
  console.log(JSON.stringify({ figures: files.length, output: root, issues }, null, 2))
  if (issues.length) process.exitCode = 1
}

main().catch((error) => {
  console.error(error.message)
  process.exitCode = 1
})
