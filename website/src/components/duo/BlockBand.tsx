import React, { useEffect, useRef } from 'react'
import { numberVar, onThemeChange, reducedMotion, rgbVar, rng } from './pixels'
import styles from './Ground.module.css'

export type BandTone = 'blue' | 'ember' | 'violet'

type Item = [number, number, number, number]

const ALPHA: Record<BandTone, number[]> = {
  blue: [0.05, 0.075, 0.1, 0.13],
  ember: [0.045, 0.07, 0.095, 0.125],
  violet: [0.05, 0.075, 0.1, 0.13],
}
// The ember band deepens towards the section's end; the others are even.
const TONE: Record<BandTone, (v: number) => number> = {
  blue: () => 1,
  ember: v => (v < 0.35 ? 0 : v < 0.7 ? 1 : 2),
  violet: () => 1,
}
const STEPS = [0.7, 0.45, 0.25, 0.1, 0]
const K = 2
const U = 4

function smoothstep(x: number): number {
  const t = Math.max(0, Math.min(1, x))
  return t * t * (3 - 2 * t)
}

/* Duo's section band (sections.js): out of the paper under the section's head as fine 8 px blocks, a calm mosaic of
 * 32 px tiles, then back into fine blocks before the section ends. It settles out of noise in five steps the first
 * time it is seen, and is drawn once per size after that. */
function mountBand(wrap: HTMLElement, tone: BandTone, seed: number): () => void {
  const cv = wrap.querySelector('canvas')
  const ctx = cv?.getContext('2d')
  const section = wrap.parentElement
  if (!cv || !ctx || !section) return () => undefined
  let styles: string[] = []
  let items: Item[] = []
  let fine: [number, number][] = []
  let w = 0
  let h = 0
  let settled = false
  let raf = 0

  const paint = () => {
    const rgb = rgbVar(`--band-${tone}`, '77 111 163')
    const boost = numberVar('--band-boost', 1)
    styles = ALPHA[tone].map(a => `rgba(${rgb}, ${Math.min(1, a * boost).toFixed(3)})`)
    styles.push(`rgb(${rgbVar(`--band-${tone}-accent`, rgb)})`)
  }

  const build = (): boolean => {
    w = section.clientWidth
    h = section.clientHeight
    if (!w || !h) return false
    const cols = Math.ceil(w / (U * K) / 4) * 4
    const rows = Math.ceil(h / (U * K))
    cv.width = cols * U
    cv.height = rows * U
    cv.style.width = `${cols * U * K}px`
    cv.style.height = `${rows * U * K}px`
    const head = section.querySelector('.site-section-intro')
    const origin = section.getBoundingClientRect().top
    const top = head ? Math.ceil((head.getBoundingClientRect().bottom - origin + 28) / (U * K)) : 0
    const inTop = Math.ceil((top + 9) / 4) * 4
    const inEnd = Math.floor((rows - 12) / 4) * 4
    const r = rng(seed * 977 + cols)
    items = []
    fine = []
    for (let row = top; row < rows; row++) {
      const tile = row >= inTop && row < inEnd
      if (tile && (row - inTop) % 4) continue
      const v = (row - top) / Math.max(1, rows - top)
      for (let c = 0; c < cols; c += tile ? 4 : 1) {
        if (tile) {
          items.push([c * U, row * U, 4 * U - 1, TONE[tone](v)])
          continue
        }
        const p = row < inTop ? smoothstep((row - top) / (inTop - top)) : smoothstep((rows - 1 - row) / (rows - inEnd))
        fine.push([c * U, row * U])
        if (r() < p * 0.85) {
          const accent = r() < 0.004 && p > 0.2 && p < 0.8
          items.push([c * U, row * U, U - 1, accent ? 4 : TONE[tone](v) + (r() < 0.3 ? 1 : 0)])
        }
      }
    }
    return true
  }

  const draw = (noise: number) => {
    ctx.clearRect(0, 0, cv.width, cv.height)
    const buckets: number[][][] = styles.map(() => [])
    const r = rng((noise * 1000) | 0)
    for (const it of items) {
      if (!noise || r() > noise * 0.65) buckets[Math.min(it[3], styles.length - 1)].push(it)
    }
    if (noise) {
      for (const f of fine) {
        if (r() < noise * 0.1) buckets[0].push([f[0], f[1], U - 1])
      }
    }
    buckets.forEach((bucket, k) => {
      if (!bucket.length) return
      ctx.fillStyle = styles[k]
      for (const it of bucket) ctx.fillRect(it[0], it[1], it[2], it[2])
    })
  }

  const settle = () => {
    if (settled || !items.length) return
    settled = true
    if (reducedMotion().matches) {
      draw(0)
      return
    }
    let k = -1
    let t0 = 0
    const frame = (t: number) => {
      if (!t0) t0 = t
      const due = Math.min(STEPS.length - 1, Math.floor((t - t0) / 130))
      if (due > k) {
        k = due
        draw(STEPS[k])
      }
      raf = k < STEPS.length - 1 ? requestAnimationFrame(frame) : 0
    }
    draw(1)
    raf = requestAnimationFrame(frame)
  }

  paint()
  if (!build()) return () => undefined
  const io = new IntersectionObserver((entries) => {
    if (entries[0]?.isIntersecting) {
      settle()
      io.disconnect()
    }
  }, { rootMargin: '0px 0px -12% 0px' })
  io.observe(wrap)
  const ro = new ResizeObserver(() => {
    if (Math.abs(section.clientWidth - w) < 1 && Math.abs(section.clientHeight - h) < 8) return
    if (build() && settled) draw(0)
  })
  ro.observe(section)
  const offTheme = onThemeChange(() => {
    paint()
    if (settled) draw(0)
  })

  return () => {
    if (raf) cancelAnimationFrame(raf)
    io.disconnect()
    ro.disconnect()
    offTheme()
  }
}

export default function BlockBand({ tone, seed }: { tone: BandTone, seed: number }): JSX.Element {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => (ref.current ? mountBand(ref.current, tone, seed) : undefined), [tone, seed])
  return (
    <div ref={ref} className={styles.blocks} aria-hidden="true">
      <canvas />
    </div>
  )
}
