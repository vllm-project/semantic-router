import React, { useEffect, useRef } from 'react'
import clsx from 'clsx'
import { onThemeChange, reducedMotion, rgbVar, rng } from './pixels'
import styles from './Ground.module.css'

type Square = { d: number, x: number, y: number, kind: 'ink' | 'ember' | 'sky', a: number }

/* Duo's mosaic rule: three rows of 6 px squares, 2 px apart, densest at the centre, drawn outwards from the centre
 * the first time it is seen (at once under prefers-reduced-motion). */
function mountRule(cv: HTMLCanvasElement, seed: number): () => void {
  const ctx = cv.getContext('2d')
  if (!ctx) return () => undefined
  let pattern: Square[] = []
  let width = 0
  let drawn = false
  let raf = 0

  const build = (w: number, h: number) => {
    const step = 8
    const cols = Math.floor(w / step)
    const x0 = (w - cols * step + 2) / 2
    const y0 = (h - 3 * step + 2) / 2
    const r = rng(cols * 31 + seed)
    const mid = cols / 2
    pattern = []
    for (let c = 0; c < cols; c++) {
      const d = Math.abs(c - mid) / mid
      for (let row = 0; row < 3; row++) {
        const p = Math.exp(-d * d * 4.2) * (row === 1 ? 1 : 0.62)
        const q = r()
        const accent = r()
        if (q > p) continue
        const kind = accent < 0.035 ? 'ember' : accent < 0.06 ? 'sky' : 'ink'
        pattern.push({ d, x: x0 + c * step, y: y0 + row * step, kind, a: 0.1 + p * 0.55 })
      }
    }
    width = w
  }

  const draw = (progress: number) => {
    const w = cv.clientWidth
    const h = cv.clientHeight
    if (!w || !h) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    if (cv.width !== Math.round(w * dpr)) {
      cv.width = Math.round(w * dpr)
      cv.height = Math.round(h * dpr)
    }
    if (width !== w) build(w, h)
    const ink = rgbVar('--rule-ink', '22 24 29')
    const fills = { ember: `rgb(${rgbVar('--band-ember-accent', '217 68 31')})`, sky: `rgb(${rgbVar('--band-blue-accent', '127 162 216')})` }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    for (const s of pattern) {
      if (s.d > progress) continue
      ctx.fillStyle = s.kind === 'ink' ? `rgba(${ink}, ${s.a.toFixed(3)})` : fills[s.kind]
      ctx.fillRect(s.x, s.y, 6, 6)
    }
  }

  const settle = () => {
    if (drawn) return
    drawn = true
    if (reducedMotion().matches) {
      draw(1)
      return
    }
    const t0 = performance.now()
    const frame = (t: number) => {
      const k = Math.min(1, (t - t0) / 700)
      draw(1 - (1 - k) ** 3)
      raf = k < 1 ? requestAnimationFrame(frame) : 0
    }
    raf = requestAnimationFrame(frame)
  }

  const io = new IntersectionObserver((entries) => {
    if (entries[0]?.isIntersecting) {
      settle()
      io.disconnect()
    }
  }, { rootMargin: '0px 0px -8% 0px' })
  io.observe(cv)
  const ro = new ResizeObserver(() => {
    if (drawn && cv.clientWidth !== width) draw(1)
  })
  ro.observe(cv)
  const offTheme = onThemeChange(() => {
    if (drawn) draw(1)
  })

  return () => {
    if (raf) cancelAnimationFrame(raf)
    io.disconnect()
    ro.disconnect()
    offTheme()
  }
}

export default function MosaicRule({ seed = 1, className }: { seed?: number, className?: string }): JSX.Element {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => (ref.current ? mountRule(ref.current, seed) : undefined), [seed])
  return <canvas ref={ref} className={clsx(styles.rule, className)} aria-hidden="true" />
}
