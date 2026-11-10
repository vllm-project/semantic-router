import React, { useEffect, useRef } from 'react'
import { rng } from './pixels'

/* Duo's dusk stars (mosaic.js): sparse pixel squares that appear as the gradient darkens, brighter towards the ink, a
 * few of them ember and dawn blue. The seed is fixed, so every visit gets the same sky; drawn once per width. */
function mountStars(cv: HTMLCanvasElement): () => void {
  const ctx = cv.getContext('2d')
  if (!ctx) return () => undefined
  let drawnWidth = 0
  const draw = () => {
    const w = cv.clientWidth
    const h = cv.clientHeight
    if (!w || !h || w === drawnWidth) return
    drawnWidth = w
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    cv.width = Math.round(w * dpr)
    cv.height = Math.round(h * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const r = rng(4241)
    const n = Math.round(w * h / 2600)
    for (let i = 0; i < n; i++) {
      const x = r() * w
      const y = h * (0.45 + 0.55 * Math.sqrt(r()))
      const depth = (y / h - 0.45) / 0.55
      const kind = r()
      if (r() > depth) continue
      const size = kind < 0.08 ? 3 : 2
      const a = (0.18 + 0.6 * depth * r()).toFixed(3)
      ctx.fillStyle = kind < 0.035
        ? `rgba(239, 86, 48, ${a})`
        : kind < 0.07 ? `rgba(147, 179, 228, ${a})` : `rgba(243, 239, 230, ${a})`
      ctx.fillRect(Math.round(x), Math.round(y), size, size)
    }
  }
  draw()
  const ro = new ResizeObserver(draw)
  ro.observe(cv)
  return () => ro.disconnect()
}

export default function Stars({ className }: { className?: string }): JSX.Element {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => (ref.current ? mountStars(ref.current) : undefined), [])
  return <canvas ref={ref} className={className} aria-hidden="true" />
}
