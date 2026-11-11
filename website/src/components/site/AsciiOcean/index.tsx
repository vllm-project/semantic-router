// Adapted from stackedlol/ascii-sharks (MIT); see LICENSE in this folder.
import React, { useEffect, useRef } from 'react'
import type { OceanEngine, QuietRect } from './engine'
import type { PaletteName } from './palettes'
import type { SharkSpec } from './world'
import styles from './styles.module.css'

export type { QuietRect } from './engine'
export type { PaletteName } from './palettes'
export type { SharkKind, SharkSpec } from './world'
export { quietRectsIn, textRects } from './rects'

export interface AsciiOceanProps {
  className?: string
  /** 'auto' (the default) follows the site theme from the first paint: paper in light, night in dark. */
  palette?: PaletteName | 'auto'
  /**
   * The hero copy's rects in viewport coordinates, read again as the copy moves. Whatever swims under them draws at
   * 20 % alpha at most, and the fish and sharks steer around them.
   */
  quietRects?: () => readonly QuietRect[]
  /** The sharks in the water, size relative to the biggest; read once, on mount. */
  sharks?: SharkSpec[]
  /** Multiplies the area-based fish population; read once, on mount. */
  density?: number
}

const PALETTE_CLASS: Record<PaletteName | 'auto', string> = {
  auto: styles.auto,
  night: styles.night,
  paper: styles.paper,
}

/** How long the engine waits for its web font (ms) before it starts with a fallback face and redraws later. */
const FONT_WAIT = 1500

/** Settles once the canvas's font has loaded or failed, or after FONT_WAIT. */
function fontSettled(canvas: HTMLCanvasElement): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, FONT_WAIT)
    const done = () => {
      window.clearTimeout(timer)
      resolve()
    }
    if (!document.fonts) done()
    else document.fonts.load(`600 16px ${getComputedStyle(canvas).fontFamily}`).then(done, done)
  })
}

/**
 * Fish schools and the sharks that hunt them, drawn in ASCII on a canvas that fills its box. Decorative: aria-hidden
 * and without pointer events of its own (it listens on the window). The engine loads after mount, so the server
 * renders a bare canvas on its palette's ground; it pauses off screen and in hidden tabs, draws one still frame under
 * prefers-reduced-motion, and stops and cleans up on unmount.
 */
export default function AsciiOcean({ className, palette = 'auto', quietRects, sharks, density }: AsciiOceanProps): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<OceanEngine | null>(null)
  const paletteRef = useRef(palette)
  const quietRef = useRef(quietRects)
  const roster = useRef({ sharks, density })

  useEffect(() => {
    quietRef.current = quietRects
  }, [quietRects])

  useEffect(() => {
    paletteRef.current = palette
    engineRef.current?.setPalette(palette)
  }, [palette])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    let cancelled = false
    // The font loads beside the engine's chunk, so the first frame draws every glyph once, in the right face.
    Promise.all([import('./engine'), fontSettled(canvas)]).then(([{ OceanEngine }]) => {
      if (cancelled) return
      engineRef.current = new OceanEngine(canvas, {
        ...roster.current,
        palette: paletteRef.current,
        quietRects: () => quietRef.current?.() ?? [],
      })
    }, () => undefined)
    return () => {
      cancelled = true
      engineRef.current?.destroy()
      engineRef.current = null
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={[styles.ocean, PALETTE_CLASS[palette], className].filter(Boolean).join(' ')}
    />
  )
}
