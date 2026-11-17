import * as React from 'react'
import { cn } from '../lib/cn'

interface Palette {
  ink: string
  blue: string
  gold: string
  base: number
}

function readPalette(): Palette {
  const style = getComputedStyle(document.documentElement)
  const dark = document.documentElement.classList.contains('dark')
  return {
    ink: style.getPropertyValue('--foreground').trim() || (dark ? 'white' : 'black'),
    blue: style.getPropertyValue('--brand').trim() || 'blue',
    gold: style.getPropertyValue('--brand-gold').trim() || 'orange',
    base: dark ? 0.055 : 0.032,
  }
}

const smoothstep = (edge0: number, edge1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

/**
 * Animated mosaic of square tiles: tiles assemble on load, a slow diagonal
 * wave passes through them, a few brand-colored tiles glint and the pointer
 * lights its neighbourhood. The center stays calm so content above it reads.
 */
export function MosaicBackground({
  className,
  cell = 22,
  gap = 3,
}: {
  className?: string
  cell?: number
  gap?: number
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)

  React.useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const pointer = { x: -1e4, y: -1e4, strength: 0, target: 0 }
    let palette = readPalette()
    let width = 0
    let height = 0
    let cols = 0
    let rows = 0
    let seeds = new Float32Array(0)
    let frame = 0
    let last = 0
    const start = performance.now()

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      width = rect.width
      height = rect.height
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      context.setTransform(dpr, 0, 0, dpr, 0, 0)
      cols = Math.ceil(width / cell) + 1
      rows = Math.ceil(height / cell) + 1
      seeds = new Float32Array(cols * rows)
      for (let index = 0; index < seeds.length; index++) {
        const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453
        seeds[index] = value - Math.floor(value)
      }
    }

    const draw = (now: number) => {
      const t = reduceMotion ? 6 : (now - start) / 1000
      pointer.strength += (pointer.target - pointer.strength) * 0.08
      context.clearRect(0, 0, width, height)
      const cx = width / 2
      const cy = height / 2
      const radius = Math.hypot(cx, cy)
      const size = cell - gap
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const seed = seeds[row * cols + col]
          const x = col * cell
          const y = row * cell
          const reveal = reduceMotion
            ? 1
            : smoothstep(0, 1, (t * 1.25 - seed * 0.9 - (col + row) * 0.004) * 2.6)
          if (reveal <= 0) continue
          const distance = Math.hypot(x + size / 2 - cx, y + size / 2 - cy) / radius
          const calm =
            smoothstep(0.2, 0.7, distance) * (1 - smoothstep(0.85, 1.2, distance) * 0.6)
          if (calm <= 0.01) continue
          const wave = (Math.sin(col * 0.21 + row * 0.13 - t * 0.85 + seed * 1.7) + 1) / 2
          const glint = Math.pow(Math.max(0, Math.sin(t * (0.35 + seed * 0.9) + seed * 61)), 18)
          const near =
            pointer.strength > 0.01
              ? Math.max(0, 1 - Math.hypot(x - pointer.x, y - pointer.y) / 170) * pointer.strength
              : 0
          let alpha = palette.base * (0.45 + 0.85 * wave) + near * 0.16
          let color = palette.ink
          if (seed > 0.965) {
            color = seed > 0.988 ? palette.gold : palette.blue
            alpha = alpha * 0.8 + glint * 0.55 + near * 0.25
          } else {
            alpha += glint * 0.035
          }
          context.globalAlpha = Math.min(0.6, alpha * calm * reveal)
          context.fillStyle = color
          context.fillRect(x, y, size, size)
        }
      }
      context.globalAlpha = 1
    }

    const loop = (now: number) => {
      frame = requestAnimationFrame(loop)
      if (document.hidden || now - last < 33) return
      last = now
      draw(now)
    }

    const onPointer = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect()
      pointer.x = event.clientX - rect.left
      pointer.y = event.clientY - rect.top
      pointer.target = 1
    }
    const onLeave = () => {
      pointer.target = 0
    }
    const observer = new MutationObserver(() => {
      palette = readPalette()
      if (reduceMotion) draw(performance.now())
    })
    const resizeObserver = new ResizeObserver(() => {
      resize()
      if (reduceMotion) draw(performance.now())
    })

    resize()
    resizeObserver.observe(canvas)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-accent'],
    })
    if (reduceMotion) {
      draw(performance.now())
    } else {
      window.addEventListener('pointermove', onPointer)
      document.addEventListener('pointerleave', onLeave)
      frame = requestAnimationFrame(loop)
    }
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      resizeObserver.disconnect()
      window.removeEventListener('pointermove', onPointer)
      document.removeEventListener('pointerleave', onLeave)
    }
  }, [cell, gap])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className={cn('pointer-events-none absolute inset-0 size-full', className)}
    />
  )
}
