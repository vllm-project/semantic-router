import { useEffect, useRef } from 'react'
import styles from './AnimatedBackground.module.css'

type Rgb = readonly [number, number, number]

interface BlobDefinition {
  baseX: number
  baseY: number
  radius: number
  phase: number
  color: Rgb
  opacity: number
}

interface AnimatedBackgroundProps {
  speed?: 'slow' | 'normal'
}

const FRAME_INTERVAL = 1000 / 24
const STATIC_PHASE = 11.5
const BLOB_DEFINITIONS: readonly BlobDefinition[] = [
  { baseX: 0.16, baseY: 0.22, radius: 0.42, phase: 0.2, color: [214, 216, 220], opacity: 0.2 },
  { baseX: 0.78, baseY: 0.18, radius: 0.34, phase: 1.45, color: [96, 100, 108], opacity: 0.24 },
  { baseX: 0.82, baseY: 0.74, radius: 0.4, phase: 2.8, color: [227, 27, 35], opacity: 0.22 },
  { baseX: 0.34, baseY: 0.82, radius: 0.38, phase: 4.1, color: [52, 54, 59], opacity: 0.32 },
  { baseX: 0.52, baseY: 0.5, radius: 0.28, phase: 5.2, color: [156, 159, 165], opacity: 0.14 },
]

const rgba = ([red, green, blue]: Rgb, opacity: number) =>
  `rgba(${red}, ${green}, ${blue}, ${opacity})`

const drawBackground = (
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  phase: number,
) => {
  context.clearRect(0, 0, width, height)

  const backdrop = context.createRadialGradient(
    width * 0.5,
    height * 0.45,
    0,
    width * 0.5,
    height * 0.45,
    Math.max(width, height) * 0.72,
  )
  backdrop.addColorStop(0, '#111113')
  backdrop.addColorStop(1, '#030303')
  context.fillStyle = backdrop
  context.fillRect(0, 0, width, height)

  const shortestSide = Math.min(width, height)
  const travel = shortestSide * 0.18

  BLOB_DEFINITIONS.forEach((blob, index) => {
    const x = width * blob.baseX + Math.sin(phase + blob.phase) * travel
    const y = height * blob.baseY + Math.cos(phase * 0.86 + blob.phase * 1.2) * travel
    const radius =
      Math.max(170, shortestSide * blob.radius) * (1 + Math.sin(phase * 0.72 + index) * 0.08)
    const gradient = context.createRadialGradient(x, y, 0, x, y, radius)

    gradient.addColorStop(0, rgba(blob.color, blob.opacity))
    gradient.addColorStop(0.36, rgba(blob.color, blob.opacity * 0.66))
    gradient.addColorStop(0.72, rgba(blob.color, blob.opacity * 0.24))
    gradient.addColorStop(1, rgba(blob.color, 0))

    context.fillStyle = gradient
    context.beginPath()
    context.arc(x, y, radius, 0, Math.PI * 2)
    context.fill()
  })
}

const AnimatedBackground = ({ speed = 'normal' }: AnimatedBackgroundProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    const parent = canvas?.parentElement
    if (!canvas || !context || !parent) return

    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const phaseRate = speed === 'slow' ? 0.11 : 0.34
    let width = 1
    let height = 1
    let phase = 0
    let lastPaint = 0

    const paint = (nextPhase = phase) => {
      phase = nextPhase
      drawBackground(context, width, height, phase)
    }

    const resize = () => {
      const bounds = parent.getBoundingClientRect()
      width = Math.max(1, Math.round(bounds.width))
      height = Math.max(1, Math.round(bounds.height))
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.25)

      canvas.width = Math.round(width * pixelRatio)
      canvas.height = Math.round(height * pixelRatio)
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
      paint(motionQuery.matches ? STATIC_PHASE : phase)
    }

    const stopAnimation = () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
    }

    const animate = (timestamp: number) => {
      if (timestamp - lastPaint >= FRAME_INTERVAL) {
        lastPaint = timestamp
        paint(timestamp * 0.001 * phaseRate)
      }
      animationRef.current = requestAnimationFrame(animate)
    }

    const syncMotion = () => {
      stopAnimation()
      canvas.dataset.motion = motionQuery.matches ? 'static' : 'animated'

      if (motionQuery.matches) {
        paint(STATIC_PHASE)
      } else if (!document.hidden) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopAnimation()
      } else {
        syncMotion()
      }
    }

    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(parent)
    motionQuery.addEventListener('change', syncMotion)
    document.addEventListener('visibilitychange', handleVisibilityChange)
    resize()
    syncMotion()

    return () => {
      stopAnimation()
      resizeObserver.disconnect()
      motionQuery.removeEventListener('change', syncMotion)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [speed])

  return (
    <canvas
      ref={canvasRef}
      className={styles.canvas}
      data-testid="playground-motion-background"
      data-motion="animated"
      aria-hidden="true"
    />
  )
}

export default AnimatedBackground
