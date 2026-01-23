import React, { useEffect, useRef } from 'react'
import styles from './AnimatedBackground.module.css'

interface Blob {
  x: number
  y: number
  baseX: number
  baseY: number
  radius: number
  color: { h: number; s: number; l: number }
  offsetX: number
  offsetY: number
  speedX: number
  speedY: number
}

interface AnimatedBackgroundProps {
  speed?: 'slow' | 'normal'
}

const AnimatedBackground: React.FC<AnimatedBackgroundProps> = ({ speed = 'normal' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    const blobs: Blob[] = []
    const colors = [
      { h: 84, s: 70, l: 50 },   // NVIDIA green
      { h: 90, s: 65, l: 55 },   // Yellow-green
      { h: 78, s: 75, l: 45 },   // Deep green
      { h: 160, s: 60, l: 50 },  // Cyan accent
      { h: 200, s: 55, l: 55 },  // Blue accent
    ]

    const speedMultiplier = speed === 'slow' ? 0.3 : 1

    for (let i = 0; i < 5; i++) {
      const angle = (i / 5) * Math.PI * 2
      const distance = Math.min(canvas.width, canvas.height) * 0.35

      blobs.push({
        baseX: canvas.width / 2 + Math.cos(angle) * distance,
        baseY: canvas.height / 2 + Math.sin(angle) * distance,
        x: 0,
        y: 0,
        radius: 200 + Math.random() * 150,
        color: colors[i],
        offsetX: 0,
        offsetY: 0,
        speedX: (0.006 + Math.random() * 0.003) * speedMultiplier,
        speedY: (0.006 + Math.random() * 0.003) * speedMultiplier,
      })
    }

    let time = 0

    const animate = () => {
      time += 1

      const bgGradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) / 2
      )
      bgGradient.addColorStop(0, '#0a0a0a')
      bgGradient.addColorStop(1, '#000000')
      ctx.fillStyle = bgGradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      blobs.forEach((blob, index) => {
        blob.offsetX = Math.sin(time * blob.speedX + index) * 250
        blob.offsetY = Math.cos(time * blob.speedY + index * 1.3) * 250

        blob.x = blob.baseX + blob.offsetX
        blob.y = blob.baseY + blob.offsetY

        const pulseRadius = blob.radius + Math.sin(time * 0.001 + index * 0.7) * 30

        const gradient = ctx.createRadialGradient(
          blob.x, blob.y, 0,
          blob.x, blob.y, pulseRadius
        )

        const { h, s, l } = blob.color

        gradient.addColorStop(0, `hsla(${h}, ${s}%, ${l + 10}%, 0.8)`)
        gradient.addColorStop(0.3, `hsla(${h}, ${s}%, ${l}%, 0.6)`)
        gradient.addColorStop(0.6, `hsla(${h}, ${s - 10}%, ${l - 10}%, 0.3)`)
        gradient.addColorStop(1, `hsla(${h}, ${s - 20}%, ${l - 20}%, 0)`)

        ctx.save()
        ctx.filter = 'blur(60px)'
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(blob.x, blob.y, pulseRadius, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  return <canvas ref={canvasRef} className={styles.canvas} />
}

export default AnimatedBackground

