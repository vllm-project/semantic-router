import React, { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import styles from './LandingPage.module.css'

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

const LandingPage: React.FC = () => {
  const navigate = useNavigate()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // macOS Big Sur style fluid animation
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

    // Create organic blobs with NVIDIA green palette
    const blobs: Blob[] = []
    const colors = [
      { h: 84, s: 70, l: 50 },   // NVIDIA green
      { h: 90, s: 65, l: 55 },   // Yellow-green
      { h: 78, s: 75, l: 45 },   // Deep green
      { h: 160, s: 60, l: 50 },  // Cyan accent
      { h: 200, s: 55, l: 55 },  // Blue accent
    ]

    // Create 5 large organic blobs - more spread out
    for (let i = 0; i < 5; i++) {
      const angle = (i / 5) * Math.PI * 2
      const distance = Math.min(canvas.width, canvas.height) * 0.35 // More spread out

      blobs.push({
        baseX: canvas.width / 2 + Math.cos(angle) * distance,
        baseY: canvas.height / 2 + Math.sin(angle) * distance,
        x: 0,
        y: 0,
        radius: 200 + Math.random() * 150,
        color: colors[i],
        offsetX: 0,
        offsetY: 0,
        speedX: 0.006 + Math.random() * 0.003,
        speedY: 0.006 + Math.random() * 0.003,
      })
    }

    let time = 0

    const animate = () => {
      time += 1

      // Dark gradient background
      const bgGradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) / 2
      )
      bgGradient.addColorStop(0, '#0a0a0a')
      bgGradient.addColorStop(1, '#000000')
      ctx.fillStyle = bgGradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Update and draw blobs
      blobs.forEach((blob, index) => {
        // Organic movement using Perlin-like noise simulation - larger range
        blob.offsetX = Math.sin(time * blob.speedX + index) * 250
        blob.offsetY = Math.cos(time * blob.speedY + index * 1.3) * 250

        blob.x = blob.baseX + blob.offsetX
        blob.y = blob.baseY + blob.offsetY

        // Pulsating radius
        const pulseRadius = blob.radius + Math.sin(time * 0.001 + index * 0.7) * 30

        // Create multi-stop gradient for depth
        const gradient = ctx.createRadialGradient(
          blob.x, blob.y, 0,
          blob.x, blob.y, pulseRadius
        )

        const { h, s, l } = blob.color

        gradient.addColorStop(0, `hsla(${h}, ${s}%, ${l + 10}%, 0.8)`)
        gradient.addColorStop(0.3, `hsla(${h}, ${s}%, ${l}%, 0.6)`)
        gradient.addColorStop(0.6, `hsla(${h}, ${s - 10}%, ${l - 10}%, 0.3)`)
        gradient.addColorStop(1, `hsla(${h}, ${s - 20}%, ${l - 20}%, 0)`)

        // Draw blob with heavy blur for soft edges
        ctx.save()
        ctx.filter = 'blur(60px)'
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(blob.x, blob.y, pulseRadius, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()
      })

      // Add subtle noise overlay for texture
      ctx.globalAlpha = 0.03
      ctx.fillStyle = `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 0.5)`
      for (let i = 0; i < 50; i++) {
        const x = Math.random() * canvas.width
        const y = Math.random() * canvas.height
        ctx.fillRect(x, y, 1, 1)
      }
      ctx.globalAlpha = 1

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

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.backgroundCanvas} />

      {/* Main Content - Centered */}
      <main className={styles.mainContent}>
        <div className={styles.heroSection}>
          <h1 className={styles.title}>
            <img src="/vllm.png" alt="vLLM Logo" className={styles.logoInline} />
            LLM Semantic Router
          </h1>

          <p className={styles.subtitle}>
            System Level Intelligence for{' '}
            <span className={styles.highlight}>Mixture-of-Models</span>
          </p>
          <p className={styles.deployTargets}>
            Cloud · Data Center · Edge
          </p>

          <button
            className={styles.launchButton}
            onClick={() => navigate('/playground')}
          >
            <span className={styles.launchText}>Launch</span>
            <div className={styles.launchGlow}></div>
          </button>
        </div>
      </main>
    </div>
  )
}

export default LandingPage
