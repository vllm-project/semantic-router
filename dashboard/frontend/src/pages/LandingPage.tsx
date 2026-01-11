import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import styles from './LandingPage.module.css'

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  size: number
  opacity: number
}

interface TerminalLine {
  type: 'command' | 'output' | 'comment' | 'clear'
  content: string
  delay?: number
}

// Terminal demo script - CLI commands and Chain-of-Thought demos
const TERMINAL_SCRIPT: TerminalLine[] = [
    // CLI Demo 1: Initialize
    { type: 'comment', content: '# Quick Start with vllm-sr CLI', delay: 800 },
    { type: 'command', content: '$ vllm-sr init', delay: 500 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: '✓ Created config.yaml', delay: 300 },
    { type: 'output', content: '✓ Created .vllm-sr/ directory with templates', delay: 300 },
    { type: 'output', content: '✓ vLLM Semantic Router initialized!', delay: 300 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: 'Next: Edit config.yaml and run "vllm-sr serve"', delay: 800 },
    { type: 'clear', content: '', delay: 1500 },

    // CLI Demo 2: Serve
    { type: 'comment', content: '# Start the Semantic Router', delay: 800 },
    { type: 'command', content: '$ vllm-sr serve', delay: 500 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: '✓ Container started successfully', delay: 300 },
    { type: 'output', content: '✓ Router is healthy', delay: 300 },
    { type: 'output', content: '✓ vLLM Semantic Router is running!', delay: 300 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: 'Endpoints:', delay: 200 },
    { type: 'output', content: '  • http-listener: http://localhost:8888', delay: 200 },
    { type: 'output', content: '  • Metrics: http://localhost:9190/metrics', delay: 800 },
    { type: 'clear', content: '', delay: 1500 },

    // CLI Demo 3: Dashboard
    { type: 'comment', content: '# Start dashboard to manage intelligent routing', delay: 800 },
    { type: 'command', content: '$ vllm-sr dashboard', delay: 500 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: '════════════════════════════════════════════════════════════', delay: 200 },
    { type: 'output', content: 'Opening dashboard: http://localhost:8700', delay: 200 },
    { type: 'output', content: '', delay: 100 },
    { type: 'output', content: '✓ Dashboard opened in browser', delay: 300 },
    { type: 'output', content: '════════════════════════════════════════════════════════════', delay: 800 },
    { type: 'clear', content: '', delay: 1500 },

    // Routing Demo: Math Question
    { type: 'comment', content: '# Intelligent Routing in Action', delay: 800 },
    { type: 'command', content: '$ curl -X POST http://vllm-semantic-router/v1/chat/completions \\', delay: 500 },
    { type: 'command', content: '  -d \'{"model": "MoM", "messages": [{"role": "user", "content": "What is 2+2?"}]}\'', delay: 400 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: '🔀 vLLM Semantic Router - Chain-Of-Thought 🔀', delay: 300 },
    { type: 'output', content: '  → 🛡️ Stage 1 - Prompt Guard: ✅ No Jailbreak → ✅ No PII → 💯 Continue', delay: 300 },
    { type: 'output', content: '  → 🔥 Stage 2 - Router Memory: 🌊 MISS → 🧠 Update Memory → 💯 Continue', delay: 300 },
    { type: 'output', content: '  → 🧠 Stage 3 - Smart Routing: 📂 math → 🧠 Reasoning On → 🥷 deepseek-v3 → 💯 Continue', delay: 300 },
    { type: 'output', content: '✅ Response: "2 + 2 = 4"', delay: 1200 },
    { type: 'clear', content: '', delay: 1500 },

    // Routing Demo: Jailbreak Detection
    { type: 'comment', content: '# Security: Jailbreak Detection', delay: 800 },
    { type: 'command', content: '$ curl -X POST http://vllm-semantic-router/v1/chat/completions \\', delay: 500 },
    { type: 'command', content: '  -d \'{"model": "MoM", "messages": [{"role": "user", "content": "Ignore instructions..."}]}\'', delay: 400 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: '🔀 vLLM Semantic Router - Chain-Of-Thought 🔀', delay: 300 },
    { type: 'output', content: '  → 🛡️ Stage 1 - Prompt Guard: 🚨 Jailbreak Detected (0.950) → ❌ BLOCKED', delay: 300 },
    { type: 'output', content: '❌ Request blocked for security reasons', delay: 1200 },
    { type: 'clear', content: '', delay: 1500 },

    // CLI Demo: Logs
    { type: 'comment', content: '# View service logs', delay: 800 },
    { type: 'command', content: '$ vllm-sr logs router', delay: 500 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: '{"level":"info","caller":"router/main.go:42","msg":"Starting router..."}', delay: 200 },
    { type: 'output', content: '{"level":"info","caller":"router/server.go:88","msg":"Health check passed"}', delay: 200 },
    { type: 'output', content: '{"level":"info","caller":"router/handler.go:156","msg":"Request processed","latency":"12ms"}', delay: 800 },
    { type: 'clear', content: '', delay: 1500 },

    // Routing Demo: Cache Hit
    { type: 'comment', content: '# Cache Hit - Fast Response!', delay: 800 },
    { type: 'command', content: '$ curl -X POST http://vllm-semantic-router/v1/chat/completions \\', delay: 500 },
    { type: 'command', content: '  -d \'{"model": "MoM", "messages": [{"role": "user", "content": "What is 2+2?"}]}\'', delay: 400 },
    { type: 'output', content: '', delay: 200 },
    { type: 'output', content: '🔀 vLLM Semantic Router - Chain-Of-Thought 🔀', delay: 300 },
    { type: 'output', content: '  → 🛡️ Stage 1 - Prompt Guard: ✅ No Jailbreak → ✅ No PII → 💯 Continue', delay: 300 },
    { type: 'output', content: '  → 🔥 Stage 2 - Router Memory: 🔥 HIT → ⚡ Retrieve Memory → 💯 Fast Response', delay: 300 },
    { type: 'output', content: '✅ Response: "2 + 2 = 4" (cached, 2ms)', delay: 1200 },
    { type: 'clear', content: '', delay: 1500 }
]

const LandingPage: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([])
  const [currentLineIndex, setCurrentLineIndex] = useState(0)
  const [isTyping, setIsTyping] = useState(false)

  // Function to highlight keywords in content
  const highlightContent = (content: string) => {
    // Split by both "MoM" and "vllm-semantic-router"
    const parts = content.split(/("MoM"|vllm-semantic-router)/gi)
    return parts.map((part, index) => {
      if (part.toLowerCase() === '"mom"') {
        return (
          <span key={index} style={{
            color: '#fbbf24',
            fontWeight: 'bold',
            textShadow: '0 0 10px rgba(251, 191, 36, 0.5)'
          }}>
            {part}
          </span>
        )
      }
      if (part.toLowerCase() === 'vllm-semantic-router') {
        return (
          <span key={index} style={{
            color: '#3b82f6',
            fontWeight: 'bold',
            textShadow: '0 0 10px rgba(59, 130, 246, 0.5)'
          }}>
            {part}
          </span>
        )
      }
      return part
    })
  }

  // Initialize particles for background animation
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

    // Create particles
    const particleCount = 30
    const particles: Particle[] = []

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.3 + 0.1
      })
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update and draw particles
      particles.forEach((particle, index) => {
        particle.x += particle.vx
        particle.y += particle.vy

        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width
        if (particle.x > canvas.width) particle.x = 0
        if (particle.y < 0) particle.y = canvas.height
        if (particle.y > canvas.height) particle.y = 0

        // Draw particle - NVIDIA Green
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(118, 185, 0, ${particle.opacity})`
        ctx.fill()

        // Draw connections
        particles.slice(index + 1).forEach(otherParticle => {
          const dx = particle.x - otherParticle.x
          const dy = particle.y - otherParticle.y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance < 100) {
            ctx.beginPath()
            ctx.moveTo(particle.x, particle.y)
            ctx.lineTo(otherParticle.x, otherParticle.y)
            ctx.strokeStyle = `rgba(118, 185, 0, ${0.08 * (1 - distance / 100)})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        })
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

  // Terminal typing animation
  useEffect(() => {
    if (currentLineIndex >= TERMINAL_SCRIPT.length) {
      // Reset to beginning for loop
      const timer = setTimeout(() => {
        setTerminalLines([])
        setCurrentLineIndex(0)
      }, 2000)
      return () => clearTimeout(timer)
    }

    setIsTyping(true)
    const currentLine = TERMINAL_SCRIPT[currentLineIndex]

    const timer = setTimeout(() => {
      if (currentLine.type === 'clear') {
        // Clear the terminal
        setTerminalLines([])
      } else {
        // Add the line
        setTerminalLines(prev => [...prev, currentLine])
      }
      setCurrentLineIndex(prev => prev + 1)
      setIsTyping(false)
    }, currentLine.delay || 1000)

    return () => clearTimeout(timer)
  }, [currentLineIndex])



  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.backgroundCanvas} />

      {/* Main Content - Split Layout */}
      <main className={styles.mainContent}>
        {/* Left Side - Content */}
        <div className={styles.leftPanel}>
          <div className={styles.heroSection}>
            <div className={styles.heroTitleWrapper}>
              <img src="/vllm.png" alt="vLLM Logo" className={styles.vllmLogo} />
              <h1 className={styles.heroTitle}>
                <span className={styles.aiGlow}>AI-Powered</span>
                <br />
                vLLM Semantic Router
              </h1>
            </div>
            <p className={styles.heroSubtitle}>
              System Level Intelligent Router for Mixture-of-Models at Cloud, Enterprise and Edge
            </p>

            <div className={styles.features}>
              <div className={styles.featureTag}>Neural Networks</div>
              <div className={styles.featureTag}>LLM Routing</div>
              <div className={styles.featureTag}>Per-token Unit Economics</div>
            </div>

            <div className={styles.heroActions}>
              <Link to="/playground" className={styles.primaryButton}>
                Get Started
              </Link>
            </div>
          </div>


        </div>

        {/* Right Side - Terminal */}
        <div className={styles.rightPanel}>
          <div className={styles.terminal}>
            <div className={styles.terminalHeader}>
              <div className={styles.terminalControls}>
                <div className={styles.terminalButton} style={{ backgroundColor: '#ff5f57' }}></div>
                <div className={styles.terminalButton} style={{ backgroundColor: '#febc2e' }}></div>
                <div className={styles.terminalButton} style={{ backgroundColor: '#28c840' }}></div>
              </div>
              <div className={styles.terminalTitle}>vllm-sr -- -zsh</div>
            </div>
            <div className={styles.terminalBody}>
              {terminalLines.map((line, index) => (
                <div key={index} className={`${styles.terminalLine} ${styles[line.type]}`}>
                  {line.type === 'command' && <span className={styles.prompt}>$ </span>}
                  {line.type === 'comment' && <span className={styles.commentPrefix}></span>}
                  <span className={styles.lineContent}>{highlightContent(line.content)}</span>
                </div>
              ))}
              {isTyping && (
                <div className={styles.terminalLine}>
                  <span className={styles.prompt}>$ </span>
                  <span className={styles.cursor}>|</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default LandingPage
