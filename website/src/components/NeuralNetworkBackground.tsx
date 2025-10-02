import React, { useEffect, useRef } from 'react'
import styles from './NeuralNetworkBackground.module.css'

interface Node {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  opacity: number
  pulsePhase: number
  isAI: boolean
}

interface Connection {
  nodeA: Node
  nodeB: Node
  distance: number
  opacity: number
  isActive: boolean
}

const NeuralNetworkBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const nodesRef = useRef<Node[]>([])
  const connectionsRef = useRef<Connection[]>([])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let width: number, height: number

    const resizeCanvas = () => {
      const rect = canvas.parentElement?.getBoundingClientRect()
      if (!rect) return
      width = rect.width
      height = rect.height
      canvas.width = width
      canvas.height = height
    }

    // Initialize nodes
    const initNodes = () => {
      const nodeCount = Math.min(50, Math.floor((width * height) / 15000))
      nodesRef.current = []

      for (let i = 0; i < nodeCount; i++) {
        nodesRef.current.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          radius: Math.random() * 3 + 1,
          opacity: Math.random() * 0.8 + 0.2,
          pulsePhase: Math.random() * Math.PI * 2,
          isAI: Math.random() < 0.1, // 10% chance to be an AI node
        })
      }
    }

    // Create connections between nearby nodes
    const updateConnections = () => {
      connectionsRef.current = []
      const maxDistance = 120

      for (let i = 0; i < nodesRef.current.length; i++) {
        for (let j = i + 1; j < nodesRef.current.length; j++) {
          const nodeA = nodesRef.current[i]
          const nodeB = nodesRef.current[j]
          const distance = Math.sqrt(
            Math.pow(nodeA.x - nodeB.x, 2) + Math.pow(nodeA.y - nodeB.y, 2),
          )

          if (distance < maxDistance) {
            connectionsRef.current.push({
              nodeA,
              nodeB,
              distance,
              opacity: (1 - distance / maxDistance) * 0.3,
              isActive: Math.random() < 0.1, // 10% chance to be active
            })
          }
        }
      }
    }

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, width, height)

      // Update and draw connections
      connectionsRef.current.forEach((connection) => {
        const { nodeA, nodeB, opacity, isActive } = connection

        ctx.beginPath()
        ctx.moveTo(nodeA.x, nodeB.y)
        ctx.lineTo(nodeB.x, nodeB.y)

        if (isActive) {
          // Active connections with gradient
          const gradient = ctx.createLinearGradient(nodeA.x, nodeA.y, nodeB.x, nodeB.y)
          gradient.addColorStop(0, `rgba(9, 105, 218, ${opacity * 0.8})`)
          gradient.addColorStop(0.5, `rgba(88, 166, 255, ${opacity})`)
          gradient.addColorStop(1, `rgba(130, 80, 223, ${opacity * 0.8})`)
          ctx.strokeStyle = gradient
          ctx.lineWidth = 2
        }
        else {
          ctx.strokeStyle = `rgba(9, 105, 218, ${opacity * 0.4})`
          ctx.lineWidth = 1
        }

        ctx.stroke()
      })

      // Update and draw nodes
      nodesRef.current.forEach((node) => {
        // Update position
        node.x += node.vx
        node.y += node.vy

        // Bounce off edges
        if (node.x < 0 || node.x > width) node.vx *= -1
        if (node.y < 0 || node.y > height) node.vy *= -1

        // Keep nodes in bounds
        node.x = Math.max(0, Math.min(width, node.x))
        node.y = Math.max(0, Math.min(height, node.y))

        // Update pulse
        node.pulsePhase += 0.02
        const pulse = Math.sin(node.pulsePhase) * 0.3 + 0.7

        // Draw node
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius * pulse, 0, Math.PI * 2)

        if (node.isAI) {
          // AI nodes with special glow
          const gradient = ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, node.radius * pulse * 3,
          )
          gradient.addColorStop(0, `rgba(253, 181, 22, ${node.opacity * pulse})`)
          gradient.addColorStop(0.5, `rgba(253, 181, 22, ${node.opacity * pulse * 0.5})`)
          gradient.addColorStop(1, 'rgba(253, 181, 22, 0)')
          ctx.fillStyle = gradient
          ctx.fill()

          // Core
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.radius * 0.6, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(253, 181, 22, ${node.opacity})`
          ctx.fill()
        }
        else {
          // Regular nodes
          const gradient = ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, node.radius * pulse * 2,
          )
          gradient.addColorStop(0, `rgba(88, 166, 255, ${node.opacity * pulse})`)
          gradient.addColorStop(1, 'rgba(88, 166, 255, 0)')
          ctx.fillStyle = gradient
          ctx.fill()

          // Core
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.radius * 0.5, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(88, 166, 255, ${node.opacity})`
          ctx.fill()
        }
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    // Initialize
    resizeCanvas()
    initNodes()
    updateConnections()
    animate()

    // Handle resize
    const handleResize = () => {
      resizeCanvas()
      initNodes()
      updateConnections()
    }

    window.addEventListener('resize', handleResize)

    // Update connections periodically
    const connectionInterval = setInterval(updateConnections, 2000)

    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
      }
      window.removeEventListener('resize', handleResize)
      clearInterval(connectionInterval)
    }
  }, [])

  return (
    <div className={styles.neuralNetworkContainer}>
      <canvas
        ref={canvasRef}
        className={styles.neuralNetworkCanvas}
      />
    </div>
  )
}

export default NeuralNetworkBackground
