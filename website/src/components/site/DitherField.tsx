import React, { useEffect, useRef } from 'react'

interface DitherFieldProps {
  className?: string
}

const bayerMatrix = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
]

function getThreshold(x: number, y: number): number {
  return bayerMatrix[y % 4][x % 4] / 16 - 0.5
}

export default function DitherField({ className }: DitherFieldProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<number>()
  const timeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return undefined
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return undefined
    }

    let width = 0
    let height = 0

    const resize = () => {
      const parent = canvas.parentElement
      width = parent?.clientWidth ?? window.innerWidth
      height = parent?.clientHeight ?? window.innerHeight
      canvas.width = width
      canvas.height = height
    }

    const draw = () => {
      context.fillStyle = '#111111'
      context.fillRect(0, 0, width, height)

      const cell = 6
      const cols = Math.ceil(width / cell)
      const rows = Math.ceil(height / cell)
      const waveCenterY = rows / 2
      const waveAmplitude = rows / 4.2
      const frequency = 0.045
      const speed = 0.018

      for (let y = 0; y < rows; y += 1) {
        for (let x = 0; x < cols; x += 1) {
          const waveA = Math.sin(x * frequency + timeRef.current) * waveAmplitude
          const waveB = Math.cos(x * frequency * 0.5 - timeRef.current) * (waveAmplitude * 0.55)
          const combinedWave = waveA + waveB
          const distance = Math.abs(y - (waveCenterY + combinedWave))
          let intensity = Math.max(0, 1 - distance / 15)
          intensity += (Math.random() - 0.5) * 0.08

          if (intensity + getThreshold(x, y) > 0.5) {
            context.fillStyle = '#ecece7'
            context.fillRect(x * cell, y * cell, cell - 1, cell - 1)
          }
        }
      }

      timeRef.current += speed
      frameRef.current = requestAnimationFrame(draw)
    }

    window.addEventListener('resize', resize)
    resize()
    draw()

    return () => {
      window.removeEventListener('resize', resize)
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
      }
    }
  }, [])

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />
}
