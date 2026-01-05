import React, { useEffect, useRef } from 'react'
import Translate from '@docusaurus/Translate'
import styles from './AIChipAnimation.module.css'

const AIChipAnimation: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return

    // Add pulsing animation to circuit paths
    const paths = svg.querySelectorAll<SVGPathElement>('.circuit-path')
    paths.forEach((path, index) => {
      path.style.animationDelay = `${index * 0.2}s`
    })

    // Add data flow animation
    const dataPoints = svg.querySelectorAll<SVGCircleElement>('.data-point')
    dataPoints.forEach((point, index) => {
      point.style.animationDelay = `${index * 0.3}s`
    })
  }, [])

  return (
    <div className={styles.chipContainer}>
      <svg
        ref={svgRef}
        className={styles.chipSvg}
        viewBox="0 0 400 300"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Chip base */}
        <rect
          x="50"
          y="50"
          width="300"
          height="200"
          rx="20"
          fill="url(#chipGradient)"
          stroke="url(#chipBorder)"
          strokeWidth="2"
          className={styles.chipBase}
        />

        {/* Circuit patterns */}
        <g className={styles.circuitGroup}>
          {/* Horizontal circuits */}
          <path
            d="M 70 100 L 130 100 L 140 110 L 180 110 L 190 100 L 250 100 L 260 110 L 300 110 L 310 100 L 330 100"
            stroke="url(#circuitGradient)"
            strokeWidth="3"
            fill="none"
            className="circuit-path"
          />
          <path
            d="M 70 140 L 110 140 L 120 130 L 160 130 L 170 140 L 210 140 L 220 130 L 260 130 L 270 140 L 330 140"
            stroke="url(#circuitGradient)"
            strokeWidth="3"
            fill="none"
            className="circuit-path"
          />
          <path
            d="M 70 180 L 120 180 L 130 170 L 170 170 L 180 180 L 220 180 L 230 170 L 270 170 L 280 180 L 330 180"
            stroke="url(#circuitGradient)"
            strokeWidth="3"
            fill="none"
            className="circuit-path"
          />

          {/* Vertical circuits */}
          <path
            d="M 120 70 L 120 110 L 130 120 L 130 160 L 120 170 L 120 210 L 130 220 L 130 230"
            stroke="url(#circuitGradient)"
            strokeWidth="3"
            fill="none"
            className="circuit-path"
          />
          <path
            d="M 200 70 L 200 90 L 210 100 L 210 140 L 200 150 L 200 190 L 210 200 L 210 230"
            stroke="url(#circuitGradient)"
            strokeWidth="3"
            fill="none"
            className="circuit-path"
          />
          <path
            d="M 280 70 L 280 100 L 270 110 L 270 150 L 280 160 L 280 200 L 270 210 L 270 230"
            stroke="url(#circuitGradient)"
            strokeWidth="3"
            fill="none"
            className="circuit-path"
          />
        </g>

        {/* Processing cores */}
        <g className={styles.coresGroup}>
          <circle cx="150" cy="120" r="15" fill="url(#coreGradient)" className={styles.processingCore} />
          <circle cx="250" cy="120" r="15" fill="url(#coreGradient)" className={styles.processingCore} />
          <circle cx="150" cy="180" r="15" fill="url(#coreGradient)" className={styles.processingCore} />
          <circle cx="250" cy="180" r="15" fill="url(#coreGradient)" className={styles.processingCore} />

          {/* Core labels */}
          <text x="150" y="125" textAnchor="middle" className={styles.coreLabel}>AI</text>
          <text x="250" y="125" textAnchor="middle" className={styles.coreLabel}>ML</text>
          <text x="150" y="185" textAnchor="middle" className={styles.coreLabel}>NN</text>
          <text x="250" y="185" textAnchor="middle" className={styles.coreLabel}>LLM</text>
        </g>

        {/* Data flow points */}
        <g className={styles.dataFlowGroup}>
          <circle cx="90" cy="100" r="3" fill="#FDB516" className="data-point" />
          <circle cx="160" cy="110" r="3" fill="#FDB516" className="data-point" />
          <circle cx="230" cy="100" r="3" fill="#FDB516" className="data-point" />
          <circle cx="310" cy="110" r="3" fill="#FDB516" className="data-point" />

          <circle cx="100" cy="140" r="3" fill="#30A2FF" className="data-point" />
          <circle cx="180" cy="130" r="3" fill="#30A2FF" className="data-point" />
          <circle cx="250" cy="140" r="3" fill="#30A2FF" className="data-point" />
          <circle cx="300" cy="130" r="3" fill="#30A2FF" className="data-point" />
        </g>

        {/* Chip pins */}
        <g className={styles.pinsGroup}>
          {/* Left pins */}
          <rect x="30" y="80" width="20" height="4" fill="#8CC5FF" />
          <rect x="30" y="100" width="20" height="4" fill="#8CC5FF" />
          <rect x="30" y="120" width="20" height="4" fill="#8CC5FF" />
          <rect x="30" y="140" width="20" height="4" fill="#8CC5FF" />
          <rect x="30" y="160" width="20" height="4" fill="#8CC5FF" />
          <rect x="30" y="180" width="20" height="4" fill="#8CC5FF" />
          <rect x="30" y="200" width="20" height="4" fill="#8CC5FF" />
          <rect x="30" y="220" width="20" height="4" fill="#8CC5FF" />

          {/* Right pins */}
          <rect x="350" y="80" width="20" height="4" fill="#8CC5FF" />
          <rect x="350" y="100" width="20" height="4" fill="#8CC5FF" />
          <rect x="350" y="120" width="20" height="4" fill="#8CC5FF" />
          <rect x="350" y="140" width="20" height="4" fill="#8CC5FF" />
          <rect x="350" y="160" width="20" height="4" fill="#8CC5FF" />
          <rect x="350" y="180" width="20" height="4" fill="#8CC5FF" />
          <rect x="350" y="200" width="20" height="4" fill="#8CC5FF" />
          <rect x="350" y="220" width="20" height="4" fill="#8CC5FF" />
        </g>

        {/* Gradients and definitions */}
        <defs>
          <linearGradient id="chipGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#1a1a2e" />
            <stop offset="50%" stopColor="#16213e" />
            <stop offset="100%" stopColor="#0f3460" />
          </linearGradient>

          <linearGradient id="chipBorder" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#58A6FF" />
            <stop offset="50%" stopColor="#30A2FF" />
            <stop offset="100%" stopColor="#0969DA" />
          </linearGradient>

          <linearGradient id="circuitGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#58A6FF" />
            <stop offset="50%" stopColor="#FDB516" />
            <stop offset="100%" stopColor="#58A6FF" />
          </linearGradient>

          <radialGradient id="coreGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#FDB516" />
            <stop offset="70%" stopColor="#30A2FF" />
            <stop offset="100%" stopColor="#0969DA" />
          </radialGradient>
        </defs>
      </svg>

      <div className={styles.chipLabel}>
        <span className={styles.chipTitle}>
          <Translate id="aiChip.title">Neural Processing Unit</Translate>
        </span>
        <span className={styles.chipSubtitle}>
          <Translate id="aiChip.subtitle">Embedding • Classify • Similarity</Translate>
        </span>
      </div>
    </div>
  )
}

export default AIChipAnimation
