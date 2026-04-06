import React from 'react'
import styles from './DashboardPage.module.css'
import {
  SIGNAL_COLORS,
  type SignalStats,
} from './dashboardPageSupport'

interface DashboardMiniFlowDiagramProps {
  signals: SignalStats
  decisions: number
  models: number
  plugins: number
}

const DashboardMiniFlowDiagram: React.FC<DashboardMiniFlowDiagramProps> = React.memo(
  ({ signals, decisions, models, plugins }) => {
    const signalTypes = Object.entries(signals.byType).sort((a, b) => b[1] - a[1])
    const visibleSignals = signalTypes.slice(0, 7)
    const hiddenCount = signalTypes.length - visibleSignals.length
    const rowH = 34
    const sH = Math.max(
      visibleSignals.length * rowH + (hiddenCount > 0 ? 28 : 0) + 30,
      180,
    )
    const height = Math.max(sH, 220)

    const colSignal = 90
    const colDecision = 310
    const colModel = 530
    const midY = height / 2

    return (
      <svg
        viewBox={`0 0 620 ${height}`}
        className={styles.flowSvg}
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-auto"
          >
            <path
              d="M 0 0 L 10 5 L 0 10 z"
              fill="var(--color-text-muted)"
            />
          </marker>
        </defs>

        {visibleSignals.map(([type, count], index) => {
          const y = 16 + index * rowH
          const color = SIGNAL_COLORS[type] || '#999'
          const endY = y + 14
          const cx1 = colSignal + 92
          const cx2 = colDecision - 90
          return (
            <g key={type} className={styles.flowNode}>
              <rect
                x={colSignal - 55}
                y={y}
                width={110}
                height={26}
                rx={6}
                fill={`${color}18`}
                stroke={color}
                strokeWidth={1}
              />
              <text
                x={colSignal}
                y={y + 17}
                textAnchor="middle"
                fill={color}
                fontSize={10.5}
                fontFamily="var(--font-mono)"
              >
                {type} ({count})
              </text>
              <path
                d={`M ${colSignal + 55} ${endY} C ${cx1} ${endY}, ${cx2} ${midY}, ${
                  colDecision - 52
                } ${midY}`}
                fill="none"
                stroke="var(--color-border-hover)"
                strokeWidth={1}
                opacity={0.35}
                markerEnd="url(#arrow)"
              />
            </g>
          )
        })}

        {hiddenCount > 0 && (
          <text
            x={colSignal}
            y={16 + visibleSignals.length * rowH + 14}
            textAnchor="middle"
            fill="var(--color-text-muted)"
            fontSize={10}
            fontStyle="italic"
          >
            +{hiddenCount} more
          </text>
        )}

        <rect
          x={colDecision - 52}
          y={midY - 30}
          width={104}
          height={60}
          rx={10}
          fill="var(--color-primary)"
          fillOpacity={0.12}
          stroke="var(--color-primary)"
          strokeWidth={1.5}
        />
        <text
          x={colDecision}
          y={midY - 6}
          textAnchor="middle"
          fill="var(--color-primary)"
          fontSize={11}
          fontWeight="bold"
        >
          Decision
        </text>
        <text
          x={colDecision}
          y={midY + 12}
          textAnchor="middle"
          fill="var(--color-primary)"
          fontSize={10.5}
          opacity={0.85}
        >
          {decisions} layers
        </text>

        <line
          x1={colDecision + 54}
          y1={midY}
          x2={colModel - 54}
          y2={midY}
          stroke="var(--color-border-hover)"
          strokeWidth={1.5}
          markerEnd="url(#arrow)"
        />

        <rect
          x={colModel - 52}
          y={midY - 30}
          width={104}
          height={60}
          rx={10}
          fill="var(--color-accent-cyan)"
          fillOpacity={0.1}
          stroke="var(--color-accent-cyan)"
          strokeWidth={1.5}
        />
        <text
          x={colModel}
          y={midY - 6}
          textAnchor="middle"
          fill="var(--color-accent-cyan)"
          fontSize={11}
          fontWeight="bold"
        >
          Models
        </text>
        <text
          x={colModel}
          y={midY + 12}
          textAnchor="middle"
          fill="var(--color-accent-cyan)"
          fontSize={10.5}
          opacity={0.85}
        >
          {models} models
        </text>

        {plugins > 0 && (
          <g>
            <rect
              x={colDecision - 30}
              y={midY + 40}
              width={60}
              height={22}
              rx={11}
              fill="var(--color-accent-purple)"
              fillOpacity={0.15}
              stroke="var(--color-accent-purple)"
              strokeWidth={1}
            />
            <text
              x={colDecision}
              y={midY + 55}
              textAnchor="middle"
              fill="var(--color-accent-purple)"
              fontSize={10}
            >
              {plugins} plugins
            </text>
          </g>
        )}

        <text
          x={colSignal}
          y={height - 4}
          textAnchor="middle"
          fill="var(--color-text-muted)"
          fontSize={9}
          letterSpacing="0.05em"
        >
          SIGNALS
        </text>
        <text
          x={colDecision}
          y={height - 4}
          textAnchor="middle"
          fill="var(--color-text-muted)"
          fontSize={9}
          letterSpacing="0.05em"
        >
          DECISIONS
        </text>
        <text
          x={colModel}
          y={height - 4}
          textAnchor="middle"
          fill="var(--color-text-muted)"
          fontSize={9}
          letterSpacing="0.05em"
        >
          MODELS
        </text>
      </svg>
    )
  },
)

DashboardMiniFlowDiagram.displayName = 'DashboardMiniFlowDiagram'

export default DashboardMiniFlowDiagram
