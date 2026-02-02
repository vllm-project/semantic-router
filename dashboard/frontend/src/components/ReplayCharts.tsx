import React, { useMemo } from 'react'
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import styles from './ReplayCharts.module.css'

interface Signal {
  keyword?: string[]
  embedding?: string[]
  domain?: string[]
  fact_check?: string[]
  user_feedback?: string[]
  preference?: string[]
  language?: string[]
  latency?: string[]
  context?: string[]
  complexity?: string[]
}

interface ReplayRecord {
  id: string
  timestamp: string
  request_id?: string
  decision?: string
  category?: string
  original_model?: string
  selected_model?: string
  reasoning_mode?: string
  confidence_score?: number
  selection_method?: string
  signals: Signal
  request_body?: string
  response_body?: string
  response_status?: number
  from_cache?: boolean
  streaming?: boolean
  request_body_truncated?: boolean
  response_body_truncated?: boolean
}

interface ReplayChartsProps {
  records: ReplayRecord[]
}

const COLORS = ['#76b900', '#8fd400', '#6ba300', '#5a8f00', '#718096', '#5a6c7d', '#606c7a', '#556b7d']

// Custom label renderer for pie charts with white text
interface PieLabelProps {
  cx: number
  cy: number
  midAngle: number
  innerRadius: number
  outerRadius: number
  percent: number
  name: string
}

const renderCustomLabel = ({ cx, cy, midAngle, outerRadius, percent, name }: PieLabelProps) => {
  const RADIAN = Math.PI / 180
  // Position label outside the pie chart
  const radius = outerRadius + 25
  const x = cx + radius * Math.cos(-midAngle * RADIAN)
  const y = cy + radius * Math.sin(-midAngle * RADIAN)

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      style={{ fontSize: '11px', fontWeight: '500' }}
    >
      {`${name}: ${(percent * 100).toFixed(0)}%`}
    </text>
  )
}

// Generate gradient colors from NVIDIA green to light gray
const generateBarColors = (count: number): string[] => {
  const colors: string[] = []
  for (let i = 0; i < count; i++) {
    const ratio = i / Math.max(count - 1, 1)
    // Interpolate from NVIDIA green (#76b900) to light gray (#9ca3af)
    const r = Math.round(118 + (156 - 118) * ratio)
    const g = Math.round(185 + (163 - 185) * ratio)
    const b = Math.round(0 + (175 - 0) * ratio)
    colors.push(`rgb(${r}, ${g}, ${b})`)
  }
  return colors
}

const ReplayCharts: React.FC<ReplayChartsProps> = ({ records }) => {
  // Calculate model statistics
  const modelData = useMemo(() => {
    const counts: Record<string, number> = {}
    records.forEach(record => {
      const model = record.selected_model || 'Unknown'
      counts[model] = (counts[model] || 0) + 1
    })
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([name, value]) => ({ name, value }))
  }, [records])

  const barColors = useMemo(() => generateBarColors(modelData.length), [modelData.length])

  // Calculate decision statistics
  const decisionData = useMemo(() => {
    const counts: Record<string, number> = {}
    records.forEach(record => {
      const decision = record.decision || 'Unknown'
      counts[decision] = (counts[decision] || 0) + 1
    })
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name, value]) => ({ name, value }))
  }, [records])

  // Calculate signal statistics
  const signalData = useMemo(() => {
    const counts: Record<string, number> = {}
    records.forEach(record => {
      const signals = record.signals
      if (signals.keyword?.length) counts['keyword'] = (counts['keyword'] || 0) + signals.keyword.length
      if (signals.embedding?.length) counts['embedding'] = (counts['embedding'] || 0) + signals.embedding.length
      if (signals.domain?.length) counts['domain'] = (counts['domain'] || 0) + signals.domain.length
      if (signals.fact_check?.length) counts['fact_check'] = (counts['fact_check'] || 0) + signals.fact_check.length
      if (signals.user_feedback?.length) counts['user_feedback'] = (counts['user_feedback'] || 0) + signals.user_feedback.length
      if (signals.preference?.length) counts['preference'] = (counts['preference'] || 0) + signals.preference.length
      if (signals.language?.length) counts['language'] = (counts['language'] || 0) + signals.language.length
      if (signals.latency?.length) counts['latency'] = (counts['latency'] || 0) + signals.latency.length
      if (signals.context?.length) counts['context'] = (counts['context'] || 0) + signals.context.length
      if (signals.complexity?.length) counts['complexity'] = (counts['complexity'] || 0) + signals.complexity.length
    })
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({ name, value }))
  }, [records])

  if (records.length === 0) {
    return null
  }

  return (
    <div className={styles.chartsContainer}>
      <div className={styles.chartsRow}>
        {/* Bar Chart */}
        <div className={styles.chartSection}>
          <h3 className={styles.chartTitle}>
            <svg className={styles.chartIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="18" />
              <rect x="14" y="8" width="7" height="13" />
            </svg>
            Model Selection
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelData} margin={{ top: 20, right: 0, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
              <XAxis
                dataKey="name"
                angle={-45}
                textAnchor="end"
                height={80}
                tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }}
              />
              <YAxis tick={{ fill: 'var(--color-text-secondary)', fontSize: 11 }} />
              <Tooltip
                cursor={false}
                contentStyle={{
                  background: 'var(--color-bg-secondary)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '4px',
                  color: 'var(--color-text-primary)'
                }}
                itemStyle={{
                  color: 'var(--color-text-primary)'
                }}
              />
              <Bar dataKey="value" name="Count">
                {modelData.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={barColors[index]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Decision Pie Chart */}
        <div className={styles.chartSection}>
          <h3 className={styles.chartTitle}>
            <svg className={styles.chartIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2 L12 12 L20 12" />
            </svg>
            Decision Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={decisionData}
                cx="50%"
                cy="50%"
                labelLine={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1 }}
                label={renderCustomLabel}
                outerRadius={70}
                fill="#8884d8"
                dataKey="value"
              >
                {decisionData.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: 'var(--color-bg-secondary)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '4px',
                  color: 'var(--color-text-primary)'
                }}
                itemStyle={{
                  color: 'var(--color-text-primary)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Signal Pie Chart */}
        <div className={styles.chartSection}>
          <h3 className={styles.chartTitle}>
            <svg className={styles.chartIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2 L12 12 L20 12" />
            </svg>
            Signal Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={signalData}
                cx="50%"
                cy="50%"
                labelLine={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1 }}
                label={renderCustomLabel}
                outerRadius={70}
                fill="#8884d8"
                dataKey="value"
              >
                {signalData.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: 'var(--color-bg-secondary)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '4px',
                  color: 'var(--color-text-primary)'
                }}
                itemStyle={{
                  color: 'var(--color-text-primary)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

export default ReplayCharts

