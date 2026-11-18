import { useId } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as ChartTip,
  XAxis,
  YAxis,
} from 'recharts'

import {
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartLegend,
  ChartTooltip,
  chartAxis,
  chartGrid,
  useChartReady,
} from '../ui'
import { formatRequestCount, type PublicModelTrafficEntry } from './usePublicModelTraffic'

const SERIES_COLORS = [
  'var(--chart-1)',
  'var(--chart-4)',
  'var(--chart-3)',
  'var(--chart-6)',
  'var(--chart-7)',
]

const hourLabel = (start: string) => {
  const time = Date.parse(start)
  return Number.isNaN(time)
    ? ''
    : new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
}

/** Requests per public model over the last 24 hours, stacked by the hour. */
export default function ConfigPageRecipesTrafficChart({
  traffic,
}: {
  traffic: ReadonlyArray<readonly [string, PublicModelTrafficEntry]>
}) {
  const navigate = useNavigate()
  const ready = useChartReady()
  const id = useId().replace(/:/g, '')
  const series = traffic.filter(([, entry]) => entry.requests > 0)
  if (series.length === 0) return null
  const hours = series.reduce<string[]>(
    (longest, [, entry]) => (entry.hours.length > longest.length ? entry.hours : longest),
    [],
  )
  const data = hours.map((start, index) => ({
    hour: hourLabel(start),
    // Model names may hold dots, which chart data keys read as paths.
    ...Object.fromEntries(series.map(([, entry], key) => [`m${key}`, entry.hourly[index] ?? 0])),
  }))
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Requests by public model</CardTitle>
        <CardDescription>Last 24 hours, stacked</CardDescription>
        <CardAction>
          <Button variant="ghost" size="xs" onClick={() => navigate('/observe/requests')}>
            Requests
            <ArrowRight />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="pt-2">
        <ChartLegend
          className="mb-2"
          items={series.map(([name, entry], index) => ({
            label: name,
            color: SERIES_COLORS[index % SERIES_COLORS.length],
            value: formatRequestCount(entry.requests),
          }))}
        />
        <div className="h-[220px]">
          {ready ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <defs>
                  {series.map(([name], index) => (
                    <linearGradient key={name} id={`pm-${id}-${index}`} x1="0" x2="0" y1="0" y2="1">
                      <stop
                        offset="0%"
                        stopColor={SERIES_COLORS[index % SERIES_COLORS.length]}
                        stopOpacity={0.26}
                      />
                      <stop
                        offset="100%"
                        stopColor={SERIES_COLORS[index % SERIES_COLORS.length]}
                        stopOpacity={0.02}
                      />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid {...chartGrid} />
                <XAxis dataKey="hour" {...chartAxis} interval={3} />
                <YAxis
                  {...chartAxis}
                  width={52}
                  allowDecimals={false}
                  tickFormatter={(value: number) => formatRequestCount(value)}
                />
                <ChartTip
                  cursor={{ stroke: 'var(--border-strong)', strokeDasharray: '3 3' }}
                  content={<ChartTooltip format={(value) => Number(value).toLocaleString()} />}
                />
                {series.map(([name], index) => (
                  <Area
                    key={name}
                    type="monotone"
                    dataKey={`m${index}`}
                    name={name}
                    stackId="requests"
                    stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                    strokeWidth={1.8}
                    fill={`url(#pm-${id}-${index})`}
                    animationDuration={900 + index * 150}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          ) : null}
        </div>
      </CardContent>
    </Card>
  )
}
