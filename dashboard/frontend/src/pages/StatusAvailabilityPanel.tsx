import { useId, useRef, useState, type KeyboardEvent } from 'react'
import {
  CheckCircle2,
  CircleDashed,
  OctagonX,
  PowerOff,
  ServerOff,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react'

import {
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  ChartLegend,
  EmptyState,
  StatusDot,
  StatusLabel,
  cn,
  type StatusTone,
} from '../ui'
import type {
  ServiceStatus,
  ServiceStatusHistory,
  StatusHistoryState,
  SystemStatus,
} from '../utils/routerRuntime'
import {
  deploymentUptime,
  describeOverall,
  describeService,
  formatStatusLabel,
  formatUptime,
  historyHourLabel,
  historySummary,
  historyUptime,
  orderedServiceNames,
} from './statusPageSupport'

interface StatusAvailabilityPanelProps {
  status: SystemStatus | null
  lastUpdated: Date | null
  unavailable?: boolean
}

const segmentColor: Record<StatusHistoryState, string> = {
  operational: 'bg-success/80 hover:bg-success',
  starting: 'bg-info/70 hover:bg-info',
  unavailable: 'bg-destructive/80 hover:bg-destructive',
  unknown: 'bg-muted hover:bg-border-strong',
}

const toneAccent: Record<StatusTone, string> = {
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-destructive',
  info: 'bg-info',
  neutral: 'bg-border-strong',
}

const toneIcon: Record<StatusTone, string> = {
  success: 'bg-success-soft text-success ring-success/25',
  warning: 'bg-warning-soft text-warning ring-warning/30',
  danger: 'bg-destructive-soft text-destructive ring-destructive/25',
  info: 'bg-info-soft text-info ring-info/25',
  neutral: 'bg-surface-subtle text-muted-foreground ring-border',
}

const serviceGrid =
  'grid-cols-[minmax(0,1fr)_7.5rem] xl:grid-cols-[minmax(200px,1.1fr)_minmax(0,2.4fr)_76px_132px]'

const historyStartLabel = (hourCount: number) =>
  hourCount <= 1 ? 'Now' : `${hourCount - 1} hours ago`

function StatusHistoryTrack({ service }: { service: ServiceStatusHistory }) {
  const tooltipId = useId()
  const segmentRefs = useRef<Array<HTMLButtonElement | null>>([])
  const lastIndex = Math.max(0, service.hours.length - 1)
  const [tabStopIndex, setTabStopIndex] = useState(lastIndex)
  const [activeIndex, setActiveIndex] = useState<number | null>(null)
  const activeHour = activeIndex === null ? null : service.hours[activeIndex]

  const focusSegment = (index: number) => {
    const nextIndex = Math.max(0, Math.min(index, lastIndex))
    setTabStopIndex(nextIndex)
    setActiveIndex(nextIndex)
    segmentRefs.current[nextIndex]?.focus()
  }

  const activatePointerSegment = (clientX: number, track: HTMLDivElement) => {
    const bounds = track.getBoundingClientRect()
    if (bounds.width <= 0 || service.hours.length === 0) return
    const position = Math.max(0, Math.min(clientX - bounds.left, bounds.width - 1))
    const index = Math.min(lastIndex, Math.floor((position / bounds.width) * service.hours.length))
    setTabStopIndex(index)
    setActiveIndex(index)
  }

  const handleSegmentKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex: number | null = null
    if (event.key === 'ArrowLeft') nextIndex = Math.max(0, index - 1)
    else if (event.key === 'ArrowRight') nextIndex = Math.min(lastIndex, index + 1)
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = lastIndex
    if (nextIndex === null) return
    event.preventDefault()
    focusSegment(nextIndex)
  }

  const placement =
    activeIndex === null || activeIndex < 8
      ? 'start'
      : activeIndex > lastIndex - 8
        ? 'end'
        : 'center'

  return (
    <div className="relative min-w-0">
      {activeHour ? (
        <span
          id={tooltipId}
          role="tooltip"
          className={cn(
            'pointer-events-none absolute bottom-[calc(100%+6px)] z-10 whitespace-nowrap rounded-md bg-foreground px-2 py-1 text-[11px] font-medium text-background shadow-md',
            placement === 'start' && 'left-0',
            placement === 'end' && 'right-0',
            placement === 'center' && '-translate-x-1/2',
          )}
          style={
            placement === 'center' && activeIndex !== null
              ? { left: `${(activeIndex / Math.max(1, lastIndex)) * 100}%` }
              : undefined
          }
        >
          {historyHourLabel(activeHour)}
        </span>
      ) : null}
      <div
        className="flex h-7 items-stretch gap-px xl:gap-[2px]"
        role="group"
        aria-label={historySummary(service)}
        onPointerDown={(event) => activatePointerSegment(event.clientX, event.currentTarget)}
        onPointerMove={(event) => activatePointerSegment(event.clientX, event.currentTarget)}
        onPointerLeave={(event) => {
          if (!event.currentTarget.contains(document.activeElement)) setActiveIndex(null)
        }}
        onBlur={(event) => {
          const nextTarget = event.relatedTarget
          if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
            setActiveIndex(null)
          }
        }}
      >
        {service.hours.map((hour, index) => (
          <button
            key={hour.observedAt}
            ref={(element) => {
              segmentRefs.current[index] = element
            }}
            type="button"
            className={cn(
              'min-w-0 flex-1 rounded-[2px] outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring',
              segmentColor[hour.status],
              activeIndex === index && 'ring-2 ring-ring/60',
            )}
            tabIndex={index === tabStopIndex ? 0 : -1}
            aria-label={`History hour ${index + 1} of ${service.hours.length}, ${historyHourLabel(hour)}`}
            aria-describedby={activeIndex === index ? tooltipId : undefined}
            data-status-history-hour
            onFocus={() => {
              setTabStopIndex(index)
              setActiveIndex(index)
            }}
            onClick={() => setActiveIndex(index)}
            onKeyDown={(event) => handleSegmentKeyDown(event, index)}
          />
        ))}
      </div>
    </div>
  )
}

const overviewIcon = (status: SystemStatus | null, tone: StatusTone): LucideIcon => {
  if (!status) return CircleDashed
  if (tone === 'success') return CheckCircle2
  if (!status.services.length) return ServerOff
  if (status.overall === 'not_running' || status.overall === 'stopped') return PowerOff
  return tone === 'danger' ? OctagonX : TriangleAlert
}

export default function StatusAvailabilityPanel({
  status,
  lastUpdated,
  unavailable = false,
}: StatusAvailabilityPanelProps) {
  const overall = describeOverall(status)
  const tone: StatusTone = !status && unavailable ? 'danger' : overall.tone
  const Icon = overviewIcon(status, tone)
  const healthyServices = status?.services.filter((service) => service.healthy).length ?? 0
  const histories = new Map<string, ServiceStatusHistory>(
    (status?.history?.services ?? []).map((history) => [history.name, history] as const),
  )
  const currentServices = new Map<string, ServiceStatus>(
    (status?.services ?? []).map((service) => [service.name, service] as const),
  )
  const serviceNames = orderedServiceNames(status)
  const windowHours = status?.history?.windowHours ?? 90
  const stats = [
    {
      label: 'Services',
      value: status ? `${healthyServices}/${status.services.length}` : '—',
      hint: status ? 'healthy now' : undefined,
    },
    {
      label: 'Current state',
      value: status ? formatStatusLabel(status.overall) : 'Unavailable',
    },
    {
      label: `Uptime · ${windowHours} hours`,
      value: formatUptime(deploymentUptime(status)),
      hint: 'observed hours',
    },
    {
      label: 'Last checked',
      value: lastUpdated?.toLocaleTimeString() || '—',
    },
  ]

  return (
    <div className="flex min-w-0 flex-col gap-4" data-testid="status-availability">
      <Card
        data-testid="status-overview"
        aria-live="polite"
        className="relative min-w-0 overflow-hidden"
      >
        <span
          aria-hidden="true"
          className={cn('pointer-events-none absolute inset-y-0 left-0 w-1', toneAccent[tone])}
        />
        <div className="flex flex-wrap items-center gap-4 px-5 py-4">
          <span
            className={cn(
              'grid size-11 shrink-0 place-items-center rounded-xl ring-1 ring-inset',
              toneIcon[tone],
            )}
          >
            <Icon className="size-5" strokeWidth={1.8} aria-hidden="true" />
          </span>
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-[17px] font-semibold tracking-[-0.015em]">{overall.title}</h2>
              {status && status.services.length > 0 ? (
                <Badge
                  tone={
                    overall.fullyOperational
                      ? 'success'
                      : tone === 'danger'
                        ? 'danger'
                        : tone === 'info'
                          ? 'info'
                          : 'warning'
                  }
                >
                  {healthyServices} of {status.services.length} operational
                </Badge>
              ) : null}
            </div>
            <p className="text-[13px] text-muted-foreground">{overall.copy}</p>
          </div>
        </div>
        <dl className="grid grid-cols-2 border-t border-border sm:grid-cols-4">
          {stats.map((stat, index) => (
            <div
              key={stat.label}
              className={cn(
                'flex min-w-0 flex-col gap-0.5 border-border px-5 py-3',
                index % 2 === 1 && 'border-l',
                index > 0 && 'sm:border-l',
                index >= 2 && 'max-sm:border-t',
              )}
            >
              <dt className="text-xs text-subtle-foreground">{stat.label}</dt>
              <dd className="flex min-w-0 flex-wrap items-baseline gap-x-1.5">
                <span className="truncate text-[15px] font-semibold tabular-nums">
                  {stat.value}
                </span>
                {stat.hint ? (
                  <span className="text-xs text-muted-foreground">{stat.hint}</span>
                ) : null}
              </dd>
            </div>
          ))}
        </dl>
      </Card>

      <Card
        role="region"
        className="min-w-0"
        aria-labelledby="component-status-title"
        data-testid="status-services-section"
      >
        <CardHeader className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
          <div className="flex min-w-0 flex-col gap-1">
            <CardTitle id="component-status-title">Services</CardTitle>
            <CardDescription>{windowHours}-hour observed history</CardDescription>
          </div>
          {serviceNames.length ? (
            <div role="group" aria-label="Status history legend">
              <ChartLegend
                items={[
                  { label: 'Operational', color: 'var(--success)' },
                  { label: 'Starting', color: 'var(--info)' },
                  { label: 'Unavailable', color: 'var(--destructive)' },
                  { label: 'No observation', color: 'var(--muted)' },
                ]}
              />
            </div>
          ) : null}
        </CardHeader>
        <CardContent className="pt-3">
          {serviceNames.length ? (
            <>
              <div
                className={cn(
                  'hidden gap-4 border-b border-border pb-2 text-xs text-subtle-foreground xl:grid',
                  serviceGrid,
                )}
                aria-hidden="true"
              >
                <span>Service</span>
                <span className="flex justify-between">
                  <span>{historyStartLabel(windowHours)}</span>
                  <span>Now</span>
                </span>
                <span className="text-right">Uptime</span>
                <span className="text-right">State</span>
              </div>
              <div className="divide-y divide-border" data-testid="status-service-grid">
                {serviceNames.map((name) => {
                  const service = currentServices.get(name)
                  const history = histories.get(name)
                  const state = describeService(service)
                  const uptime = history ? historyUptime(history) : null
                  return (
                    <article
                      key={name}
                      className={cn(
                        'grid min-w-0 items-center gap-x-4 gap-y-2 py-3.5',
                        serviceGrid,
                      )}
                    >
                      <div className="order-1 flex min-w-0 items-start gap-2.5">
                        <StatusDot
                          tone={state.tone}
                          pulse={state.tone === 'info'}
                          className="mt-1.5"
                        />
                        <div className="flex min-w-0 flex-col gap-0.5">
                          <h3 className="truncate text-[13px] font-medium">{name}</h3>
                          <span
                            className="truncate text-xs text-subtle-foreground"
                            title={service?.message}
                          >
                            {[
                              service?.component && formatStatusLabel(service.component),
                              service?.message,
                            ]
                              .filter(Boolean)
                              .join(' · ') || 'No details reported'}
                          </span>
                        </div>
                      </div>
                      {history ? (
                        <div className="order-3 min-w-0 xl:order-2">
                          <StatusHistoryTrack service={history} />
                          <div className="mt-1 flex justify-between text-[11px] text-subtle-foreground xl:hidden">
                            <span>{historyStartLabel(history.hours.length)}</span>
                            <span>Now</span>
                          </div>
                        </div>
                      ) : (
                        <div className="order-3 flex h-7 items-center rounded-md border border-dashed border-border-strong px-3 text-xs text-muted-foreground xl:order-2">
                          History unavailable
                        </div>
                      )}
                      <span
                        className="order-4 self-start whitespace-nowrap text-right font-mono text-xs tabular-nums xl:order-3 xl:self-center"
                        title={uptime === null ? undefined : 'Operational share of observed hours'}
                      >
                        <span className="text-subtle-foreground xl:hidden">Uptime </span>
                        {formatUptime(uptime)}
                      </span>
                      <span className="order-2 flex flex-col items-end gap-0.5 xl:order-4">
                        <StatusLabel tone={state.tone}>{state.label}</StatusLabel>
                        {state.condition !== state.label && state.condition !== 'Normal' ? (
                          <span className="text-[11px] text-subtle-foreground">
                            {state.condition}
                          </span>
                        ) : null}
                      </span>
                    </article>
                  )
                })}
              </div>
            </>
          ) : (
            <EmptyState
              compact
              icon={ServerOff}
              tone="slate"
              title="No services reported"
              description="Availability appears here when the Router starts."
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
}
