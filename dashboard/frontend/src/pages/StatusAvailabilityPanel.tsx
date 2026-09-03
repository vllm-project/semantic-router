import { useId, useRef, useState, type KeyboardEvent } from 'react'

import ProductIcon from '../components/ProductIcon'
import type {
  ServiceStatus,
  ServiceStatusHistory,
  StatusHistoryHour,
  StatusHistoryState,
  SystemStatus,
} from '../utils/routerRuntime'
import { formatStatusLabel } from './statusPageSupport'
import styles from './StatusPage.module.css'

interface StatusAvailabilityPanelProps {
  status: SystemStatus | null
  lastUpdated: Date | null
}

const historyClassName = (state: StatusHistoryState) => {
  switch (state) {
    case 'operational':
      return styles.historyOperational
    case 'starting':
      return styles.historyStarting
    case 'unavailable':
      return styles.historyUnavailable
    default:
      return styles.historyUnknown
  }
}

function historySummary(service: ServiceStatusHistory): string {
  const counts: Record<StatusHistoryState, number> = {
    operational: 0,
    starting: 0,
    unavailable: 0,
    unknown: 0,
  }
  for (const hour of service.hours) counts[hour.status] += 1
  return [
    `${service.name}, ${service.hours.length}-hour observed history:`,
    `${counts.operational} operational, ${counts.starting} starting,`,
    `${counts.unavailable} unavailable, ${counts.unknown} unknown.`,
  ].join(' ')
}

const historyHourLabel = (hour: StatusHistoryHour) =>
  `${hour.observedAt.slice(0, 10)} ${hour.observedAt.slice(11, 16)} UTC: ${formatStatusLabel(hour.status)}`

const historyStartLabel = (hourCount: number) => {
  return hourCount <= 1 ? 'Now' : `${hourCount - 1} hours ago`
}

const historyUptimeLabel = (service: ServiceStatusHistory) => {
  const observedHours = service.hours.filter((hour) => hour.status !== 'unknown')
  if (!observedHours.length) return 'No uptime data'
  const operationalHours = observedHours.filter((hour) => hour.status === 'operational').length
  const uptime = (operationalHours / observedHours.length) * 100
  const precision = uptime === 0 || uptime === 100 ? 0 : 2
  return `${uptime.toFixed(precision)}% uptime`
}

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

  const tooltipPlacement =
    activeIndex === null || activeIndex < 8
      ? styles.historyTooltipStart
      : activeIndex > lastIndex - 8
        ? styles.historyTooltipEnd
        : styles.historyTooltipCenter
  const tooltipStyle =
    activeIndex !== null && tooltipPlacement === styles.historyTooltipCenter
      ? { left: `${(activeIndex / Math.max(1, lastIndex)) * 100}%` }
      : undefined

  return (
    <div className={styles.historyTrackShell}>
      {activeHour ? (
        <span
          id={tooltipId}
          role="tooltip"
          className={`${styles.historyTooltip} ${tooltipPlacement}`}
          style={tooltipStyle}
        >
          {historyHourLabel(activeHour)}
        </span>
      ) : null}
      <div
        className={styles.historyTrack}
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
            className={`${styles.historySegment} ${historyClassName(hour.status)}`}
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

export default function StatusAvailabilityPanel({
  status,
  lastUpdated,
}: StatusAvailabilityPanelProps) {
  const healthyServices = status?.services.filter((service) => service.healthy).length ?? 0
  const fullyOperational = Boolean(
    status &&
      status.services.length > 0 &&
      status.overall === 'healthy' &&
      healthyServices === status.services.length,
  )
  const noServices = Boolean(status && status.services.length === 0)
  const healthLabel = status ? formatStatusLabel(status.overall) : 'Unavailable'
  const bannerTitle = fullyOperational
    ? 'All systems operational'
    : !status
      ? 'Status unavailable'
      : noServices
        ? 'No running services detected'
        : healthLabel
  const bannerCopy = fullyOperational
    ? 'Services are responding normally.'
    : !status
      ? 'Live service status could not be verified.'
      : noServices
        ? 'Availability will appear when the Router starts.'
        : 'One or more components need attention.'
  const histories = new Map<string, ServiceStatusHistory>(
    (status?.history?.services ?? []).map((history) => [history.name, history] as const),
  )
  const currentServices = new Map<string, ServiceStatus>(
    (status?.services ?? []).map((service) => [service.name, service] as const),
  )
  const serviceNames = [
    ...(status?.services.map((service) => service.name) ?? []),
    ...(status?.history?.services
      .map((service) => service.name)
      .filter((name) => !currentServices.has(name)) ?? []),
  ]

  return (
    <div className={styles.availabilityPanel} data-testid="status-availability">
      <section
        data-testid="status-overview"
        className={`${styles.overallBanner} ${fullyOperational ? styles.overallHealthy : styles.overallDegraded}`}
        aria-live="polite"
      >
        <span className={styles.overallIcon}>
          <ProductIcon name={fullyOperational ? 'check' : 'alert'} aria-hidden="true" />
        </span>
        <div>
          <h2>{bannerTitle}</h2>
          <p>{bannerCopy}</p>
        </div>
        <dl>
          <div>
            <dt>Services</dt>
            <dd>{status ? `${healthyServices}/${status.services.length}` : '—'}</dd>
          </div>
          <div>
            <dt>Current state</dt>
            <dd>{status ? healthLabel : 'Unavailable'}</dd>
          </div>
          <div>
            <dt>Last checked</dt>
            <dd>{lastUpdated?.toLocaleTimeString() || '—'}</dd>
          </div>
        </dl>
      </section>

      <section
        className={styles.componentBoard}
        aria-labelledby="component-status-title"
        data-testid="status-services-section"
      >
        <div className={styles.componentBoardHeader}>
          <div>
            <span>Availability</span>
            <h2 id="component-status-title">Services</h2>
          </div>
          <span>90-hour observed history</span>
        </div>

        {serviceNames.length ? (
          <div className={styles.componentRows} data-testid="status-service-grid">
            {serviceNames.map((name) => {
              const service: ServiceStatus | undefined = currentServices.get(name)
              const history = histories.get(name)
              const stateLabel = service
                ? service.healthy
                  ? 'Operational'
                  : formatStatusLabel(service.status)
                : 'Not reported'
              const conditionLabel = service
                ? service.healthy
                  ? 'Normal'
                  : service.status === 'starting'
                    ? 'Starting'
                    : 'Unavailable'
                : 'Unknown'
              return (
                <article key={name} className={styles.componentRow}>
                  <header className={styles.componentIdentity}>
                    <strong>{name}</strong>
                  </header>
                  <span
                    className={
                      service
                        ? service.healthy
                          ? styles.componentOperational
                          : service.status === 'unavailable'
                            ? styles.componentUnavailable
                            : styles.componentIssue
                        : styles.componentUnknown
                    }
                  >
                    <ProductIcon name={service?.healthy ? 'check' : 'alert'} aria-hidden="true" />
                    {stateLabel}
                  </span>
                  {history ? (
                    <div className={styles.historyBlock}>
                      <StatusHistoryTrack service={history} />
                      <div className={styles.historyAxis}>
                        <span>{historyStartLabel(history.hours.length)}</span>
                        <span>{historyUptimeLabel(history)}</span>
                        <span>Now</span>
                      </div>
                      <span className={styles.componentCondition}>{conditionLabel}</span>
                    </div>
                  ) : (
                    <div className={styles.historyMissing}>
                      <span>History unavailable</span>
                      <strong>{conditionLabel}</strong>
                    </div>
                  )}
                </article>
              )
            })}
          </div>
        ) : (
          <div className={styles.noServices}>
            <strong>No services reported</strong>
            <span>Availability appears here when the Router starts.</span>
          </div>
        )}

        {serviceNames.length ? (
          <div className={styles.historyLegend} aria-label="Status history legend">
            <span>
              <i className={styles.historyOperational} aria-hidden="true" /> Operational
            </span>
            <span>
              <i className={styles.historyStarting} aria-hidden="true" /> Starting
            </span>
            <span>
              <i className={styles.historyUnavailable} aria-hidden="true" /> Unavailable
            </span>
            <span>
              <i className={styles.historyUnknown} aria-hidden="true" /> No observation
            </span>
          </div>
        ) : null}
      </section>
    </div>
  )
}
