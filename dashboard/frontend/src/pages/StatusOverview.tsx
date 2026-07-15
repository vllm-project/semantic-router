import React from 'react'
import {
  describeRouterRuntime,
  type ModelStatusSummary,
  type RouterRuntimeStatus,
  type SystemStatus,
} from '../utils/routerRuntime'
import styles from './StatusOverview.module.css'

interface StatusOverviewProps {
  status: SystemStatus
  modelStatus: ModelStatusSummary
  runtime: RouterRuntimeStatus | null
  healthyServices: number
  loadedModels: number
  knownModels: number
}

interface ModelFleetSummary {
  label: string
  detail: string
  tone: 'ok' | 'warn' | 'down'
}

function getOverallLabel(overall: string): string {
  if (overall === 'not_running') return 'Not Running'
  if (overall === 'stopped') return 'Stopped'
  return overall.charAt(0).toUpperCase() + overall.slice(1)
}

function formatDeploymentType(type: string): string {
  if (type === 'none') return 'Not Detected'
  return type.charAt(0).toUpperCase() + type.slice(1)
}

function getOverallTone(overall: string): 'ok' | 'warn' | 'down' {
  if (overall === 'healthy') return 'ok'
  if (overall === 'degraded') return 'warn'
  return 'down'
}

function getToneClass(tone: 'ok' | 'warn' | 'down'): string {
  if (tone === 'ok') return styles.toneOk
  if (tone === 'warn') return styles.toneWarn
  return styles.toneDown
}

function formatPhaseLabel(phase?: string): string {
  if (!phase) return 'Pending'
  return phase
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase())
    .replace(/ Models$/, '')
}

function getModelFleetSummary(
  status: SystemStatus,
  modelStatus: ModelStatusSummary,
  runtime: RouterRuntimeStatus | null,
  loadedModels: number,
  knownModels: number,
): ModelFleetSummary {
  if (status.overall === 'not_running' || status.overall === 'stopped') {
    return { label: 'Offline', detail: 'Router is not running', tone: 'down' }
  }

  if (runtime) {
    return {
      label: formatPhaseLabel(runtime.phase),
      detail: describeRouterRuntime(runtime),
      tone: modelStatus.tone,
    }
  }

  if (knownModels === 0) {
    return {
      label: 'Not reported',
      detail: 'The router has not reported model metadata yet.',
      tone: 'warn',
    }
  }

  const modelsSummary = status.models?.summary
  if (modelsSummary?.phase === 'error') {
    return {
      label: 'Error',
      detail: modelsSummary.message || 'Router model startup failed',
      tone: 'down',
    }
  }

  if (modelsSummary?.ready === false) {
    return {
      label: formatPhaseLabel(modelsSummary.phase),
      detail: modelsSummary.message || 'Reported models are still starting',
      tone: 'warn',
    }
  }

  if (modelsSummary?.ready === true || loadedModels >= knownModels) {
    return {
      label: 'Ready',
      detail: modelsSummary?.message || 'All reported models are ready',
      tone: 'ok',
    }
  }

  return {
    label: loadedModels > 0 ? 'Partial' : 'Pending',
    detail: modelsSummary?.message || 'Some reported models are not ready',
    tone: 'warn',
  }
}

const StatusOverview: React.FC<StatusOverviewProps> = ({
  status,
  modelStatus,
  runtime,
  healthyServices,
  loadedModels,
  knownModels,
}) => {
  const hasServices = status.services.length > 0
  const hasModels = knownModels > 0
  const overallLabel = hasServices ? getOverallLabel(status.overall) : 'Not reported'
  const overallTone = hasServices ? getOverallTone(status.overall) : 'warn'
  const allServicesHealthy = hasServices && healthyServices === status.services.length
  const runtimeModelCount =
    typeof runtime?.total_models === 'number' && runtime.total_models > 0
      ? `${runtime.ready_models ?? 0}/${runtime.total_models}`
      : null
  const modelCount = hasModels ? `${loadedModels}/${knownModels}` : (runtimeModelCount ?? '—')
  const modelFleet = getModelFleetSummary(status, modelStatus, runtime, loadedModels, knownModels)

  return (
    <section className={styles.panel} data-testid="status-overview">
      <div className={styles.panelHeader}>
        <div>
          <span className={styles.eyebrow}>Runtime overview</span>
          <h2 className={styles.title}>Router and model readiness</h2>
        </div>
        <span
          className={`${styles.stateBadge} ${getToneClass(overallTone)}`}
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <span className={styles.stateDot} aria-hidden="true" />
          {overallLabel}
        </span>
      </div>

      <div className={styles.overviewBody}>
        <div className={styles.healthSummary}>
          <span className={styles.healthLabel}>Current health</span>
          <strong className={`${styles.healthValue} ${getToneClass(overallTone)}`}>
            {overallLabel}
          </strong>
          <p className={styles.healthNarrative}>
            {!hasServices
              ? 'No router services have been reported.'
              : allServicesHealthy
                ? 'All router services are responding normally.'
                : `${healthyServices} of ${status.services.length} router services are healthy.`}
          </p>
        </div>

        <dl className={styles.metricGrid}>
          <div className={styles.metric} data-testid="status-metric-services">
            <dt>Services</dt>
            <dd>
              {status.services.length > 0 ? `${healthyServices}/${status.services.length}` : '—'}
            </dd>
            <dd className={styles.metricHint}>
              {!hasServices ? 'Not reported' : allServicesHealthy ? 'All online' : 'Check services'}
            </dd>
          </div>
          <div className={styles.metric} data-testid="status-metric-models">
            <dt>Models</dt>
            <dd>{modelCount}</dd>
            <dd className={styles.metricHint}>{modelFleet.label}</dd>
          </div>
          <div className={styles.metric} data-testid="status-metric-deployment">
            <dt>Deployment</dt>
            <dd>{formatDeploymentType(status.deployment_type)}</dd>
            <dd className={styles.metricHint}>Active runtime</dd>
          </div>
          <div className={styles.metric} data-testid="status-metric-version">
            <dt>Version</dt>
            <dd>{status.version || 'Unknown'}</dd>
            <dd className={styles.metricHint}>Router build</dd>
          </div>
        </dl>
      </div>

      <div className={styles.readinessRow}>
        <div
          className={styles.readinessIdentity}
          data-testid="status-model-fleet"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <span
            className={`${styles.readinessDot} ${getToneClass(modelFleet.tone)}`}
            aria-hidden="true"
          />
          <div>
            <span className={styles.readinessLabel}>Model fleet</span>
            <strong className={styles.readinessValue}>{modelFleet.label}</strong>
          </div>
        </div>
        <p className={styles.readinessDescription}>{modelFleet.detail}</p>
        {runtime && (
          <div className={styles.runtimeFacts}>
            <span>{runtime.phase.replace(/_/g, ' ')}</span>
            {runtime.downloading_model && <span>{runtime.downloading_model}</span>}
            {typeof runtime.total_models === 'number' && runtime.total_models > 0 && (
              <span>
                {runtime.ready_models ?? 0}/{runtime.total_models} ready
              </span>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

export default StatusOverview
