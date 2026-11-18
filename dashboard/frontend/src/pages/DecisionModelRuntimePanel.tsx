import { useState } from 'react'
import { ChevronDown, Cpu } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import {
  Alert,
  Card,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  EmptyState,
  KeyValueGrid,
  MiniStat,
  ProviderLogo,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  StatusLabel,
} from '../ui'
import { canAccessDashboardPath } from '../utils/accessControl'
import { decisionRuntimeModelName, type ModelRuntimeInventory } from './decisionModelManagement'
import {
  DECISION_MODEL_TIME_WINDOWS,
  decisionModelChartPoints,
  type DecisionModelTimeWindow,
} from './decisionModelMetrics'
import DecisionModelMonitoringCharts, {
  DecisionModelMetricCards,
} from './DecisionModelMonitoringCharts'
import DecisionReplicaStatus from './DecisionReplicaStatus'
import { useDecisionModelMetrics } from './useDecisionModelMetrics'

const shown = (value?: string | number) =>
  value === undefined || value === '' ? 'Not reported' : String(value)

export default function DecisionModelRuntimePanel({
  inventory,
  refreshedAt,
  loading,
  initialDeployment,
}: {
  inventory: ModelRuntimeInventory | null
  refreshedAt: Date | null
  loading: boolean
  initialDeployment?: string | null
}) {
  const { user } = useAuth()
  const [timeWindow, setTimeWindow] = useState<DecisionModelTimeWindow>(3_600)
  const [selectedDeployment, setSelectedDeployment] = useState(initialDeployment ?? '')
  const canReadMetrics = canAccessDashboardPath(user, '/observe/metrics')
  const deployments = inventory?.deployments ?? []
  const deployment = deployments.find(({ name }) => name === selectedDeployment) ?? deployments[0]
  const metrics = useDecisionModelMetrics(
    deployment ? [deployment.name] : [],
    refreshedAt,
    canReadMetrics,
    timeWindow,
  )
  const points = deployment ? decisionModelChartPoints(metrics.series, deployment.name) : []
  const ready = deployment?.ready && deployment.state === 'ready'

  return (
    <section className="flex flex-col gap-4" aria-labelledby="decision-runtime-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 id="decision-runtime-title" className="text-[15px] font-semibold tracking-[-0.012em]">
            Model runtime deployments
          </h2>
          <p className="text-[13px] text-muted-foreground">
            Traffic, latency and cache behavior for each runtime, from Prometheus.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {deployments.length > 1 ? (
            <Select value={deployment?.name} onValueChange={setSelectedDeployment}>
              <SelectTrigger size="sm" className="gap-2" aria-label="Runtime deployment">
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                {deployments.map((candidate) => (
                  <SelectItem key={candidate.name} value={candidate.name}>
                    <span className="font-mono text-[12.5px]">{candidate.name}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : null}
          <Segmented
            aria-label="Monitoring time range"
            value={String(timeWindow)}
            onValueChange={(value) => setTimeWindow(Number(value) as DecisionModelTimeWindow)}
            options={DECISION_MODEL_TIME_WINDOWS.map((window) => ({
              value: String(window.seconds),
              label: window.label,
            }))}
          />
        </div>
      </div>
      {!canReadMetrics ? (
        <Alert tone="neutral" title="Statistics need observability access">
          Viewing model statistics requires observability read access.
        </Alert>
      ) : metrics.unavailable.length > 0 ? (
        <Alert tone="warning" title="Statistics are incomplete">
          Some model statistics are unavailable. Check the Prometheus connection and router scrape.{' '}
          Unavailable: {metrics.unavailable.join(', ')}.
        </Alert>
      ) : null}
      {!deployment ? (
        <Card>
          <EmptyState
            compact
            icon={Cpu}
            tone="violet"
            title={
              loading ? 'Loading runtime deployments…' : 'No model runtime deployments reported.'
            }
            description={
              loading
                ? 'Fetching the latest deployment observations.'
                : 'Deploy a decision model, or start the Router, to inspect its runtime and collected observations.'
            }
          />
        </Card>
      ) : (
        <article aria-label={deployment.name} className="flex flex-col gap-4">
          <Card className="gap-4 p-5">
            <div className="flex flex-wrap items-center gap-3">
              <ProviderLogo provider="vllm" size={20} framed />
              <div className="flex min-w-0 flex-1 flex-col">
                <h3 className="truncate font-mono text-[14px] font-semibold">
                  {decisionRuntimeModelName(deployment) ?? deployment.name}
                </h3>
                <span className="truncate text-xs text-subtle-foreground">
                  Deployment <span className="font-mono">{deployment.name}</span>
                </span>
              </div>
              <StatusLabel tone={ready ? 'success' : deployment.ready ? 'warning' : 'neutral'}>
                {ready ? 'Ready' : deployment.state || 'Not ready'}
              </StatusLabel>
            </div>
            {deployment.reason ? (
              <p className="text-xs text-muted-foreground">{deployment.reason}</p>
            ) : null}
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <MiniStat
                label="Device"
                value={<span className="font-mono text-base">{shown(deployment.device)}</span>}
              />
              <MiniStat
                label="Engine backend"
                value={<span className="font-mono text-base">{shown(deployment.engine)}</span>}
              />
              <MiniStat label="Restarts" value={deployment.restarts ?? 'Not reported'} />
              <MiniStat
                label="Ownership"
                value={
                  <span className="text-base">
                    {deployment.managed ? 'Managed worker' : 'External deployment'}
                  </span>
                }
              />
            </div>
            <DecisionReplicaStatus deployment={deployment} />
            <Collapsible>
              <CollapsibleTrigger className="group inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40">
                <ChevronDown className="size-3.5 transition-transform group-data-[state=open]:rotate-180" />
                Deployment details
              </CollapsibleTrigger>
              <CollapsibleContent>
                <KeyValueGrid
                  className="mt-3 rounded-lg border border-border bg-surface-subtle/50 p-3"
                  columns={3}
                  items={[
                    { label: 'Deployment', value: deployment.name, mono: true },
                    { label: 'Repository', value: shown(deployment.repo), mono: true },
                    { label: 'Revision', value: shown(deployment.revision), mono: true },
                    { label: 'Profile', value: shown(deployment.profile) },
                    { label: 'Family', value: shown(deployment.family) },
                    { label: 'Process', value: shown(deployment.process), mono: true },
                    { label: 'Served model', value: shown(deployment.served_name), mono: true },
                    {
                      label: 'Supported APIs',
                      value: deployment.surfaces?.length
                        ? deployment.surfaces.join(', ')
                        : 'Not reported',
                    },
                    {
                      label: 'Model heads',
                      value: deployment.heads?.length
                        ? deployment.heads.map((head) => head.name).join(', ')
                        : 'Not reported',
                    },
                  ]}
                />
              </CollapsibleContent>
            </Collapsible>
          </Card>
          {canReadMetrics ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                <span>Latest observation · 5-minute rolling rates</span>
                <span>
                  {metrics.loading
                    ? 'Updating statistics…'
                    : metrics.updatedAt
                      ? `Observed ${metrics.updatedAt.toLocaleTimeString()}`
                      : 'Waiting for samples'}
                </span>
              </div>
              <DecisionModelMetricCards points={points} />
              <DecisionModelMonitoringCharts points={points} loading={metrics.loading} />
              <p className="text-xs text-subtle-foreground">
                Aggregated across routers scraped by Prometheus. Runtime calls may share a question
                batch; they do not count backend LLM chat requests. Gaps mean no usable samples, not
                zero. Token counts and GPU memory are not reported by these metrics.
              </p>
            </>
          ) : null}
        </article>
      )}
    </section>
  )
}
