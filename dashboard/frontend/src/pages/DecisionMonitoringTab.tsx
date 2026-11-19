import { useSearchParams } from 'react-router-dom'
import { RefreshCw } from 'lucide-react'
import { Alert, Button, cn } from '../ui'
import DecisionModelRuntimePanel from './DecisionModelRuntimePanel'
import DecisionTaskMonitoring from './DecisionTaskMonitoring'
import SystemOneAutoMonitoring from './SystemOneAutoMonitoring'
import { useDecisionModelMonitoring } from './useDecisionModelMonitoring'

/** Operational state only: runtime observations and Prometheus statistics, never configuration. */
export default function DecisionMonitoringTab() {
  const runtime = useDecisionModelMonitoring()
  const [params] = useSearchParams()
  return (
    <div className="flex flex-col gap-8">
      <div className="-mb-4 flex flex-wrap items-center gap-3">
        <span className="text-xs text-muted-foreground" role="status">
          {runtime.updatedAt
            ? `Last checked ${runtime.updatedAt.toLocaleTimeString()} · refreshes every 10 seconds`
            : 'Loading runtime observations…'}
        </span>
        <Button
          variant="outline"
          size="sm"
          className="ml-auto"
          onClick={() => void runtime.refresh()}
          disabled={runtime.refreshing}
        >
          <RefreshCw className={cn(runtime.refreshing && 'animate-spin')} />
          Refresh
        </Button>
      </div>
      {runtime.errors.length > 0 ? (
        <div role="alert">
          <Alert tone="warning" title="Runtime observations are unavailable">
            Statistics below stay unknown until the Router reports its runtime again.
            <ul className="mt-1.5 flex flex-col gap-0.5 font-mono text-[11.5px]">
              {runtime.errors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </Alert>
        </div>
      ) : null}
      <DecisionModelRuntimePanel
        inventory={runtime.inventory}
        refreshedAt={runtime.updatedAt}
        loading={runtime.loading}
        initialDeployment={params.get('deployment')}
      />
      <SystemOneAutoMonitoring refreshedAt={runtime.updatedAt} />
      <DecisionTaskMonitoring refreshedAt={runtime.updatedAt} />
    </div>
  )
}
