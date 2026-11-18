import { StatusLabel } from '../ui'
import type { ModelRuntimeDeployment } from './decisionModelManagement'

export default function DecisionReplicaStatus({
  deployment,
}: {
  deployment?: ModelRuntimeDeployment
}) {
  if (!deployment?.replicas?.length)
    return (
      <p className="text-xs text-subtle-foreground">Replica observations are not available yet.</p>
    )
  return (
    <ul className="overflow-hidden rounded-lg border border-border">
      {deployment.replicas.map((replica) => (
        <li
          key={replica.id}
          className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 border-b border-border px-3 py-2.5 last:border-0 sm:grid-cols-[minmax(0,1fr)_7rem_5rem_5rem]"
        >
          <div className="flex min-w-0 flex-col">
            <span className="truncate font-mono text-[12.5px] font-medium">
              {replica.device || (replica.managed ? 'Managed worker' : 'Attached worker')}
            </span>
            <span
              className="truncate font-mono text-[11px] text-subtle-foreground"
              title={replica.id}
            >
              {replica.id}
            </span>
            {replica.reason ? (
              <span className="text-xs text-muted-foreground">{replica.reason}</span>
            ) : null}
          </div>
          <StatusLabel tone={replica.ready ? 'success' : 'warning'}>{replica.state}</StatusLabel>
          <span className="text-xs text-muted-foreground max-sm:hidden">
            In flight <span className="font-mono tabular text-foreground">{replica.inflight}</span>
          </span>
          <span className="text-xs text-muted-foreground max-sm:hidden">
            Restarts <span className="font-mono tabular text-foreground">{replica.restarts}</span>
          </span>
        </li>
      ))}
    </ul>
  )
}
