import { useState } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import {
  Alert,
  Button,
  Input,
  MiniStat,
  SectionLabel,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  toast,
} from '../ui'
import type { DecisionDeploymentRow } from './decisionDeploymentRows'
import {
  deploymentPlacements,
  withDecisionReplicas,
  type ReplicaPlacement,
} from './decisionReplicaConfig'
import DecisionObservationStatus from './DecisionObservationStatus'
import DecisionReplicaStatus from './DecisionReplicaStatus'
import type { DecisionModelManagement } from './useDecisionModelManagement'

export default function DecisionReplicaPanel({
  model,
  row,
  writable,
}: {
  model: DecisionModelManagement
  row: DecisionDeploymentRow
  writable: boolean
}) {
  const [draft, setDraft] = useState<ReplicaPlacement[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const editable = row.kind === 'model_runtime' && Boolean(row.declaration)
  const placements = draft ?? (row.declaration ? deploymentPlacements(row.declaration) : [])
  const disabled = !writable || model.deploying || !editable || model.observationState.config.stale
  const update = (index: number, value: ReplicaPlacement) =>
    setDraft(placements.map((entry, position) => (position === index ? value : entry)))
  const save = async () => {
    setError(null)
    try {
      const result = await model.updateConfig((current) =>
        withDecisionReplicas(current, row.name, placements),
      )
      setDraft(null)
      toast.success(`Replicas saved for ${row.name}`, { description: result.message })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Replica update failed.')
    }
  }
  return (
    <section className="flex flex-col gap-3" aria-label="Replicas">
      <div className="flex items-end justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <SectionLabel>Replicas</SectionLabel>
          <p className="text-xs text-muted-foreground">
            Each replica serves the same model. Fused question batches stay together on one worker.
          </p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <MiniStat
          label="Ready / desired"
          value={
            <span className="tabular">
              {row.observed?.ready_replicas ?? '—'}
              <span className="text-sm font-medium text-subtle-foreground">
                {' '}
                / {row.observed?.desired_replicas ?? placements.length}
              </span>
            </span>
          }
        />
        <MiniStat label="Serving state" value={row.observed?.state || 'Not observed'} />
        <MiniStat label="Model profile" value={row.observed?.profile || 'Not reported'} />
      </div>
      <DecisionObservationStatus
        label="Replica health"
        observation={model.observationState.inventory}
      />
      <DecisionReplicaStatus deployment={row.observed} />
      {editable ? (
        <div className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface-subtle/50 p-3">
          <div className="flex flex-col gap-0.5">
            <span className="text-[13px] font-medium">Configure replicas</span>
            <span className="text-xs text-muted-foreground">
              Use different devices for data parallelism, or repeat a device to run several workers
              on one GPU. Attached runtimes must serve the same artifact and capabilities.
            </span>
          </div>
          <ol className="flex flex-col gap-2">
            {placements.map((replica, index) => {
              const attached = replica.endpoint !== undefined
              return (
                <li
                  key={index}
                  className="grid grid-cols-[1.5rem_8rem_minmax(0,1fr)_auto] items-center gap-2"
                >
                  <span className="text-center font-mono text-xs text-subtle-foreground tabular">
                    {index + 1}
                  </span>
                  <Select
                    value={attached ? 'attached' : 'managed'}
                    disabled={disabled}
                    onValueChange={(value) =>
                      update(index, value === 'attached' ? { endpoint: '' } : { device: 'auto' })
                    }
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full"
                      aria-label={`Replica ${index + 1} owner`}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="managed">Managed</SelectItem>
                      <SelectItem value="attached">Attached</SelectItem>
                    </SelectContent>
                  </Select>
                  {attached ? (
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <Input
                        aria-label={`Replica ${index + 1} endpoint`}
                        className="h-8 font-mono text-[12.5px]"
                        disabled={disabled}
                        value={replica.endpoint}
                        placeholder="http://runtime:8100"
                        onChange={(event) =>
                          update(index, { ...replica, endpoint: event.target.value })
                        }
                      />
                      <Input
                        aria-label={`Replica ${index + 1} served model`}
                        className="h-8 font-mono text-[12.5px]"
                        disabled={disabled}
                        value={replica.model || ''}
                        placeholder="Deployment default"
                        onChange={(event) =>
                          update(index, { ...replica, model: event.target.value })
                        }
                      />
                    </div>
                  ) : (
                    <Input
                      aria-label={`Replica ${index + 1} device`}
                      className="h-8 font-mono text-[12.5px]"
                      disabled={disabled}
                      value={replica.device || 'auto'}
                      placeholder="rocm:0"
                      onChange={(event) => update(index, { device: event.target.value })}
                    />
                  )}
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Remove replica ${index + 1}`}
                    disabled={disabled || placements.length === 1}
                    onClick={() => setDraft(placements.filter((_, position) => position !== index))}
                  >
                    <Trash2 />
                  </Button>
                </li>
              )
            })}
          </ol>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={disabled || placements.length >= 64}
              onClick={() => setDraft([...placements, { device: 'auto' }])}
            >
              <Plus />
              Add replica
            </Button>
            <Button
              size="sm"
              className="ml-auto"
              disabled={disabled || draft === null}
              loading={model.deploying && draft !== null}
              onClick={() => void save()}
            >
              Apply replicas
            </Button>
          </div>
        </div>
      ) : (
        <p className="text-xs text-subtle-foreground">
          {row.kind === 'systemone'
            ? 'A remote System One API manages its own capacity.'
            : 'Declare this deployment in the configuration to manage its replicas here.'}
        </p>
      )}
      {error ? (
        <div role="alert">
          <Alert tone="danger" title="Replicas were not saved">
            {error}
          </Alert>
        </div>
      ) : null}
    </section>
  )
}
