import { useState } from 'react'
import type { useDecisionModelManagement } from './useDecisionModelManagement'
import { decisionRuntimeDeclarations } from './decisionRuntimeDeployment'
import {
  deploymentPlacements,
  withDecisionReplicas,
  type ReplicaPlacement,
} from './decisionReplicaConfig'
import DecisionReplicaStatus from './DecisionReplicaStatus'
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionReplicaPanel.module.css'
import pageStyles from './DecisionModelPage.module.css'

export default function DecisionReplicaPanel({
  model,
  writable,
}: {
  model: ReturnType<typeof useDecisionModelManagement>
  writable: boolean
}) {
  const [selected, setSelected] = useState('')
  const [draft, setDraft] = useState<{ name: string; placements: ReplicaPlacement[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const deployments = Object.entries(decisionRuntimeDeclarations(model.config)).filter(
    ([, value]) => value.provider === 'model_runtime',
  )
  const selectedEntry = deployments.find(([name]) => name === selected) ?? deployments[0]
  if (!selectedEntry) return null
  const [name, config] = selectedEntry
  const observed = model.inventory?.deployments.find((item) => item.name === name)
  const placements = draft?.name === name ? draft.placements : deploymentPlacements(config)
  const disabled = !writable || model.deploying
  const update = (index: number, value: ReplicaPlacement) =>
    setDraft({ name, placements: placements.map((entry, i) => (i === index ? value : entry)) })
  const save = async () => {
    setError(null)
    setMessage(null)
    try {
      const result = await model.updateConfig((current) =>
        withDecisionReplicas(current, name, placements),
      )
      setMessage(result.message)
      setDraft(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Replica update failed.')
    }
  }
  return (
    <section className={pageStyles.panel} aria-labelledby="replica-pool-title">
      <div className={styles.heading}>
        <div>
          <span className={pageStyles.eyebrow}>Runtime capacity</span>
          <h2 id="replica-pool-title">Model replicas</h2>
          <p className={styles.hint}>
            Each replica serves the same model. Fused question batches stay together on one worker.
          </p>
        </div>
        <SystemOneSelect
          label="Runtime deployment"
          value={name}
          onChange={(value) => {
            setSelected(value)
            setDraft(null)
            setError(null)
            setMessage(null)
          }}
          options={deployments.map(([value, item]) => ({
            value,
            label: value,
            description: item.artifact,
          }))}
        />
      </div>
      <div className={styles.capacity}>
        <div>
          <span>Ready / desired</span>
          <strong>
            {observed?.ready_replicas ?? '—'}{' '}
            <small>/ {observed?.desired_replicas ?? placements.length}</small>
          </strong>
        </div>
        <div>
          <span>Serving state</span>
          <strong>{observed?.state || 'Not observed'}</strong>
        </div>
        <div>
          <span>Model profile</span>
          <strong>{config.profile || 'exact'}</strong>
        </div>
      </div>
      <DecisionReplicaStatus deployment={observed} />
      <details className={styles.editor}>
        <summary>Configure replicas</summary>
        <p className={styles.hint}>
          Use different devices for data parallelism, or repeat a device to measure multiple workers
          on one GPU. Attached runtimes must serve the same artifact and capabilities.
        </p>
        <div className={styles.placements}>
          {placements.map((replica, index) => (
            <div className={styles.placement} key={index}>
              <span className={styles.index}>{index + 1}</span>
              <SystemOneSelect
                label={`Replica ${index + 1} owner`}
                value={replica.endpoint !== undefined ? 'attached' : 'managed'}
                disabled={disabled}
                onChange={(value) =>
                  update(index, value === 'attached' ? { endpoint: '' } : { device: 'auto' })
                }
                options={[
                  { value: 'managed', label: 'Managed' },
                  { value: 'attached', label: 'Attached' },
                ]}
              />
              {replica.endpoint !== undefined ? (
                <>
                  <label>
                    Runtime endpoint
                    <input
                      disabled={disabled}
                      value={replica.endpoint}
                      placeholder="http://runtime:8100"
                      onChange={(event) =>
                        update(index, { ...replica, endpoint: event.target.value })
                      }
                    />
                  </label>
                  <label>
                    Served model name
                    <input
                      disabled={disabled}
                      value={replica.served_name || ''}
                      placeholder="Deployment default"
                      onChange={(event) =>
                        update(index, { ...replica, served_name: event.target.value })
                      }
                    />
                  </label>
                </>
              ) : (
                <label className={styles.device}>
                  Device
                  <input
                    disabled={disabled}
                    value={replica.device || 'auto'}
                    placeholder="rocm:0"
                    onChange={(event) => update(index, { device: event.target.value })}
                  />
                </label>
              )}
              <button
                type="button"
                aria-label={`Remove replica ${index + 1}`}
                disabled={disabled || placements.length === 1}
                onClick={() =>
                  setDraft({ name, placements: placements.filter((_, i) => i !== index) })
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
        <div className={styles.actions}>
          <button
            type="button"
            disabled={disabled || placements.length >= 64}
            onClick={() => setDraft({ name, placements: [...placements, { device: 'auto' }] })}
          >
            Add replica
          </button>
          <button
            type="button"
            className={pageStyles.primary}
            disabled={disabled || draft?.name !== name}
            onClick={() => void save()}
          >
            {model.deploying ? 'Applying…' : 'Apply replicas'}
          </button>
        </div>
      </details>
      {error && (
        <p className={pageStyles.notice} role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className={pageStyles.notice} role="status">
          {message}
        </p>
      )}
    </section>
  )
}
