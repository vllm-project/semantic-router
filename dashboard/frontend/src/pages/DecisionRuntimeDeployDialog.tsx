import { useState } from 'react'
import { Link } from 'react-router-dom'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type { RouterConfig } from './dashboardPageTypes'
import type { DecisionModelApplyResult } from './decisionModelManagement'
import {
  DECISION_PROVIDERS,
  runtimeDeploymentName,
  runtimeFamilyLabel,
  type DecisionRuntimeCatalogEntry,
} from './decisionRuntimeCatalog'
import {
  compatibleRuntimeConsumers,
  consumerLabel,
  decisionRuntimeConsumers,
  decisionRuntimeDeclarations,
  type DecisionRuntimeDeploymentRequest,
} from './decisionRuntimeDeployment'
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionModelCatalog.module.css'

interface Props {
  entry: DecisionRuntimeCatalogEntry
  existingName?: string
  config: RouterConfig | null
  writable: boolean
  onDeploy: (request: DecisionRuntimeDeploymentRequest) => Promise<DecisionModelApplyResult>
  onClose: () => void
}

export default function DecisionRuntimeDeployDialog({
  entry,
  existingName,
  config,
  writable,
  onDeploy,
  onClose,
}: Props) {
  const existing = existingName ? decisionRuntimeDeclarations(config)[existingName] : undefined
  const [name, setName] = useState(existingName ?? runtimeDeploymentName(entry))
  const [device, setDevice] = useState(existing?.device ?? 'auto')
  const [consumerId, setConsumerId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DecisionModelApplyResult | null>(null)
  const consumers = compatibleRuntimeConsumers(config)
  const bindings = decisionRuntimeConsumers(config).filter(
    (consumer) => consumer.deployment === existingName,
  )
  const selected = consumers.find((consumer) => consumer.id === consumerId)
  const willActivate = Boolean(selected || bindings.length)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !busy,
  })
  const provider = DECISION_PROVIDERS[entry.provider]
  const deviceOptions = [
    {
      value: 'auto',
      label: 'Automatic',
      description: 'Let the runtime select an available device.',
    },
    { value: 'cpu', label: 'CPU' },
    { value: 'cuda:0', label: 'NVIDIA GPU · cuda:0' },
    { value: 'rocm:0', label: 'AMD GPU · rocm:0' },
  ]
  if (!deviceOptions.some((option) => option.value === device))
    deviceOptions.push({ value: device, label: device })
  const submit = async () => {
    if (!writable || busy || !name.trim()) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const applied = await onDeploy({ entry, name, device, consumerId, existingName })
      setResult({
        ...applied,
        message: willActivate
          ? applied.message
          : 'Configuration saved without a consumer binding. This model will start only after a question or decision selector uses it.',
      })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The deployment request failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={styles.backdrop}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="decision-runtime-dialog-title"
        tabIndex={-1}
      >
        <header className={styles.dialogHeader}>
          <div className={styles.identity}>
            {provider && <img className={styles.logo} src={provider.logo} alt={provider.name} />}
            <div>
              <span className={styles.eyebrow}>{runtimeFamilyLabel(entry)} · Custom decisions</span>
              <h2 id="decision-runtime-dialog-title">
                {existingName ? 'Manage' : 'Configure'} {entry.name}
              </h2>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            aria-label="Close deployment dialog"
          >
            ×
          </button>
        </header>
        <div className={styles.dialogBody}>
          <p className={styles.help}>
            Bind this runtime to a custom question or a decision selector. Built-in routing signals
            keep their current Vela model.
          </p>
          <div className={styles.formGrid}>
            <label className={styles.field}>
              Deployment name
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                disabled={busy || !writable || Boolean(existingName)}
                data-dialog-initial-focus
              />
            </label>
            <div className={styles.field}>
              <SystemOneSelect
                label="Runtime device"
                value={device}
                options={deviceOptions}
                onChange={setDevice}
                disabled={busy || !writable}
              />
            </div>
          </div>
          <div className={styles.field}>
            <SystemOneSelect
              label="Deployment consumer"
              value={consumerId}
              onChange={setConsumerId}
              disabled={busy || !writable}
              options={[
                {
                  value: '',
                  label: bindings.length ? 'Keep existing bindings' : 'Save configuration only',
                  description: bindings.length
                    ? `${bindings.length} existing binding${bindings.length === 1 ? '' : 's'} retained.`
                    : 'No model process starts until a consumer is bound.',
                },
                ...consumers.map((consumer) => ({
                  value: consumer.id,
                  label: consumerLabel(consumer),
                  description: `${consumer.kind === 'question' ? consumer.questionType + ' question' : 'Decision selector'} · ${consumer.deployment || 'follows router default'}`,
                })),
              ]}
            />
          </div>
          {selected && (
            <div className={styles.bindingPreview}>
              <strong>{consumerLabel(selected)}</strong>
              <p>
                {selected.deployment || 'Router default'} → {name || 'New deployment'}
              </p>
              <small>
                Only this {selected.kind} binding changes. Other questions and recipes are
                preserved.
              </small>
            </div>
          )}
          {bindings.length > 0 && (
            <div className={styles.bindingPreview}>
              <strong>Current consumers</strong>
              <ul>
                {bindings.map((consumer) => (
                  <li key={consumer.id}>
                    {consumerLabel(consumer)} · {consumer.questionType}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {!consumers.length && (
            <div className={styles.bindingPreview}>
              <strong>No compatible consumers yet</strong>
              <p>
                Create a choice, score or noul decision question, or a decision selector, then
                return here to bind it. Decision 1.0 and 2.0 do not support span or set.
              </p>
              <Link to="/config/signals" target="_blank" rel="noopener noreferrer">
                Open Signals to create a question ↗
              </Link>
              <p className={styles.help}>
                This dialog stays open with your selected model and deployment name.
              </p>
            </div>
          )}
          <dl className={styles.packageInfo}>
            <div>
              <dt>Artifact</dt>
              <dd>{entry.id}</dd>
            </div>
            <div>
              <dt>Revision</dt>
              <dd>{existing ? existing.revision || 'Runtime default revision' : entry.revision}</dd>
            </div>
            <div>
              <dt>Hardware floor</dt>
              <dd>{entry.minMemoryGiB} GiB device memory · runtime registry minimum</dd>
            </div>
          </dl>
          <p className={styles.help}>
            The runtime validates device support and available resources during activation. Saving
            configuration does not confirm that weights are loaded.
          </p>
          {!writable && (
            <p className={styles.feedback}>
              Configuration write access is required to save or deploy this model.
            </p>
          )}
          {error && (
            <div role="alert" className={styles.feedback}>
              <strong>Configuration request failed</strong>
              <p>{error}</p>
            </div>
          )}
          {result && (
            <div role="status" className={styles.feedback}>
              <strong>
                {!willActivate
                  ? 'Saved without activation'
                  : result.status === 'restart_required'
                    ? 'Restart required'
                    : result.status === 'persisted'
                      ? 'Saved; rollout required'
                      : 'Configuration applied'}
              </strong>
              <p>{result.message}</p>
              {willActivate && (
                <Link to="/decision-model/monitoring">Check runtime readiness →</Link>
              )}
            </div>
          )}
        </div>
        <footer className={styles.dialogFooter}>
          <span className={styles.help}>
            {willActivate
              ? 'Saves and requests runtime activation.'
              : 'Saves a declaration; no new runtime starts.'}
          </span>
          <button
            type="button"
            className={styles.primary}
            disabled={!writable || busy || !name.trim() || Boolean(result)}
            onClick={() => void submit()}
          >
            {busy ? 'Applying…' : willActivate ? 'Deploy and bind model' : 'Save configuration'}
          </button>
        </footer>
      </div>
    </div>
  )
}
