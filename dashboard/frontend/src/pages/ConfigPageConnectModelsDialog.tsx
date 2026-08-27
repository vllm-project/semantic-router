import { useEffect, useId, useMemo, useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import {
  emptyConnectModelAdvancedValues,
  requestedConnectedModelName,
  resolveConnectedModelName,
  type ConnectModelAdvancedValues,
} from './configPageConnectModelSupport'
import type { ModelPricing, ProviderReliability, RoutingModelCard } from './configPageSupport'
import ConfigPageConnectModelAdvanced from './ConfigPageConnectModelAdvanced'
import ModelProviderLogo from './ModelProviderLogo'
import { modelProviderCatalog, type ModelProviderPreset } from './modelProviderCatalog'
import styles from './ConfigPageConnectModelsDialog.module.css'

export interface ConnectedModelInput {
  provider: ModelProviderPreset
  baseUrl: string
  apiKey: string
  modelIds: string[]
  modelNames: Record<string, string>
  reasoningFamily?: string
  metadata: Omit<RoutingModelCard, 'name'>
  pricing?: ModelPricing
  reliability?: ProviderReliability
}

interface Props {
  isOpen: boolean
  existingModelNames: string[]
  reasoningFamilies: string[]
  onClose: () => void
  onImport: (input: ConnectedModelInput) => Promise<void>
  onManualSetup: () => void
}

interface DiscoveryResponse {
  models?: unknown
  error?: unknown
}

const providerCategories = ['Start here', 'Model APIs', 'Private runtimes'] as const

const listValues = (value: string) => [
  ...new Set(
    value
      .split(',')
      .map((entry) => entry.trim())
      .filter(Boolean),
  ),
]

const optionalNumber = (value: string): number | undefined => {
  if (!value.trim()) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

const advancedInput = (advanced: ConnectModelAdvancedValues) => {
  const costConfigured = [
    advanced.inputCost,
    advanced.outputCost,
    advanced.cacheReadCost,
    advanced.cacheWriteCost,
  ].some((value) => value.trim())
  const inputCost = optionalNumber(advanced.inputCost)
  const pricing: ModelPricing | undefined = costConfigured
    ? {
        currency: 'USD',
        prompt_per_1m: inputCost,
        completion_per_1m: optionalNumber(advanced.outputCost),
        cached_input_per_1m: optionalNumber(advanced.cacheReadCost) ?? inputCost,
        cache_write_per_1m: optionalNumber(advanced.cacheWriteCost) ?? inputCost,
      }
    : undefined
  const reliability: ProviderReliability = {
    retry_count: optionalNumber(advanced.maxRetries),
    retry_on: advanced.retryOn.trim() || undefined,
    lb_policy: advanced.loadBalancing || undefined,
    health_check_path: advanced.healthCheckPath.trim() || undefined,
    health_check_interval: advanced.healthCheckInterval.trim() || undefined,
    health_check_timeout: advanced.healthCheckTimeout.trim() || undefined,
  }
  const cleanObject = <Value extends object>(value: Value): Value | undefined =>
    Object.values(value).some((entry) => entry !== undefined && entry !== '') ? value : undefined

  return {
    namePrefix: advanced.namePrefix,
    reasoningFamily: advanced.reasoningFamily || undefined,
    metadata: {
      description: advanced.description.trim() || undefined,
      modality: advanced.modality || undefined,
      param_size: advanced.parameterSize.trim() || undefined,
      context_window_size: optionalNumber(advanced.contextWindow),
      capabilities: listValues(advanced.capabilities),
      tags: listValues(advanced.tags),
      quality_score: optionalNumber(advanced.qualityScore),
    },
    pricing,
    reliability: cleanObject(reliability),
  }
}

export default function ConfigPageConnectModelsDialog({
  isOpen,
  existingModelNames,
  reasoningFamilies,
  onClose,
  onImport,
  onManualSetup,
}: Props) {
  const titleId = useId()
  const [stage, setStage] = useState<'provider' | 'models'>('provider')
  const [search, setSearch] = useState('')
  const [provider, setProvider] = useState<ModelProviderPreset | null>(null)
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setAPIKey] = useState('')
  const [models, setModels] = useState<string[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [modelSearch, setModelSearch] = useState('')
  const [manualModel, setManualModel] = useState('')
  const [advanced, setAdvanced] = useState(emptyConnectModelAdvancedValues)
  const [discovering, setDiscovering] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const busy = discovering || saving
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen, onClose, dismissible: !busy })

  useEffect(() => {
    if (!isOpen) return
    setStage('provider')
    setSearch('')
    setProvider(null)
    setBaseUrl('')
    setAPIKey('')
    setModels([])
    setSelected(new Set())
    setModelSearch('')
    setManualModel('')
    setAdvanced(emptyConnectModelAdvancedValues())
    setError(null)
  }, [isOpen])

  const visibleProviders = useMemo(() => {
    const query = search.trim().toLocaleLowerCase()
    if (!query) return modelProviderCatalog
    return modelProviderCatalog.filter((item) =>
      `${item.name} ${item.description}`.toLocaleLowerCase().includes(query),
    )
  }, [search])
  const existing = useMemo(() => new Set(existingModelNames), [existingModelNames])
  const visibleModels = useMemo(() => {
    const query = modelSearch.trim().toLocaleLowerCase()
    if (!query) return models
    return models.filter((model) => model.toLocaleLowerCase().includes(query))
  }, [modelSearch, models])
  const resolvedModelNames = useMemo(() => {
    if (!provider) return new Map<string, string>()
    const occupied = new Set(existing)
    const resolved = new Map<string, string>()
    for (const model of models) {
      const name = resolveConnectedModelName(advanced.namePrefix, provider.id, model, occupied)
      resolved.set(model, name)
      occupied.add(name)
    }
    return resolved
  }, [advanced.namePrefix, existing, models, provider])

  if (!isOpen) return null

  const chooseProvider = (next: ModelProviderPreset) => {
    setProvider(next)
    setBaseUrl(next.baseUrl)
    setAPIKey('')
    setModels([])
    setSelected(new Set())
    setModelSearch('')
    setError(null)
    setStage('models')
  }

  const discover = async () => {
    if (!provider || !baseUrl.trim()) {
      setError('Enter the provider base URL.')
      return
    }
    if (provider.baseUrl && provider.authMode !== 'none' && !apiKey.trim()) {
      setError('Enter your API key to connect this provider.')
      return
    }
    setDiscovering(true)
    setError(null)
    try {
      const response = await fetch('/api/models/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          baseUrl: baseUrl.trim(),
          apiKey: apiKey.trim(),
          authMode: provider.authMode,
        }),
      })
      const payload = (await response.json()) as DiscoveryResponse
      if (!response.ok) {
        throw new Error(
          typeof payload.error === 'string' ? payload.error : 'Models could not be loaded.',
        )
      }
      const discovered = Array.isArray(payload.models)
        ? payload.models.filter(
            (model): model is string => typeof model === 'string' && model.trim() !== '',
          )
        : []
      setModels(discovered)
      setSelected(new Set(discovered.length === 1 ? discovered : []))
      if (discovered.length === 0) setError('No models were returned. Add a model ID manually.')
    } catch (cause) {
      setModels([])
      setSelected(new Set())
      setError(cause instanceof Error ? cause.message : 'Models could not be loaded.')
    } finally {
      setDiscovering(false)
    }
  }

  const addManualModel = () => {
    const model = manualModel.trim()
    if (!model) return
    if (!models.includes(model)) setModels((current) => [...current, model].sort())
    setSelected((current) => new Set(current).add(model))
    setManualModel('')
    setError(null)
  }

  const submit = async () => {
    if (!provider || selected.size === 0) {
      setError('Choose at least one model.')
      return
    }
    setSaving(true)
    setError(null)
    try {
      const options = advancedInput(advanced)
      await onImport({
        provider,
        baseUrl: baseUrl.trim(),
        apiKey: apiKey.trim(),
        modelIds: [...selected],
        modelNames: Object.fromEntries(
          [...selected].map((modelId) => [modelId, resolvedModelNames.get(modelId) ?? modelId]),
        ),
        ...options,
      })
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Models could not be added.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <header className={styles.header}>
          <div className={styles.titleGroup}>
            {stage === 'models' && provider ? (
              <button
                type="button"
                className={styles.backButton}
                onClick={() => setStage('provider')}
                disabled={busy}
                aria-label="Choose another provider"
              >
                <ProductIcon name="arrow-left" aria-hidden="true" />
              </button>
            ) : null}
            {stage === 'models' && provider ? (
              <ModelProviderLogo provider={provider} size="medium" />
            ) : null}
            <div>
              <h2 id={titleId}>
                {stage === 'provider' ? 'Add models' : (provider?.name ?? 'Connect provider')}
              </h2>
              <p>
                {stage === 'provider'
                  ? 'Choose where your models run.'
                  : 'Connect once, then import one or many models.'}
              </p>
            </div>
          </div>
          <button
            type="button"
            className={styles.iconButton}
            onClick={onClose}
            disabled={busy}
            aria-label="Close"
          >
            <ProductIcon name="close" />
          </button>
        </header>

        {error ? (
          <div className={styles.error} role="alert">
            {error}
          </div>
        ) : null}

        {stage === 'provider' ? (
          <div className={styles.body}>
            <div className={styles.searchShell}>
              <ProductIcon name="search" aria-hidden="true" />
              <input
                className={styles.search}
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search providers"
                autoFocus
                data-dialog-initial-focus
              />
              <small>{visibleProviders.length} providers</small>
            </div>
            {providerCategories.map((category) => {
              const providers = visibleProviders.filter((item) => item.category === category)
              if (providers.length === 0) return null
              return (
                <section key={category} className={styles.providerSection}>
                  <h3>{category}</h3>
                  <div className={styles.providerGrid}>
                    {providers.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        className={styles.providerCard}
                        onClick={() => chooseProvider(item)}
                      >
                        <ModelProviderLogo provider={item} size="medium" />
                        <span>
                          <strong>{item.name}</strong>
                          <small>{item.description}</small>
                        </span>
                      </button>
                    ))}
                  </div>
                </section>
              )
            })}
          </div>
        ) : provider ? (
          <div className={styles.body}>
            <section className={styles.connectionPanel}>
              <div className={styles.connectionHeading}>
                <div>
                  <strong>Connection</strong>
                  <span>Credentials stay private and are only used for this provider.</span>
                </div>
                <button
                  type="button"
                  className={styles.discoverButton}
                  onClick={() => void discover()}
                  disabled={busy || !baseUrl.trim()}
                >
                  <ProductIcon name={models.length > 0 ? 'refresh' : 'search'} aria-hidden="true" />
                  {discovering ? 'Connecting…' : models.length > 0 ? 'Refresh' : 'List models'}
                </button>
              </div>
              <div className={styles.connectionGrid}>
                {!provider.baseUrl ? (
                  <label className={styles.field}>
                    <span>Base URL</span>
                    <input
                      type="url"
                      value={baseUrl}
                      onChange={(event) => setBaseUrl(event.target.value)}
                      placeholder="https://api.example.com/v1"
                      autoFocus
                    />
                  </label>
                ) : (
                  <div className={`${styles.field} ${styles.fixedEndpoint}`}>
                    <span>API endpoint</span>
                    <div>
                      <ProductIcon name="link" aria-hidden="true" />
                      <code>{provider.baseUrl}</code>
                    </div>
                  </div>
                )}
                {provider.authMode !== 'none' ? (
                  <label className={styles.field}>
                    <span>
                      API key <small>{provider.baseUrl ? 'Required' : 'Optional'}</small>
                    </span>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(event) => setAPIKey(event.target.value)}
                      placeholder="Paste your key"
                      autoComplete="new-password"
                      autoFocus={Boolean(provider.baseUrl)}
                    />
                  </label>
                ) : (
                  <div className={`${styles.field} ${styles.noCredential}`}>
                    <span>Authentication</span>
                    <div>
                      <ProductIcon name="check" aria-hidden="true" />
                      No API key required
                    </div>
                  </div>
                )}
              </div>
            </section>

            <div className={styles.manualRow}>
              <ProductIcon name="plus" aria-hidden="true" />
              <input
                value={manualModel}
                onChange={(event) => setManualModel(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    addManualModel()
                  }
                }}
                placeholder="Or enter a model ID"
              />
              <button type="button" onClick={addManualModel} disabled={!manualModel.trim()}>
                Add
              </button>
            </div>

            {models.length > 0 ? (
              <section className={styles.modelSection}>
                <div className={styles.modelSectionHeader}>
                  <div>
                    <h3>Models</h3>
                    <span>{selected.size} selected</span>
                  </div>
                  <div className={styles.modelActions}>
                    <div className={styles.modelSearch}>
                      <ProductIcon name="search" aria-hidden="true" />
                      <input
                        type="search"
                        value={modelSearch}
                        onChange={(event) => setModelSearch(event.target.value)}
                        placeholder="Filter models"
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        const allSelected = visibleModels.every((model) => selected.has(model))
                        setSelected((current) => {
                          const next = new Set(current)
                          visibleModels.forEach((model) =>
                            allSelected ? next.delete(model) : next.add(model),
                          )
                          return next
                        })
                      }}
                    >
                      <ProductIcon name="check" aria-hidden="true" />
                      Select all
                    </button>
                  </div>
                </div>
                <div className={styles.modelList}>
                  {visibleModels.map((model) => {
                    const logicalName = resolvedModelNames.get(model) ?? model
                    const renamed =
                      logicalName !== requestedConnectedModelName(advanced.namePrefix, model)
                    return (
                      <label key={model} className={styles.modelOption}>
                        <input
                          type="checkbox"
                          checked={selected.has(model)}
                          onChange={() =>
                            setSelected((current) => {
                              const next = new Set(current)
                              if (next.has(model)) next.delete(model)
                              else next.add(model)
                              return next
                            })
                          }
                        />
                        <span>
                          <strong>{logicalName}</strong>
                          {renamed ? <small>Named to avoid a public model conflict</small> : null}
                        </span>
                        <ProductIcon
                          className={styles.modelCheck}
                          name="check"
                          aria-hidden="true"
                        />
                      </label>
                    )
                  })}
                </div>
              </section>
            ) : null}

            <ConfigPageConnectModelAdvanced
              value={advanced}
              reasoningFamilies={reasoningFamilies}
              onChange={setAdvanced}
            />
          </div>
        ) : null}

        <footer className={styles.footer}>
          {stage === 'provider' ? (
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => {
                onClose()
                onManualSetup()
              }}
              disabled={busy}
            >
              <ProductIcon name="settings" aria-hidden="true" />
              Manual setup
            </button>
          ) : (
            <span />
          )}
          <div>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={onClose}
              disabled={busy}
            >
              <ProductIcon name="close" aria-hidden="true" />
              Cancel
            </button>
            {stage === 'models' ? (
              <button
                type="button"
                className={styles.primaryButton}
                onClick={() => void submit()}
                disabled={busy || selected.size === 0}
              >
                <ProductIcon name="plus" aria-hidden="true" />
                {saving
                  ? 'Adding…'
                  : `Add ${selected.size || ''} model${selected.size === 1 ? '' : 's'}`}
              </button>
            ) : null}
          </div>
        </footer>
      </div>
    </div>
  )
}
