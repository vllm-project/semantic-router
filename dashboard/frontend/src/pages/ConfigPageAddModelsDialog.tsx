import { useEffect, useId, useMemo, useRef, useState } from 'react'

import { discoverProviderModels, type DiscoveredProviderModel } from '../utils/modelDiscoveryApi'
import ModelProviderLogo from './ModelProviderLogo'
import ConfigPageModelProviderPicker from './ConfigPageModelProviderPicker'
import ConfigPageModelDiscoveryResults from './ConfigPageModelDiscoveryResults'
import {
  normalizedPrefix,
  optionalNumber,
  parseHeaders,
  parseList,
} from './configPageModelImportSupport'
import {
  FEATURED_MODEL_PROVIDERS,
  filterModelProviders,
  getModelProvider,
  type ModelProviderDefinition,
} from './modelProviderCatalog'
import styles from './ConfigPageAddModelsDialog.module.css'

export interface ModelBatchImportInput {
  models: DiscoveredProviderModel[]
  providerId: string
  baseUrl: string
  apiKey: string
  authHeader: string
  authPrefix: string
  runtimeProvider: string
  apiFormat: string
  apiVersion: string
  chatPath: string
  apiKeyEnv: string
  extraHeaders: Record<string, string>
  endpointWeight: number
  reasoningFamily: string
  namePrefix: string
  paramSize: string
  contextWindowSize?: number
  description: string
  modality: string
  capabilities: string[]
  tags: string[]
  loras: string[]
  qualityScore?: number
  pricing?: {
    currency?: string
    prompt_per_1m?: number
    cached_input_per_1m?: number
    cache_write_per_1m?: number
    completion_per_1m?: number
  }
}

interface ConfigPageAddModelsDialogProps {
  isOpen: boolean
  existingModelNames: string[]
  reasoningFamilies: string[]
  onClose: () => void
  onImport: (input: ModelBatchImportInput) => Promise<void>
}

export default function ConfigPageAddModelsDialog({
  isOpen,
  existingModelNames,
  reasoningFamilies,
  onClose,
  onImport,
}: ConfigPageAddModelsDialogProps) {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const busyRef = useRef(false)
  const [stage, setStage] = useState<'provider' | 'connection'>('provider')
  const [providerSearch, setProviderSearch] = useState('')
  const [providerId, setProviderId] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [authHeader, setAuthHeader] = useState('Authorization')
  const [authPrefix, setAuthPrefix] = useState('Bearer')
  const [runtimeProvider, setRuntimeProvider] = useState('openai')
  const [apiVersion, setApiVersion] = useState('')
  const [chatPath, setChatPath] = useState('')
  const [apiKeyEnv, setApiKeyEnv] = useState('')
  const [extraHeadersText, setExtraHeadersText] = useState('')
  const [endpointWeight, setEndpointWeight] = useState('1')
  const [reasoningFamily, setReasoningFamily] = useState(reasoningFamilies[0] ?? '')
  const [namePrefix, setNamePrefix] = useState('')
  const [paramSize, setParamSize] = useState('')
  const [contextWindowSize, setContextWindowSize] = useState('')
  const [description, setDescription] = useState('')
  const [modality, setModality] = useState('')
  const [capabilities, setCapabilities] = useState('')
  const [tags, setTags] = useState('')
  const [loras, setLoras] = useState('')
  const [qualityScore, setQualityScore] = useState('')
  const [currency, setCurrency] = useState('USD')
  const [promptPrice, setPromptPrice] = useState('')
  const [cachedInputPrice, setCachedInputPrice] = useState('')
  const [cacheWritePrice, setCacheWritePrice] = useState('')
  const [completionPrice, setCompletionPrice] = useState('')
  const [models, setModels] = useState<DiscoveredProviderModel[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [search, setSearch] = useState('')
  const [discovering, setDiscovering] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  busyRef.current = discovering || saving

  const selectedProvider = providerId ? getModelProvider(providerId) : null
  const filteredProviders = useMemo(() => filterModelProviders(providerSearch), [providerSearch])
  const catalogProviders = useMemo(() => {
    if (providerSearch.trim()) return filteredProviders
    const featuredIds = new Set(FEATURED_MODEL_PROVIDERS.map((provider) => provider.id))
    return filteredProviders.filter((provider) => !featuredIds.has(provider.id))
  }, [filteredProviders, providerSearch])
  const prefix = normalizedPrefix(namePrefix)
  const existing = useMemo(() => new Set(existingModelNames), [existingModelNames])
  useEffect(() => {
    if (!isOpen) return
    setStage('provider')
    setProviderSearch('')
    setProviderId('')
    setModels([])
    setSelected(new Set())
    setError(null)
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) return
    const previous = document.activeElement as HTMLElement | null
    dialogRef.current?.focus()
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busyRef.current) onClose()
    }
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('keydown', handleKey)
      previous?.focus()
    }
  }, [isOpen, onClose])

  useEffect(
    () => () => {
      abortRef.current?.abort()
    },
    [],
  )

  if (!isOpen) return null

  const chooseProvider = (provider: ModelProviderDefinition) => {
    setProviderId(provider.id)
    setBaseUrl(provider.baseUrl)
    setAuthHeader(provider.authHeader)
    setAuthPrefix(provider.authPrefix)
    setRuntimeProvider(provider.runtimeProvider)
    setChatPath(provider.chatPath || '')
    setExtraHeadersText(provider.extraHeaders ? JSON.stringify(provider.extraHeaders, null, 2) : '')
    setModels([])
    setSelected(new Set())
    setError(null)
    setStage('connection')
  }

  const findModels = async () => {
    if (!baseUrl.trim()) {
      setError('Enter a base URL.')
      return
    }
    if (!selectedProvider?.apiKeyOptional && !apiKey.trim()) {
      setError('Enter your API key.')
      return
    }
    let extraHeaders: Record<string, string>
    try {
      extraHeaders = parseHeaders(extraHeadersText)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Check the extra headers.')
      return
    }
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setDiscovering(true)
    setError(null)
    try {
      const discovered = await discoverProviderModels(
        {
          baseUrl: baseUrl.trim(),
          modelsPath: selectedProvider?.modelsPath,
          apiKey: apiKey.trim(),
          authHeader,
          authPrefix,
          extraHeaders,
        },
        controller.signal,
      )
      setModels(discovered)
      setSearch('')
      const importable = discovered.filter((model) => !existing.has(prefix + model.id))
      setSelected(new Set(importable.length === 1 ? [importable[0].id] : []))
      if (discovered.length === 0) setError('No models were found on this connection.')
    } catch (cause) {
      if (!controller.signal.aborted) {
        setModels([])
        setSelected(new Set())
        setError(cause instanceof Error ? cause.message : 'Models could not be loaded.')
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setDiscovering(false)
      }
    }
  }

  const submit = async () => {
    const chosen = models.filter((model) => selected.has(model.id))
    if (chosen.length === 0) {
      setError('Select at least one model.')
      return
    }
    let extraHeaders: Record<string, string>
    try {
      extraHeaders = parseHeaders(extraHeadersText)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Check the extra headers.')
      return
    }
    const parsedWeight = optionalNumber(endpointWeight)
    if (parsedWeight === undefined || parsedWeight <= 0) {
      setError('Endpoint weight must be greater than zero.')
      return
    }
    const context = optionalNumber(contextWindowSize)
    if (contextWindowSize.trim() && (!context || context < 1)) {
      setError('Context window must be a positive number.')
      return
    }
    const quality = optionalNumber(qualityScore)
    if (qualityScore.trim() && (quality === undefined || quality < 0 || quality > 1)) {
      setError('Quality score must be between 0 and 1.')
      return
    }
    const pricing = {
      currency: currency.trim() || undefined,
      prompt_per_1m: optionalNumber(promptPrice),
      cached_input_per_1m: optionalNumber(cachedInputPrice),
      cache_write_per_1m: optionalNumber(cacheWritePrice),
      completion_per_1m: optionalNumber(completionPrice),
    }
    const hasPricing = Object.entries(pricing).some(
      ([key, value]) => key !== 'currency' && value !== undefined,
    )
    setSaving(true)
    setError(null)
    try {
      await onImport({
        models: chosen,
        providerId,
        baseUrl: baseUrl
          .trim()
          .replace(/\/$/, '')
          .replace(/\/models$/, ''),
        apiKey: apiKey.trim(),
        authHeader: authHeader.trim(),
        authPrefix: authPrefix.trim(),
        runtimeProvider: runtimeProvider.trim() || 'openai',
        apiFormat: runtimeProvider === 'anthropic' ? 'anthropic' : 'openai',
        apiVersion: apiVersion.trim(),
        chatPath: chatPath.trim(),
        apiKeyEnv: apiKeyEnv.trim(),
        extraHeaders,
        endpointWeight: parsedWeight,
        reasoningFamily,
        namePrefix: prefix,
        paramSize: paramSize.trim(),
        contextWindowSize: context,
        description: description.trim(),
        modality: modality.trim(),
        capabilities: parseList(capabilities),
        tags: parseList(tags),
        loras: parseList(loras),
        qualityScore: quality,
        pricing: hasPricing ? pricing : undefined,
      })
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Models could not be imported.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !discovering && !saving) onClose()
      }}
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
          <div className={styles.headerCopy}>
            <span>Models</span>
            <h2 id={titleId}>{stage === 'provider' ? 'Choose a provider' : 'Add models'}</h2>
            <p>
              {stage === 'provider'
                ? 'Start with where your models run.'
                : 'Connect once, then import one model or an entire catalog.'}
            </p>
          </div>
          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
            disabled={discovering || saving}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        {error ? (
          <div className={styles.error} role="alert">
            <span>{error}</span>
            <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
              ×
            </button>
          </div>
        ) : null}

        {stage === 'provider' ? (
          <ConfigPageModelProviderPicker
            search={providerSearch}
            providers={catalogProviders}
            titleId={titleId}
            onSearch={setProviderSearch}
            onChoose={chooseProvider}
          />
        ) : (
          <div className={styles.body}>
            <div className={styles.selectedProviderBar}>
              <ModelProviderLogo provider={providerId} size="large" />
              <div>
                <small>Provider</small>
                <strong>{selectedProvider?.name}</strong>
                <span>{selectedProvider?.description}</span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setError(null)
                  setStage('provider')
                }}
              >
                Change
              </button>
            </div>

            <section className={styles.connectionSection}>
              <label className={`${styles.field} ${styles.baseUrlField}`}>
                <span>Base URL</span>
                <input
                  type="url"
                  value={baseUrl}
                  onChange={(event) => setBaseUrl(event.target.value)}
                  placeholder="http://localhost:8000/v1"
                  readOnly={selectedProvider?.category === 'Model APIs'}
                  aria-readonly={selectedProvider?.category === 'Model APIs'}
                  autoFocus={selectedProvider?.category !== 'Model APIs'}
                />
              </label>
              <label className={styles.field}>
                <span>
                  API key {selectedProvider?.apiKeyOptional ? <small>Optional</small> : null}
                </span>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder={
                    selectedProvider?.apiKeyOptional ? 'If required' : 'Required by provider'
                  }
                  autoComplete="off"
                  autoFocus={selectedProvider?.category === 'Model APIs'}
                />
              </label>
              <button
                type="button"
                className={styles.findButton}
                onClick={() => void findModels()}
                disabled={discovering || saving}
              >
                {discovering ? 'Connecting…' : models.length > 0 ? 'Refresh' : 'Find models'}
              </button>
            </section>

            <ConfigPageModelDiscoveryResults
              models={models}
              selected={selected}
              onSelected={setSelected}
              existing={existing}
              prefix={prefix}
              providerId={providerId}
              search={search}
              onSearch={setSearch}
            />

            <details className={styles.advanced}>
              <summary>
                <span>Advanced</span>
                <small>Routing metadata, connection details, and pricing</small>
              </summary>
              <div className={styles.advancedContent}>
                <section className={styles.advancedSection}>
                  <div className={styles.advancedHeading}>
                    <strong>Model profile</strong>
                    <span>Applied to every model in this import.</span>
                  </div>
                  <div className={styles.advancedGrid}>
                    <label className={styles.field}>
                      <span>Name prefix</span>
                      <input
                        value={namePrefix}
                        onChange={(event) => {
                          setNamePrefix(event.target.value)
                          setSelected(new Set())
                        }}
                        placeholder="Optional, e.g. production"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Reasoning family</span>
                      <select
                        value={reasoningFamily}
                        onChange={(event) => setReasoningFamily(event.target.value)}
                      >
                        <option value="">None</option>
                        {reasoningFamilies.map((family) => (
                          <option key={family} value={family}>
                            {family}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className={styles.field}>
                      <span>Parameter size</span>
                      <input
                        value={paramSize}
                        onChange={(event) => setParamSize(event.target.value)}
                        placeholder="70B"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Context window</span>
                      <input
                        type="number"
                        min="1"
                        value={contextWindowSize}
                        onChange={(event) => setContextWindowSize(event.target.value)}
                        placeholder="131072"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Modality</span>
                      <input
                        value={modality}
                        onChange={(event) => setModality(event.target.value)}
                        placeholder="text, multimodal"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Quality score</span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={qualityScore}
                        onChange={(event) => setQualityScore(event.target.value)}
                        placeholder="0.92"
                      />
                    </label>
                    <label className={`${styles.field} ${styles.fullField}`}>
                      <span>Description</span>
                      <textarea
                        value={description}
                        onChange={(event) => setDescription(event.target.value)}
                        placeholder="What this model is best at"
                        rows={2}
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Capabilities</span>
                      <textarea
                        value={capabilities}
                        onChange={(event) => setCapabilities(event.target.value)}
                        placeholder="reasoning, code, vision"
                        rows={3}
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Tags</span>
                      <textarea
                        value={tags}
                        onChange={(event) => setTags(event.target.value)}
                        placeholder="production, fast"
                        rows={3}
                      />
                    </label>
                    <label className={`${styles.field} ${styles.fullField}`}>
                      <span>LoRA adapters</span>
                      <textarea
                        value={loras}
                        onChange={(event) => setLoras(event.target.value)}
                        placeholder="support-agent, legal-domain"
                        rows={2}
                      />
                    </label>
                  </div>
                </section>

                <section className={styles.advancedSection}>
                  <div className={styles.advancedHeading}>
                    <strong>Connection</strong>
                    <span>Override the provider preset only when your endpoint requires it.</span>
                  </div>
                  <div className={styles.advancedGrid}>
                    <label className={styles.field}>
                      <span>Wire protocol</span>
                      <select
                        value={runtimeProvider}
                        disabled={selectedProvider?.category === 'Model APIs'}
                        onChange={(event) => {
                          const protocol = event.target.value
                          setRuntimeProvider(protocol)
                        }}
                      >
                        <option value="openai">OpenAI compatible</option>
                        <option value="anthropic">Anthropic Messages</option>
                      </select>
                    </label>
                    <label className={styles.field}>
                      <span>Auth header</span>
                      <input
                        value={authHeader}
                        onChange={(event) => setAuthHeader(event.target.value)}
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Auth prefix</span>
                      <input
                        value={authPrefix}
                        onChange={(event) => setAuthPrefix(event.target.value)}
                        placeholder="Bearer"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>API key environment variable</span>
                      <input
                        value={apiKeyEnv}
                        onChange={(event) => setApiKeyEnv(event.target.value)}
                        placeholder="OPENAI_API_KEY"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Endpoint weight</span>
                      <input
                        type="number"
                        min="0.01"
                        step="0.01"
                        value={endpointWeight}
                        onChange={(event) => setEndpointWeight(event.target.value)}
                      />
                    </label>
                    <label className={styles.field}>
                      <span>API version</span>
                      <input
                        value={apiVersion}
                        onChange={(event) => setApiVersion(event.target.value)}
                        placeholder="Optional"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Chat path</span>
                      <input
                        value={chatPath}
                        onChange={(event) => setChatPath(event.target.value)}
                        placeholder="/v1/chat/completions"
                      />
                    </label>
                    <label className={`${styles.field} ${styles.fullField}`}>
                      <span>
                        Extra headers <small>JSON</small>
                      </span>
                      <textarea
                        value={extraHeadersText}
                        onChange={(event) => setExtraHeadersText(event.target.value)}
                        placeholder={'{\n  "x-provider-region": "us-east"\n}'}
                        rows={4}
                        className={styles.codeInput}
                      />
                    </label>
                  </div>
                </section>

                <section className={styles.advancedSection}>
                  <div className={styles.advancedHeading}>
                    <strong>Pricing</strong>
                    <span>Optional cost per one million tokens.</span>
                  </div>
                  <div className={`${styles.advancedGrid} ${styles.pricingGrid}`}>
                    <label className={styles.field}>
                      <span>Currency</span>
                      <input
                        value={currency}
                        onChange={(event) => setCurrency(event.target.value)}
                        placeholder="USD"
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Input</span>
                      <input
                        type="number"
                        min="0"
                        step="0.0001"
                        value={promptPrice}
                        onChange={(event) => setPromptPrice(event.target.value)}
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Cached input</span>
                      <input
                        type="number"
                        min="0"
                        step="0.0001"
                        value={cachedInputPrice}
                        onChange={(event) => setCachedInputPrice(event.target.value)}
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Cache write</span>
                      <input
                        type="number"
                        min="0"
                        step="0.0001"
                        value={cacheWritePrice}
                        onChange={(event) => setCacheWritePrice(event.target.value)}
                      />
                    </label>
                    <label className={styles.field}>
                      <span>Output</span>
                      <input
                        type="number"
                        min="0"
                        step="0.0001"
                        value={completionPrice}
                        onChange={(event) => setCompletionPrice(event.target.value)}
                      />
                    </label>
                  </div>
                </section>
              </div>
            </details>
          </div>
        )}

        <footer className={styles.footer}>
          {stage === 'connection' ? (
            <button
              type="button"
              className={styles.backButton}
              onClick={() => setStage('provider')}
              disabled={discovering || saving}
            >
              Back
            </button>
          ) : (
            <span />
          )}
          <div>
            <button
              type="button"
              className={styles.cancelButton}
              onClick={onClose}
              disabled={discovering || saving}
            >
              Cancel
            </button>
            {stage === 'connection' ? (
              <button
                type="button"
                className={styles.importButton}
                onClick={() => void submit()}
                disabled={saving || selected.size === 0}
              >
                {saving
                  ? 'Importing…'
                  : selected.size > 0
                    ? `Import ${selected.size} model${selected.size === 1 ? '' : 's'}`
                    : 'Import models'}
              </button>
            ) : null}
          </div>
        </footer>
      </div>
    </div>
  )
}
