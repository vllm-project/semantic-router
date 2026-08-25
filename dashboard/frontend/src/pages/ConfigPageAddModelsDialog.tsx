import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import {
  discoverProviderModels,
  getProviderCatalogDetail,
  listProviderCatalog,
  type DiscoveredProviderModel,
  type ManagementPageInfo,
  type ProviderCatalogItem,
  type ProviderConnectionValue,
} from '../utils/providerCatalogApi'
import { createProviderCredential } from '../utils/providerCredentialApi'
import type { RoutingBulkImportRequest } from '../utils/routingManagementApi'
import { normalizedPrefix } from './configPageModelImportSupport'
import ConfigPageModelAdvancedOptions from './ConfigPageModelAdvancedOptions'
import ConfigPageModelDiscoveryResults from './ConfigPageModelDiscoveryResults'
import ConfigPageModelProviderPicker from './ConfigPageModelProviderPicker'
import ConfigPageProviderConnectionField from './ConfigPageProviderConnectionField'
import {
  buildModelControlOverrides,
  buildModelPricingOverrides,
  buildRoutingBulkImportRequest,
  initialProviderConnectionFields,
  initialProviderFieldValue,
  validatedProviderConnectionFields,
  type ControlFormValues,
  type EditableConnectionValue,
} from './configPageModelOnboardingSupport'
import ModelProviderLogo from './ModelProviderLogo'
import styles from './ConfigPageAddModelsDialog.module.css'

export type ModelBatchImportInput = RoutingBulkImportRequest

interface ConfigPageAddModelsDialogProps {
  isOpen: boolean
  existingModelNames: string[]
  onClose: () => void
  onImport: (input: ModelBatchImportInput) => Promise<void>
}

const emptyPage: ManagementPageInfo = { hasMore: false, pageSize: 50 }

export default function ConfigPageAddModelsDialog({
  isOpen,
  existingModelNames,
  onClose,
  onImport,
}: ConfigPageAddModelsDialogProps) {
  const titleId = useId()
  const providerAbortRef = useRef<AbortController | null>(null)
  const actionAbortRef = useRef<AbortController | null>(null)
  const busyRef = useRef(false)
  const [stage, setStage] = useState<'provider' | 'connection'>('provider')
  const [providerSearch, setProviderSearch] = useState('')
  const [providers, setProviders] = useState<ProviderCatalogItem[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [providerPage, setProviderPage] = useState<ManagementPageInfo>(emptyPage)
  const [providerLoading, setProviderLoading] = useState(false)
  const [provider, setProvider] = useState<ProviderCatalogItem | null>(null)
  const [catalogRevision, setCatalogRevision] = useState('')
  const [interfaceId, setInterfaceId] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [credentialName, setCredentialName] = useState('')
  const [secret, setSecret] = useState('')
  const [credentialId, setCredentialId] = useState('')
  const [connectionFields, setConnectionFields] = useState<Record<string, EditableConnectionValue>>(
    {},
  )
  const [models, setModels] = useState<DiscoveredProviderModel[]>([])
  const [modelPage, setModelPage] = useState<ManagementPageInfo>(emptyPage)
  const [discoveryClaim, setDiscoveryClaim] = useState('')
  const [discoveryExpiresAt, setDiscoveryExpiresAt] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [modelSearch, setModelSearch] = useState('')
  const [discoveryAttempted, setDiscoveryAttempted] = useState(false)
  const [discovering, setDiscovering] = useState(false)
  const [saving, setSaving] = useState(false)
  const [namePrefix, setNamePrefix] = useState('')
  const [maxRetries, setMaxRetries] = useState('')
  const [retryOn, setRetryOn] = useState<ControlFormValues['retryOn']>([])
  const [requestTimeout, setRequestTimeout] = useState('')
  const [streamTimeout, setStreamTimeout] = useState('')
  const [inputCost, setInputCost] = useState('')
  const [outputCost, setOutputCost] = useState('')
  const [cacheReadCost, setCacheReadCost] = useState('')
  const [cacheWriteCost, setCacheWriteCost] = useState('')
  const [error, setError] = useState<string | null>(null)
  busyRef.current = providerLoading || discovering || saving
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen,
    onClose,
    dismissible: !busyRef.current,
  })

  const prefix = normalizedPrefix(namePrefix)
  const existing = useMemo(() => new Set(existingModelNames), [existingModelNames])
  const primaryFields = provider?.connectionFields.filter((field) => !field.advanced) ?? []
  const advancedFields = provider?.connectionFields.filter((field) => field.advanced) ?? []

  const loadProviders = useCallback(
    async (cursor = '', append = false) => {
      providerAbortRef.current?.abort()
      const controller = new AbortController()
      providerAbortRef.current = controller
      setProviderLoading(true)
      setError(null)
      try {
        const page = await listProviderCatalog(
          { search: providerSearch, cursor: cursor || undefined, pageSize: 50 },
          controller.signal,
        )
        setProviders((current) => {
          if (!append) return page.data
          const next = new Map(current.map((item) => [item.providerId, item]))
          page.data.forEach((item) => next.set(item.providerId, item))
          return [...next.values()]
        })
        setCategories(page.categories)
        setProviderPage(page.page)
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : 'Providers could not be loaded.')
          if (!append) setProviders([])
        }
      } finally {
        if (providerAbortRef.current === controller) {
          providerAbortRef.current = null
          setProviderLoading(false)
        }
      }
    },
    [providerSearch],
  )

  useEffect(() => {
    if (!isOpen || stage !== 'provider') return
    const timer = window.setTimeout(() => void loadProviders(), 200)
    return () => window.clearTimeout(timer)
  }, [isOpen, loadProviders, stage])

  useEffect(() => {
    if (!isOpen) return
    setStage('provider')
    setProviderSearch('')
    setProvider(null)
    setInterfaceId('')
    setModels([])
    setSelected(new Set())
    setSecret('')
    setCredentialId('')
    setError(null)
  }, [isOpen])

  useEffect(
    () => () => {
      providerAbortRef.current?.abort()
      actionAbortRef.current?.abort()
    },
    [],
  )

  if (!isOpen) return null

  const invalidateDiscovery = () => {
    setCredentialId('')
    setModels([])
    setModelPage(emptyPage)
    setDiscoveryClaim('')
    setDiscoveryExpiresAt('')
    setSelected(new Set())
    setDiscoveryAttempted(false)
  }

  const chooseProvider = async (item: ProviderCatalogItem) => {
    actionAbortRef.current?.abort()
    const controller = new AbortController()
    actionAbortRef.current = controller
    setProviderLoading(true)
    setError(null)
    try {
      const detail = await getProviderCatalogDetail(item.providerId, controller.signal)
      if (!detail.data.discoverySupported) {
        throw new Error(`${detail.data.display.name} does not support model discovery.`)
      }
      setProvider(detail.data)
      setCatalogRevision(detail.catalogRevision)
      setInterfaceId(
        detail.data.interfaces.find((providerInterface) => providerInterface.default)?.id ??
          detail.data.interfaces[0]?.id ??
          '',
      )
      setBaseUrl(detail.data.origin.defaultUrl ?? '')
      setCredentialName(`${detail.data.display.name} connection`)
      setSecret('')
      setCredentialId('')
      setConnectionFields(initialProviderConnectionFields(detail.data))
      setModels([])
      setModelPage(emptyPage)
      setDiscoveryClaim('')
      setDiscoveryExpiresAt('')
      setSelected(new Set())
      setModelSearch('')
      setDiscoveryAttempted(false)
      setStage('connection')
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'Provider details could not be loaded.')
      }
    } finally {
      if (actionAbortRef.current === controller) actionAbortRef.current = null
      setProviderLoading(false)
    }
  }

  const discover = async (cursor = '', pageTurn = false) => {
    if (!provider) return
    if (provider.origin.baseUrlRequired && !baseUrl.trim()) {
      setError(`${provider.origin.label || 'Base URL'} is required.`)
      return
    }
    if (provider.credential.mode === 'required' && !secret.trim() && !credentialId) {
      setError(`${provider.credential.label || 'API key'} is required.`)
      return
    }
    let typedFields: Record<string, ProviderConnectionValue>
    try {
      typedFields = validatedProviderConnectionFields(provider, connectionFields)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Check the connection fields.')
      return
    }

    actionAbortRef.current?.abort()
    const controller = new AbortController()
    actionAbortRef.current = controller
    setDiscovering(true)
    setError(null)
    try {
      let resolvedCredentialId = credentialId
      if (provider.credential.mode !== 'none' && secret.trim() && !resolvedCredentialId) {
        if (!credentialName.trim()) throw new Error('Connection name is required.')
        const created = await createProviderCredential(
          {
            providerId: provider.providerId,
            catalogRevision,
            name: credentialName.trim(),
            secret: secret.trim(),
            baseUrl: provider.origin.mode === 'user_supplied' ? baseUrl.trim() : undefined,
            connectionFields: typedFields,
          },
          controller.signal,
        )
        resolvedCredentialId = created.data.id
        setCredentialId(resolvedCredentialId)
      }
      const result = await discoverProviderModels(
        provider.providerId,
        {
          credentialId: resolvedCredentialId || undefined,
          baseUrl: provider.origin.mode === 'user_supplied' ? baseUrl.trim() : undefined,
          connectionFields: Object.keys(typedFields).length > 0 ? typedFields : undefined,
          search: modelSearch.trim() || undefined,
          pageSize: 50,
          cursor: cursor || undefined,
        },
        controller.signal,
      )
      if (result.catalogRevision !== catalogRevision) {
        throw new Error('Provider catalog changed. Choose the provider again to continue.')
      }
      // A discovery claim signs exactly the items returned on one page. Page
      // turns therefore replace the selection instead of mixing item IDs from
      // claims that cannot be verified together.
      setModels(result.data)
      setModelPage(result.page)
      setDiscoveryClaim(result.discoveryRevision)
      setDiscoveryExpiresAt(result.expiresAt)
      setDiscoveryAttempted(true)
      const importable = result.data.filter(
        (model) => !existing.has(prefix + model.providerModelId),
      )
      setSelected(new Set(importable.length === 1 ? [importable[0].catalogItemId] : []))
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'Models could not be loaded.')
        if (!pageTurn) {
          setModels([])
          setSelected(new Set())
          setDiscoveryClaim('')
          setDiscoveryExpiresAt('')
          setDiscoveryAttempted(true)
        }
      }
    } finally {
      if (actionAbortRef.current === controller) actionAbortRef.current = null
      setDiscovering(false)
    }
  }

  const submit = async () => {
    if (!discoveryClaim || selected.size === 0) {
      setError('Select at least one model.')
      return
    }
    if (Date.parse(discoveryExpiresAt) <= Date.now()) {
      setError('This model list expired. Refresh it before importing.')
      return
    }
    try {
      if (!provider) throw new Error('Choose a provider.')
      const typedFields = validatedProviderConnectionFields(provider, connectionFields)
      const control = buildModelControlOverrides({
        maxRetries,
        retryOn,
        requestTimeout,
        streamTimeout,
      })
      const pricing = buildModelPricingOverrides({
        inputCost,
        outputCost,
        cacheReadCost,
        cacheWriteCost,
      })
      setSaving(true)
      setError(null)
      await onImport(
        buildRoutingBulkImportRequest({
          provider,
          interfaceId,
          catalogRevision,
          discoveryClaim,
          credentialId,
          baseUrl,
          connectionFields: typedFields,
          models,
          selectedCatalogItemIds: selected,
          namePrefix,
          control,
          pricing,
        }),
      )
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Models could not be imported.')
    } finally {
      setSaving(false)
    }
  }

  const selectedProviderLogo = provider ? (
    <ModelProviderLogo
      icon={provider.display.icon}
      name={provider.display.name}
      monogram={provider.display.monogram}
      accent={provider.display.accent}
      size="large"
    />
  ) : null

  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busyRef.current) onClose()
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-busy={busyRef.current}
        tabIndex={-1}
      >
        <header className={styles.header}>
          <div className={styles.headerCopy}>
            <span>Models</span>
            <h2 id={titleId}>{stage === 'provider' ? 'Choose a provider' : 'Add models'}</h2>
            <p>
              {stage === 'provider'
                ? 'Choose where your models run.'
                : 'Connect, choose models, and import.'}
            </p>
          </div>
          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
            disabled={busyRef.current}
            aria-label="Close"
          >
            <ProductIcon name="close" />
          </button>
        </header>

        {error ? (
          <div className={styles.error} role="alert">
            <span>{error}</span>
            <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
              <ProductIcon name="close" />
            </button>
          </div>
        ) : null}

        {stage === 'provider' ? (
          <ConfigPageModelProviderPicker
            search={providerSearch}
            providers={providers}
            categories={categories}
            loading={providerLoading}
            hasMore={providerPage.hasMore}
            titleId={titleId}
            onSearch={setProviderSearch}
            onChoose={(item) => void chooseProvider(item)}
            onLoadMore={() => void loadProviders(providerPage.nextCursor, true)}
          />
        ) : provider ? (
          <div className={styles.body}>
            <div className={styles.selectedProviderBar}>
              {selectedProviderLogo}
              <div>
                <small>Provider</small>
                <strong>{provider.display.name}</strong>
                <span>{provider.display.description}</span>
              </div>
              <button type="button" onClick={() => setStage('provider')} disabled={busyRef.current}>
                <ProductIcon name="refresh" />
                Change
              </button>
            </div>

            <section className={styles.connectionSection}>
              {provider.origin.mode === 'user_supplied' ? (
                <label className={`${styles.field} ${styles.baseUrlField}`}>
                  <span>{provider.origin.label || 'Base URL'}</span>
                  <input
                    type="url"
                    value={baseUrl}
                    onChange={(event) => {
                      setBaseUrl(event.target.value)
                      invalidateDiscovery()
                    }}
                    placeholder={provider.origin.hint}
                    autoFocus
                  />
                </label>
              ) : null}
              {provider.credential.mode !== 'none' ? (
                <>
                  <label className={styles.field}>
                    <span>Connection name</span>
                    <input
                      value={credentialName}
                      onChange={(event) => {
                        setCredentialName(event.target.value)
                        invalidateDiscovery()
                      }}
                      placeholder={`${provider.display.name} connection`}
                    />
                  </label>
                  <label className={styles.field}>
                    <span>
                      {provider.credential.label || 'API key'}{' '}
                      {provider.credential.mode === 'optional' ? <small>Optional</small> : null}
                    </span>
                    <input
                      type="password"
                      value={secret}
                      onChange={(event) => {
                        setSecret(event.target.value)
                        invalidateDiscovery()
                      }}
                      placeholder={provider.credential.hint || 'Paste your key'}
                      autoComplete="new-password"
                    />
                  </label>
                </>
              ) : null}
              {primaryFields.map((field) => (
                <ConfigPageProviderConnectionField
                  key={field.name}
                  field={field}
                  value={connectionFields[field.name] ?? initialProviderFieldValue(field)}
                  onChange={(value) => {
                    setConnectionFields((current) => ({ ...current, [field.name]: value }))
                    invalidateDiscovery()
                  }}
                />
              ))}
              <button
                type="button"
                className={styles.findButton}
                onClick={() => void discover()}
                disabled={discovering || saving}
              >
                <ProductIcon name={discoveryAttempted ? 'refresh' : 'search'} />
                {discovering
                  ? 'Connecting…'
                  : discoveryAttempted
                    ? 'Refresh models'
                    : 'Find models'}
              </button>
            </section>

            {discoveryAttempted ? (
              <ConfigPageModelDiscoveryResults
                models={models}
                provider={provider}
                selected={selected}
                onSelected={setSelected}
                existing={existing}
                prefix={prefix}
                search={modelSearch}
                loading={discovering}
                hasMore={modelPage.hasMore}
                onSearch={setModelSearch}
                onSubmitSearch={() => void discover()}
                onLoadMore={() => void discover(modelPage.nextCursor, true)}
              />
            ) : null}

            <ConfigPageModelAdvancedOptions
              interfaces={provider.interfaces}
              interfaceId={interfaceId}
              onInterfaceId={setInterfaceId}
              connectionFields={advancedFields}
              connectionValues={connectionFields}
              onConnectionValue={(name, value) => {
                setConnectionFields((current) => ({ ...current, [name]: value }))
                invalidateDiscovery()
              }}
              namePrefix={namePrefix}
              onNamePrefix={setNamePrefix}
              control={{ maxRetries, retryOn, requestTimeout, streamTimeout }}
              onControl={(next) => {
                setMaxRetries(next.maxRetries)
                setRetryOn(next.retryOn)
                setRequestTimeout(next.requestTimeout)
                setStreamTimeout(next.streamTimeout)
              }}
              pricing={{ inputCost, outputCost, cacheReadCost, cacheWriteCost }}
              onPricing={(field, value) => {
                if (field === 'inputCost') setInputCost(value)
                if (field === 'outputCost') setOutputCost(value)
                if (field === 'cacheReadCost') setCacheReadCost(value)
                if (field === 'cacheWriteCost') setCacheWriteCost(value)
              }}
            />
          </div>
        ) : null}

        <footer className={styles.footer}>
          {stage === 'connection' ? (
            <button
              type="button"
              className={styles.backButton}
              onClick={() => setStage('provider')}
              disabled={busyRef.current}
            >
              <ProductIcon name="chevron-left" />
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
              disabled={busyRef.current}
            >
              <ProductIcon name="close" />
              Cancel
            </button>
            {stage === 'connection' ? (
              <button
                type="button"
                className={styles.importButton}
                onClick={() => void submit()}
                disabled={saving || discovering || selected.size === 0}
              >
                <ProductIcon name="plus" />
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
