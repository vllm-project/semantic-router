import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import { DataTable, type Column } from '../components/DataTable'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import TableHeader from '../components/TableHeader'
import type { ViewSection } from '../components/ViewModal'
import {
  routingManagementApi,
  waitForRoutingMutation,
  type FallbackTrigger,
  type RoutingModel,
  type RoutingModelPatch,
} from '../utils/routingManagementApi'
import ConfigPageAddModelsDialog, { type ModelBatchImportInput } from './ConfigPageAddModelsDialog'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ModelProviderLogo from './ModelProviderLogo'
import { buildModelControlOverrides } from './configPageModelOnboardingSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { useProviderCatalogDisplayMap } from './useProviderCatalogDisplayMap'
import styles from './ConfigPage.module.css'
import modelStyles from './ConfigPageModelsSection.module.css'

interface ConfigPageModelsSectionProps {
  isReadonly: boolean
  canVerifyModels: boolean
  modelsSearch: string
  onModelsSearchChange: (value: string) => void
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
}

interface ModelFormState {
  name: string
  aliases: string
  paramSize: string
  contextWindowSize: string
  description: string
  capabilities: string
  reasoningType: string
  reasoningEfforts: string
  loras: string
  qualityScore: string
  modality: string
  tags: string
  maxRetries: number
  retryOn: FallbackTrigger[]
  requestTimeout: string
  streamTimeout: string
  inputCost: string
  outputCost: string
  cacheReadCost: string
  cacheWriteCost: string
}

type ProbeState =
  | { state: 'idle' | 'checking' }
  | { state: 'live'; latencyMilliseconds: number }
  | { state: 'error'; message: string }

const splitList = (value: string): string[] => [
  ...new Set(
    value
      .split(/[\n,]/)
      .map((item) => item.trim())
      .filter(Boolean),
  ),
]

const nullableCost = (value: string): string | null => {
  const normalized = value.trim()
  return normalized || null
}

const optionalWholeNumber = (value: string, label: string, maximum: number): number => {
  const normalized = value.trim()
  if (!normalized) return 0
  const parsed = Number(normalized)
  if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > maximum) {
    throw new Error(`${label} must be a whole number from 0 to ${maximum.toLocaleString()}.`)
  }
  return parsed
}

const optionalQualityScore = (value: string): number => {
  const normalized = value.trim()
  if (!normalized) return 0
  const parsed = Number(normalized)
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error('Quality score must be from 0 to 1.')
  }
  return parsed
}

const price = (value: string | null | undefined): string => value ?? '—'

export default function ConfigPageModelsSection({
  isReadonly,
  canVerifyModels,
  modelsSearch,
  onModelsSearchChange,
  openEditModal,
  openViewModal,
}: ConfigPageModelsSectionProps) {
  const [models, setModels] = useState<RoutingModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [addOpen, setAddOpen] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<RoutingModel | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [probeStates, setProbeStates] = useState<Map<string, ProbeState>>(new Map())
  const autoProbeRevision = useRef('')
  const providerDisplays = useProviderCatalogDisplayMap()

  const loadModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setModels(await routingManagementApi.listModels())
    } catch (cause) {
      setModels([])
      setError(cause instanceof Error ? cause.message : 'Models could not be loaded.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadModels()
  }, [loadModels])

  const verifyModel = useCallback(async (model: RoutingModel) => {
    setProbeStates((current) => new Map(current).set(model.id, { state: 'checking' }))
    try {
      const result = await routingManagementApi.probeModel(model.id)
      setProbeStates((current) =>
        new Map(current).set(
          model.id,
          result.reachable
            ? { state: 'live', latencyMilliseconds: result.latencyMilliseconds }
            : { state: 'error', message: 'Unavailable' },
        ),
      )
    } catch (cause) {
      setProbeStates((current) =>
        new Map(current).set(model.id, {
          state: 'error',
          message: cause instanceof Error ? cause.message : 'Check failed',
        }),
      )
    }
  }, [])

  useEffect(() => {
    if (!canVerifyModels || models.length === 0) return
    const revisionKey = models.map((model) => `${model.id}:${model.revision}`).join('|')
    if (autoProbeRevision.current === revisionKey) return
    autoProbeRevision.current = revisionKey
    void Promise.all(models.slice(0, 10).map(verifyModel))
  }, [canVerifyModels, models, verifyModel])

  const filteredModels = useMemo(() => {
    const query = modelsSearch.trim().toLocaleLowerCase()
    if (!query) return models
    return models.filter((model) =>
      [
        model.name,
        ...model.aliases,
        ...model.capabilities,
        ...model.loras,
        ...model.backends.flatMap((backend) => [backend.providerId, backend.providerModelId]),
      ]
        .join(' ')
        .toLocaleLowerCase()
        .includes(query),
    )
  }, [models, modelsSearch])

  const editModel = (model: RoutingModel) => {
    const form: ModelFormState = {
      name: model.name,
      aliases: model.aliases.join(', '),
      paramSize: model.paramSize ?? '',
      contextWindowSize: model.contextWindowSize ? String(model.contextWindowSize) : '',
      description: model.description ?? '',
      capabilities: model.capabilities.join(', '),
      reasoningType: model.reasoning?.type ?? '',
      reasoningEfforts: model.reasoning?.efforts?.join(', ') ?? '',
      loras: model.loras.join(', '),
      qualityScore: model.qualityScore ? String(model.qualityScore) : '',
      modality: model.modality ?? '',
      tags: model.tags.join(', '),
      maxRetries: model.control.retry.count,
      retryOn: [...model.control.retry.on],
      requestTimeout: model.control.timeout.request,
      streamTimeout: model.control.timeout.stream,
      inputCost: model.pricing.inputCostPerMillionTokens ?? '',
      outputCost: model.pricing.outputCostPerMillionTokens ?? '',
      cacheReadCost: model.pricing.cacheReadCostPerMillionTokens ?? '',
      cacheWriteCost: model.pricing.cacheWriteCostPerMillionTokens ?? '',
    }
    openEditModal<ModelFormState>(
      `Edit ${model.name}`,
      form,
      [
        { name: 'name', label: 'Name', type: 'text', required: true },
        {
          name: 'aliases',
          label: 'Aliases',
          type: 'textarea',
          description: 'Comma-separated names clients can recognize.',
        },
        { name: 'description', label: 'Description', type: 'textarea' },
        {
          name: 'paramSize',
          label: 'Parameter size',
          type: 'text',
          placeholder: '32B',
        },
        {
          name: 'contextWindowSize',
          label: 'Context window',
          type: 'text',
          placeholder: '131072',
        },
        {
          name: 'capabilities',
          label: 'Capabilities',
          type: 'textarea',
          description: 'Examples: tools, vision, reasoning.',
        },
        { name: 'reasoningType', label: 'Reasoning family', type: 'text' },
        {
          name: 'reasoningEfforts',
          label: 'Reasoning efforts',
          type: 'text',
          placeholder: 'low, medium, high',
        },
        { name: 'loras', label: 'LoRA adapters', type: 'textarea' },
        {
          name: 'qualityScore',
          label: 'Quality score',
          type: 'text',
          placeholder: '0.90',
          description: 'Optional value from 0 to 1.',
        },
        { name: 'modality', label: 'Modality', type: 'text', placeholder: 'text' },
        {
          name: 'tags',
          label: 'Tags',
          type: 'textarea',
          description: 'Comma-separated labels.',
        },
        { name: 'maxRetries', label: 'Max retries', type: 'number', min: 0, max: 5 },
        {
          name: 'retryOn',
          label: 'Retry on',
          type: 'multiselect',
          options: ['unavailable', 'timeout'],
          description: 'Only retries attempts proven safe to repeat.',
        },
        {
          name: 'requestTimeout',
          label: 'Request timeout',
          type: 'text',
          placeholder: '300s',
        },
        {
          name: 'streamTimeout',
          label: 'Stream timeout',
          type: 'text',
          placeholder: '900s',
        },
        {
          name: 'inputCost',
          label: 'Input cost / 1M tokens',
          type: 'text',
          placeholder: '0.50',
        },
        {
          name: 'outputCost',
          label: 'Output cost / 1M tokens',
          type: 'text',
          placeholder: '1.50',
        },
        {
          name: 'cacheReadCost',
          label: 'Cache read cost / 1M tokens',
          type: 'text',
          description: 'Uses the input price when left blank.',
        },
        {
          name: 'cacheWriteCost',
          label: 'Cache write cost / 1M tokens',
          type: 'text',
          description: 'Uses the input price when left blank.',
        },
      ],
      async (next) => {
        const name = next.name.trim()
        if (!name) throw new Error('Name is required.')
        const control = buildModelControlOverrides({
          maxRetries: String(next.maxRetries),
          retryOn: next.retryOn,
          requestTimeout: next.requestTimeout,
          streamTimeout: next.streamTimeout,
        })
        if (!control?.retry || !control.timeout?.request || !control.timeout.stream) {
          throw new Error('Retry count and both timeouts are required.')
        }
        const patch: RoutingModelPatch = {
          name,
          aliases: splitList(next.aliases),
          paramSize: next.paramSize.trim(),
          contextWindowSize: optionalWholeNumber(
            next.contextWindowSize,
            'Context window',
            100_000_000,
          ),
          description: next.description.trim(),
          capabilities: splitList(next.capabilities),
          reasoning: {
            ...(next.reasoningType.trim() ? { type: next.reasoningType.trim() } : {}),
            ...(splitList(next.reasoningEfforts).length
              ? { efforts: splitList(next.reasoningEfforts) }
              : {}),
          },
          loras: splitList(next.loras),
          qualityScore: optionalQualityScore(next.qualityScore),
          modality: next.modality.trim(),
          tags: splitList(next.tags),
          control: {
            retry: {
              count: control.retry.count ?? 0,
              on: control.retry.on ?? [],
            },
            timeout: {
              request: control.timeout.request,
              stream: control.timeout.stream,
            },
          },
          pricing: {
            inputCostPerMillionTokens: nullableCost(next.inputCost),
            outputCostPerMillionTokens: nullableCost(next.outputCost),
            cacheReadCostPerMillionTokens: nullableCost(next.cacheReadCost),
            cacheWriteCostPerMillionTokens: nullableCost(next.cacheWriteCost),
          },
        }
        await waitForRoutingMutation(
          await routingManagementApi.updateModel(model.id, model.revision, patch),
        )
        await loadModels()
      },
    )
  }

  const viewModel = (model: RoutingModel) => {
    const sections: ViewSection[] = [
      {
        title: 'Model',
        fields: [
          { label: 'Name', value: model.name },
          { label: 'Status', value: model.status === 'active' ? 'Live' : 'Draft' },
          { label: 'Aliases', value: model.aliases.join(', ') || '—', fullWidth: true },
          { label: 'Description', value: model.description || '—', fullWidth: true },
          { label: 'Parameter size', value: model.paramSize || '—' },
          {
            label: 'Context window',
            value: model.contextWindowSize?.toLocaleString() || '—',
          },
          {
            label: 'Capabilities',
            value: model.capabilities.join(', ') || '—',
            fullWidth: true,
          },
          { label: 'Reasoning family', value: model.reasoning?.type || '—' },
          { label: 'LoRA adapters', value: model.loras.join(', ') || '—' },
          { label: 'Quality score', value: model.qualityScore ?? '—' },
          { label: 'Modality', value: model.modality || '—' },
          { label: 'Tags', value: model.tags.join(', ') || '—', fullWidth: true },
        ],
      },
      {
        title: 'Control',
        fields: [
          { label: 'Max retries', value: model.control.retry.count },
          { label: 'Retry on', value: model.control.retry.on.join(', ') || 'Never' },
          { label: 'Request timeout', value: model.control.timeout.request },
          { label: 'Stream timeout', value: model.control.timeout.stream },
        ],
      },
      {
        title: 'Pricing / 1M tokens',
        fields: [
          { label: 'Input', value: price(model.pricing.inputCostPerMillionTokens) },
          { label: 'Output', value: price(model.pricing.outputCostPerMillionTokens) },
          { label: 'Cache read', value: price(model.pricing.cacheReadCostPerMillionTokens) },
          { label: 'Cache write', value: price(model.pricing.cacheWriteCostPerMillionTokens) },
        ],
      },
      {
        title: 'Backends',
        fields: model.backends.map((backend, index) => ({
          label: backend.providerId,
          value: `${backend.providerModelId} · weight ${backend.weight}${backend.credentialConfigured ? ' · credential ready' : ''}`,
          fullWidth: index === model.backends.length - 1,
        })),
      },
    ]
    openViewModal(
      model.name,
      sections,
      isReadonly ? undefined : () => editModel(model),
      isReadonly
        ? []
        : [
            {
              label: 'Delete model',
              tone: 'destructive',
              onClick: () => setPendingDelete(model),
            },
          ],
    )
  }

  const columns: Column<RoutingModel>[] = [
    {
      key: 'name',
      header: 'Model',
      width: '340px',
      sortable: true,
      render: (model) => (
        <div className={modelStyles.modelIdentityWithLogo}>
          <ModelProviderLogo
            icon={providerDisplays.get(model.backends[0]?.providerId || '')?.icon}
            name={
              providerDisplays.get(model.backends[0]?.providerId || '')?.name ||
              model.backends[0]?.providerId ||
              'Model'
            }
            monogram={providerDisplays.get(model.backends[0]?.providerId || '')?.monogram}
            accent={providerDisplays.get(model.backends[0]?.providerId || '')?.accent}
            size="small"
          />
          <div className={modelStyles.modelIdentity}>
            <span className={modelStyles.modelName}>{model.name}</span>
            <span className={modelStyles.modelPhysicalId}>
              {model.backends[0]?.providerModelId || model.id}
            </span>
          </div>
        </div>
      ),
    },
    {
      key: 'capabilities',
      header: 'Capabilities',
      width: '220px',
      render: (model) => model.capabilities.slice(0, 3).join(' · ') || '—',
    },
    {
      key: 'backends',
      header: 'Backends',
      width: '110px',
      align: 'center',
      render: (model) => model.backends.length,
    },
    ...(canVerifyModels
      ? [
          {
            key: 'live',
            header: 'Live',
            width: '130px',
            render: (model: RoutingModel) => {
              const state = probeStates.get(model.id) ?? { state: 'idle' as const }
              return (
                <button
                  type="button"
                  className={modelStyles.liveVerificationButton}
                  disabled={state.state === 'checking'}
                  onClick={(event) => {
                    event.stopPropagation()
                    void verifyModel(model)
                  }}
                >
                  <ProductIcon name={state.state === 'live' ? 'check' : 'refresh'} />
                  {state.state === 'checking'
                    ? 'Checking'
                    : state.state === 'live'
                      ? `Live · ${state.latencyMilliseconds} ms`
                      : state.state === 'error'
                        ? 'Retry'
                        : 'Check'}
                </button>
              )
            },
          } satisfies Column<RoutingModel>,
        ]
      : []),
    {
      key: 'control',
      header: 'Control',
      width: '170px',
      render: (model) => `${model.control.timeout.request} · ${model.control.retry.count} retries`,
    },
    {
      key: 'pricing',
      header: 'Input / Output',
      width: '180px',
      render: (model) =>
        `${price(model.pricing.inputCostPerMillionTokens)} / ${price(model.pricing.outputCostPerMillionTokens)}`,
    },
  ]

  const importModels = async (input: ModelBatchImportInput) => {
    const receipt = await routingManagementApi.bulkImportModels(input)
    await waitForRoutingMutation(receipt)
    await loadModels()
  }

  const deleteModel = async () => {
    if (!pendingDelete) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await routingManagementApi.deleteModel(pendingDelete.id, pendingDelete.revision)
      setPendingDelete(null)
      await loadModels()
    } catch (cause) {
      setDeleteError(cause instanceof Error ? cause.message : 'Model could not be deleted.')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <ConfigPageManagerLayout title="Models" description="Connect once. Route anywhere.">
      <div className={styles.sectionCard}>
        <TableHeader
          title="Models"
          count={models.length}
          searchPlaceholder="Search models, providers, or capabilities"
          searchValue={modelsSearch}
          onSearchChange={onModelsSearchChange}
          onSecondaryAction={() => void loadModels()}
          secondaryActionText={loading ? 'Loading' : 'Refresh'}
          onAdd={isReadonly ? undefined : () => setAddOpen(true)}
          addButtonText="Add model"
          variant="embedded"
        />
        {error ? (
          <div className={modelStyles.operationError} role="alert">
            <span>{error}</span>
            <button type="button" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        ) : null}
        {loading && models.length === 0 ? (
          <ProductLoadingState compact label="Loading models" />
        ) : (
          <DataTable
            columns={columns}
            data={filteredModels}
            keyExtractor={(model) => model.id}
            onView={viewModel}
            openOnRowClick
            emptyMessage="No models yet"
            className={styles.managerTable}
            readonly={isReadonly}
            pagination={{
              pageSize: 25,
              pageSizeOptions: [25, 50, 100],
              itemLabel: 'models',
              resetKey: modelsSearch,
            }}
          />
        )}
      </div>

      <ConfigPageAddModelsDialog
        isOpen={addOpen}
        existingModelNames={models.map((model) => model.name)}
        onClose={() => setAddOpen(false)}
        onImport={importModels}
      />
      <ConfirmDialog
        isOpen={pendingDelete !== null}
        title={`Delete “${pendingDelete?.name ?? ''}”?`}
        description="This model must be removed from every Mixture-of-Models first."
        eyebrow="Delete model"
        confirmLabel="Delete model"
        pending={deleting}
        details={deleteError ? <div role="alert">{deleteError}</div> : undefined}
        onCancel={() => {
          if (deleting) return
          setPendingDelete(null)
          setDeleteError(null)
        }}
        onConfirm={() => void deleteModel()}
      />
    </ConfigPageManagerLayout>
  )
}
