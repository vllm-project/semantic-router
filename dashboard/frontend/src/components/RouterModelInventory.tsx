import React from 'react'
import styles from './RouterModelInventory.module.css'
import { useReadonly } from '../contexts/ReadonlyContext'
import {
  getLoadedModelCount,
  getPreviewRouterModels,
  getRouterModelAnchor,
  getRouterModelState,
  getRouterModelStateLabel,
  getTotalKnownModelCount,
  sortRouterModels,
  type RouterModelInfo,
  type RouterModelsInfo,
} from '../utils/routerRuntime'

interface RouterModelInventoryProps {
  modelsInfo?: RouterModelsInfo | null
  mode?: 'preview' | 'full'
  previewLimit?: number
  emptyMessage?: string
  onSelectModel?: (model: RouterModelInfo) => void
}

interface DetailRow {
  label: string
  value: string
}

interface DetailSection {
  title: string
  rows: DetailRow[]
}

const TITLE_WORD_OVERRIDES: Record<string, string> = {
  amd: 'AMD',
  lora: 'LoRA',
  mmbert: 'mmBERT',
  nli: 'NLI',
  pii: 'PII',
}

const METADATA_PRIORITY: Record<string, number> = {
  model_type: 0,
  threshold: 1,
  use_cpu: 2,
  enabled: 3,
  mapping_path: 4,
  jailbreak_mapping_path: 5,
  max_sequence_length: 6,
  default_dimension: 7,
  matryoshka_supported: 8,
  min_span_length: 9,
  min_span_confidence: 10,
  context_window_size: 11,
  nli_filtering_enabled: 12,
  status: 13,
}

function formatLabel(value?: string): string {
  if (!value) return 'Unknown'
  return value.replace(/[_-]+/g, ' ')
}

function formatTitleLabel(value?: string): string {
  return formatLabel(value)
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => TITLE_WORD_OVERRIDES[word.toLowerCase()] ?? `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(' ')
}

function extractRuntimePath(value?: string): string {
  if (!value) return ''
  const match = value.match(/path=([^,)]+)/)
  return match?.[1]?.trim() ?? ''
}

function summarizeValues(values?: string[], noun = 'items'): string | undefined {
  if (!values?.length) return undefined
  if (values.length <= 3) return values.join(', ')
  return `${values.length} ${noun}`
}

function getDisplayModelID(model: RouterModelInfo): string {
  return model.registry?.local_path
    || model.resolved_model_path
    || extractRuntimePath(model.model_path)
    || model.model_path
    || model.name
}

function getModelTitle(model: RouterModelInfo): string {
  return formatTitleLabel(model.name)
}

function getModelKind(model: RouterModelInfo): string {
  if (model.registry?.purpose) {
    return formatTitleLabel(model.registry.purpose)
  }
  return formatTitleLabel(model.type)
}

function getModelDescription(model: RouterModelInfo): string | undefined {
  const description = model.registry?.description?.trim()
  return description || undefined
}

function getStateChipClass(model: RouterModelInfo): string {
  switch (getRouterModelState(model)) {
    case 'ready':
      return styles.stateReady
    case 'downloading':
      return styles.stateDownloading
    case 'pending':
    case 'initializing':
      return styles.stateWarm
    default:
      return styles.stateMuted
  }
}

function getCardToneClass(model: RouterModelInfo): string {
  switch (getRouterModelState(model)) {
    case 'ready':
      return styles.cardReady
    case 'downloading':
    case 'pending':
    case 'initializing':
      return styles.cardWarm
    default:
      return styles.cardMuted
  }
}

function formatMetadataValue(key: string, value: string): string {
  if (key === 'model_type') {
    return formatTitleLabel(value)
  }
  if (value === 'true') {
    return 'Yes'
  }
  if (value === 'false') {
    return 'No'
  }
  return value
}

function buildMetadataRows(metadata?: Record<string, string>): DetailRow[] {
  if (!metadata) return []

  return Object.entries(metadata)
    .sort(([left], [right]) => {
      const leftRank = METADATA_PRIORITY[left] ?? 99
      const rightRank = METADATA_PRIORITY[right] ?? 99
      if (leftRank !== rightRank) {
        return leftRank - rightRank
      }
      return left.localeCompare(right)
    })
    .map(([key, value]) => ({
      label: formatTitleLabel(key),
      value: formatMetadataValue(key, value),
    }))
}

function getModelChips(model: RouterModelInfo): string[] {
  const chips = new Set<string>()

  if (model.registry?.pipeline_tag) {
    chips.add(formatLabel(model.registry.pipeline_tag))
  }

  for (const tag of model.registry?.tags ?? []) {
    chips.add(formatLabel(tag))
    if (chips.size >= 5) break
  }

  if (model.categories?.length) {
    chips.add(`${model.categories.length} ${model.categories.length === 1 ? 'category' : 'categories'}`)
  }

  return [...chips].slice(0, 6)
}

function buildDetailSections(model: RouterModelInfo): DetailSection[] {
  const identityRows: DetailRow[] = [
    { label: 'Router Key', value: model.name },
    { label: 'Model ID', value: getDisplayModelID(model) },
  ]

  if (model.registry?.repo_id) {
    identityRows.push({ label: 'Repository', value: model.registry.repo_id })
  }

  if (model.registry?.base_model) {
    identityRows.push({ label: 'Base Model', value: model.registry.base_model })
  }

  const capabilityRows: DetailRow[] = [
    { label: 'Type', value: getModelKind(model) },
  ]

  if (model.registry?.parameter_size) {
    capabilityRows.push({ label: 'Size', value: model.registry.parameter_size })
  }

  if (model.registry?.max_context_length) {
    capabilityRows.push({
      label: 'Context',
      value: `${model.registry.max_context_length.toLocaleString()} tokens`,
    })
  }

  if (model.registry?.base_model_max_context) {
    capabilityRows.push({
      label: 'Base Context',
      value: `${model.registry.base_model_max_context.toLocaleString()} tokens`,
    })
  }

  if (model.registry?.embedding_dim) {
    capabilityRows.push({
      label: 'Embedding',
      value: `${model.registry.embedding_dim}d`,
    })
  }

  if (model.registry?.num_classes) {
    capabilityRows.push({
      label: 'Labels',
      value: `${model.registry.num_classes}`,
    })
  }

  if (model.registry?.license) {
    capabilityRows.push({ label: 'License', value: model.registry.license })
  }

  const languages = summarizeValues(model.registry?.languages, 'languages')
  if (languages) {
    capabilityRows.push({ label: 'Languages', value: languages })
  }

  const datasets = summarizeValues(model.registry?.datasets, 'datasets')
  if (datasets) {
    capabilityRows.push({ label: 'Datasets', value: datasets })
  }

  const runtimeRows = buildMetadataRows(model.metadata)
  if (model.load_time) {
    runtimeRows.unshift({ label: 'Load Time', value: model.load_time })
  }
  if (model.memory_usage) {
    runtimeRows.push({ label: 'Memory Usage', value: model.memory_usage })
  }

  return [
    { title: 'Identity', rows: identityRows },
    { title: 'Capabilities', rows: capabilityRows },
    { title: 'Runtime & Config', rows: runtimeRows },
  ].filter((section) => section.rows.length > 0)
}

function renderDetailRows(rows: DetailRow[]): JSX.Element {
  return (
    <dl className={styles.detailList}>
      {rows.map((row) => (
        <div key={`${row.label}-${row.value}`} className={styles.detailRow}>
          <dt className={styles.detailLabel}>{row.label}</dt>
          <dd className={styles.detailValue}>{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}

const PlatformMark: React.FC<{ compact?: boolean }> = ({ compact = false }) => (
  <img
    src="/amd-logo.png"
    alt="AMD platform"
    className={`${styles.platformLogo} ${compact ? styles.platformLogoCompact : styles.platformLogoLarge}`}
  />
)

const PreviewCardBody: React.FC<{
  model: RouterModelInfo
  isAmdPlatform: boolean
}> = ({ model, isAmdPlatform }) => (
  <>
    <div className={styles.previewTopRow}>
      <span className={styles.previewPurpose}>{getModelKind(model)}</span>
      <span className={`${styles.stateChip} ${getStateChipClass(model)}`}>
        {getRouterModelStateLabel(model)}
      </span>
    </div>

    <p className={styles.previewModelId}>{getDisplayModelID(model)}</p>

    {isAmdPlatform && (
      <div className={styles.platformFooterCompact}>
        <PlatformMark compact />
      </div>
    )}
  </>
)

const FullCardBody: React.FC<{
  model: RouterModelInfo
  isAmdPlatform: boolean
}> = ({ model, isAmdPlatform }) => {
  const sections = buildDetailSections(model)
  const chips = getModelChips(model)
  const description = getModelDescription(model)
  const hasFooter = Boolean(model.registry?.model_card_url) || isAmdPlatform

  return (
    <>
      <div className={styles.titleBlock}>
        <div className={styles.titleRow}>
          <h3 className={styles.modelName}>{getModelTitle(model)}</h3>
          <span className={`${styles.stateChip} ${getStateChipClass(model)}`}>
            {getRouterModelStateLabel(model)}
          </span>
        </div>
        <p className={styles.modelSubtitle}>{getModelKind(model)}</p>
        {description && <p className={styles.description}>{description}</p>}
      </div>

      {chips.length > 0 && (
        <div className={styles.chips}>
          {chips.map((chip) => (
            <span key={chip} className={styles.chip}>{chip}</span>
          ))}
        </div>
      )}

      <div className={styles.detailSections}>
        {sections.map((section) => (
          <section key={section.title} className={styles.detailSection}>
            <span className={styles.detailSectionTitle}>{section.title}</span>
            {renderDetailRows(section.rows)}
          </section>
        ))}
      </div>

      {hasFooter && (
        <div className={`${styles.detailFooter} ${!model.registry?.model_card_url ? styles.detailFooterEnd : ''}`}>
          {model.registry?.model_card_url ? (
            <a
              className={styles.cardLink}
              href={model.registry.model_card_url}
              target="_blank"
              rel="noreferrer"
            >
              Open model card
            </a>
          ) : null}

          {isAmdPlatform && <PlatformMark />}
        </div>
      )}
    </>
  )
}

const RouterModelInventory: React.FC<RouterModelInventoryProps> = ({
  modelsInfo,
  mode = 'full',
  previewLimit,
  emptyMessage = 'No router model metadata is available yet.',
  onSelectModel,
}) => {
  const { platform } = useReadonly()
  const isAmdPlatform = platform?.toLowerCase() === 'amd'
  const models = mode === 'preview'
    ? getPreviewRouterModels(modelsInfo, previewLimit)
    : sortRouterModels(modelsInfo?.models ?? [])

  const loadedCount = getLoadedModelCount(modelsInfo)
  const totalCount = getTotalKnownModelCount(modelsInfo)
  const phase = modelsInfo?.summary?.phase
  const summaryMessage = modelsInfo?.summary?.message

  if (models.length === 0) {
    return <div className={styles.empty}>{emptyMessage}</div>
  }

  return (
    <div className={styles.inventory}>
      {mode === 'full' && (
        <div className={styles.summaryRow}>
          <div className={styles.summaryStat}>
            <span className={styles.summaryLabel}>Loaded</span>
            <span className={styles.summaryValue}>{loadedCount}/{totalCount}</span>
          </div>
          {phase && (
            <div className={styles.summaryStat}>
              <span className={styles.summaryLabel}>Phase</span>
              <span className={styles.summaryValue}>{formatTitleLabel(phase)}</span>
            </div>
          )}
          {summaryMessage && <p className={styles.summaryMessage}>{summaryMessage}</p>}
        </div>
      )}

      <div
        className={mode === 'preview' ? styles.previewGrid : styles.fullGrid}
        data-testid={`router-model-grid-${mode}`}
      >
        {models.map((model) => {
          const className = [
            styles.card,
            mode === 'preview' ? styles.previewCard : styles.detailCard,
            getCardToneClass(model),
            onSelectModel && mode === 'preview' ? styles.cardInteractive : '',
          ].filter(Boolean).join(' ')

          const cardContent = mode === 'preview'
            ? <PreviewCardBody model={model} isAmdPlatform={isAmdPlatform} />
            : <FullCardBody model={model} isAmdPlatform={isAmdPlatform} />

          if (onSelectModel && mode === 'preview') {
            return (
              <button
                key={model.name}
                type="button"
                className={className}
                data-testid={`router-model-${mode}-${model.name}`}
                onClick={() => onSelectModel(model)}
              >
                {cardContent}
              </button>
            )
          }

          return (
            <article
              key={model.name}
              id={mode === 'full' ? getRouterModelAnchor(model) : undefined}
              className={className}
              data-testid={`router-model-${mode}-${model.name}`}
            >
              {cardContent}
            </article>
          )
        })}
      </div>
    </div>
  )
}

export default RouterModelInventory
