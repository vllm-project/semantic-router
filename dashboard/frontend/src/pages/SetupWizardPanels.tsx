import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { motion } from 'framer-motion'
import {
  Check,
  CircleCheck,
  CloudDownload,
  KeyRound,
  Layers,
  Plus,
  PlugZap,
  Search,
  Sparkles,
  Star,
  Trash2,
} from 'lucide-react'
import ConfirmDialog from '../components/ConfirmDialog'
import {
  Alert,
  Badge,
  Button,
  Card,
  Field,
  IconTile,
  Input,
  InputGroup,
  MiniStat,
  ProviderLogo,
  Skeleton,
  Tooltip,
  cn,
} from '../ui'
import SetupWizardPresetChecklist from './SetupWizardPresetChecklist'
import {
  DEFAULT_REMOTE_SETUP_CONFIG_URL,
  SETUP_MODELS_PER_PAGE,
  SETUP_STEP_LABELS,
  discoverSetupModels,
  filterSetupModels,
  getModelDraftFieldErrors,
  paginateSetupModels,
  type ImportedSetupConfig,
  type ModelDraft,
  type PresetCatalogState,
  type PresetDelta,
  type PresetInfo,
  type PresetRequestState,
  type RemoteImportState,
  type SetupConfigCounts,
  type SetupRoutingMode,
  type SetupStep,
} from './setupWizardSupport'
import { SETUP_PROVIDER_OPTIONS, type ProviderKind } from './setupWizardProviderCatalog'

const PROVIDER_LOGO: Record<ProviderKind, string> = {
  vllm: 'vllm',
  ollama: 'ollama',
  'openai-compatible': 'openai',
  anthropic: 'anthropic',
}

interface SetupWizardStepRailProps {
  currentStep: SetupStep
  finished: boolean
  summaries: string[]
  onGoToStep: (step: SetupStep) => boolean
}

export function SetupWizardStepRail({
  currentStep,
  finished,
  summaries,
  onGoToStep,
}: SetupWizardStepRailProps) {
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([])
  const focusStep = (step: number) => buttonRefs.current[step]?.focus()

  const handleStepKeyDown = (event: KeyboardEvent<HTMLButtonElement>, stepIndex: number) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key))
      return
    event.preventDefault()
    if (event.key === 'Home') focusStep(0)
    else if (event.key === 'End') focusStep(SETUP_STEP_LABELS.length - 1)
    else {
      const direction = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1 : -1
      focusStep((stepIndex + direction + SETUP_STEP_LABELS.length) % SETUP_STEP_LABELS.length)
    }
  }

  const activateStep = (step: SetupStep) => {
    if (!onGoToStep(step)) window.requestAnimationFrame(() => focusStep(currentStep))
  }

  return (
    <nav aria-label="Setup progress">
      <ol className="flex flex-col gap-0.5">
        {SETUP_STEP_LABELS.map(([index, label], stepIndex) => {
          const numericStep = stepIndex as SetupStep
          const isActive = currentStep === numericStep && !finished
          const isDone = finished || currentStep > numericStep
          return (
            <li key={label} className="relative">
              {stepIndex < SETUP_STEP_LABELS.length - 1 ? (
                <span
                  aria-hidden
                  className="absolute left-[23px] top-[42px] h-[calc(100%-30px)] w-px bg-border"
                />
              ) : null}
              <button
                ref={(element) => {
                  buttonRefs.current[stepIndex] = element
                }}
                id={`setup-step-${numericStep}-button`}
                type="button"
                disabled={finished}
                aria-current={isActive ? 'step' : undefined}
                aria-controls={`setup-step-${numericStep}-panel`}
                aria-label={`Step ${index}: ${label}${isDone ? ', complete' : ''}`}
                tabIndex={currentStep === numericStep ? 0 : -1}
                onKeyDown={(event) => handleStepKeyDown(event, stepIndex)}
                onClick={() => activateStep(numericStep)}
                className="relative flex w-full items-start gap-3 rounded-lg px-2.5 py-2.5 text-left outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-default enabled:hover:bg-accent/50"
              >
                {isActive ? (
                  <motion.span
                    layoutId="setup-rail-active"
                    className="absolute inset-0 rounded-lg bg-surface shadow-xs ring-1 ring-border dark:bg-accent"
                    transition={{ type: 'spring', bounce: 0.15, duration: 0.45 }}
                  />
                ) : null}
                <span
                  className={cn(
                    'relative grid size-7 shrink-0 place-items-center rounded-full border text-xs font-semibold tabular transition-colors duration-300',
                    isDone && 'border-transparent bg-success text-white',
                    isActive && 'border-primary bg-primary-soft text-primary-soft-foreground',
                    !isDone && !isActive && 'border-border-strong bg-surface text-muted-foreground',
                  )}
                >
                  {isDone ? <Check className="size-3.5" strokeWidth={3} /> : index}
                </span>
                <span className="relative flex min-w-0 flex-col pt-0.5">
                  <span
                    className={cn(
                      'text-[13px] font-medium',
                      !isDone && !isActive && 'text-muted-foreground',
                    )}
                  >
                    {label}
                  </span>
                  <span className="truncate text-xs text-subtle-foreground">
                    {summaries[stepIndex]}
                  </span>
                </span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

interface ModelStepPanelProps {
  models: ModelDraft[]
  defaultModelId: string
  shouldShowStepOneIssues: boolean
  stepOneErrors: string[]
  stepOneAttempted: boolean
  draftBuildError: string | null
  removedModel: ModelDraft | null
  onAddModel: () => void
  onUpdateModel: (id: string, field: keyof ModelDraft, value: string) => void
  onRemoveModel: (id: string) => void
  onUndoRemoveModel: () => void
  onSelectDefaultModel: (id: string) => void
}

type ConnectionTest =
  | { state: 'idle' }
  | { state: 'testing' }
  | { state: 'ok'; models: string[] }
  | { state: 'error'; message: string }

function ModelCard({
  model,
  index,
  isDefault,
  canRemove,
  nameError,
  baseUrlError,
  onUpdate,
  onRemove,
  onMakeDefault,
}: {
  model: ModelDraft
  index: number
  isDefault: boolean
  canRemove: boolean
  nameError?: string
  baseUrlError?: string
  onUpdate: (field: keyof ModelDraft, value: string) => void
  onRemove: () => void
  onMakeDefault: () => void
}) {
  const [test, setTest] = useState<ConnectionTest>({ state: 'idle' })
  const provider = SETUP_PROVIDER_OPTIONS.find((option) => option.id === model.providerKind)
  const title = model.name.trim() || 'New model'
  const update = (field: keyof ModelDraft, value: string) => {
    if (test.state !== 'idle' && test.state !== 'testing') setTest({ state: 'idle' })
    onUpdate(field, value)
  }

  const runTest = async () => {
    setTest({ state: 'testing' })
    try {
      setTest({ state: 'ok', models: await discoverSetupModels(model) })
    } catch (error) {
      setTest({
        state: 'error',
        message: error instanceof Error ? error.message : 'The endpoint could not be reached.',
      })
    }
  }

  return (
    <article className="rounded-xl border border-border bg-card shadow-xs">
      <header className="flex items-center gap-3 border-b border-border px-4 py-3">
        <ProviderLogo provider={PROVIDER_LOGO[model.providerKind]} size={18} framed />
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
            Model {index + 1}
          </span>
          <span
            className={cn(
              'truncate font-mono text-[13px] font-medium',
              !model.name.trim() && 'text-muted-foreground',
            )}
          >
            {title}
          </span>
        </div>
        {isDefault ? (
          <Badge tone="brand">
            <Star />
            Default
          </Badge>
        ) : (
          <Button variant="ghost" size="xs" onClick={onMakeDefault} aria-pressed={false}>
            Make default
          </Button>
        )}
        <Tooltip content={canRemove ? 'Remove' : 'Setup needs at least one model'}>
          <span className="inline-flex">
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={`Remove model ${model.name.trim() || index + 1}`}
              onClick={onRemove}
              disabled={!canRemove}
              className="hover:text-destructive-foreground"
            >
              <Trash2 />
            </Button>
          </span>
        </Tooltip>
      </header>
      <div className="@container flex flex-col gap-4 px-4 py-4">
        <div
          role="radiogroup"
          aria-label={`Provider for model ${index + 1}`}
          className="grid grid-cols-2 gap-2 @[42rem]:grid-cols-4"
        >
          {SETUP_PROVIDER_OPTIONS.map((option) => {
            const active = option.id === model.providerKind
            return (
              <button
                key={option.id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => update('providerKind', option.id)}
                className={cn(
                  'flex items-center gap-2 rounded-lg border bg-card px-2.5 py-2 text-left text-[12.5px] font-medium outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/40',
                  active
                    ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
                    : 'border-border hover:border-border-strong',
                )}
              >
                <ProviderLogo provider={PROVIDER_LOGO[option.id]} size={16} />
                <span className="truncate">{option.label}</span>
              </button>
            )
          })}
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Field
            label="Model name"
            htmlFor={`${model.id}-name`}
            error={nameError}
            hint={!nameError ? 'The provider’s model ID; apps can send it as model.' : undefined}
          >
            <Input
              id={`${model.id}-name`}
              value={model.name}
              onChange={(event) => update('name', event.target.value)}
              placeholder="qwen/qwen3.5-rocm"
              spellCheck={false}
              aria-invalid={Boolean(nameError)}
              className="font-mono text-[12.5px]"
            />
          </Field>
          <Field
            label="Base URL or host"
            htmlFor={`${model.id}-base-url`}
            error={baseUrlError}
            hint={
              !baseUrlError
                ? `${provider?.description ?? ''} A URL like ${provider?.placeholder ?? ''} or a host such as localhost:8000/v1.`
                : undefined
            }
          >
            <Input
              id={`${model.id}-base-url`}
              value={model.baseUrl}
              onChange={(event) => update('baseUrl', event.target.value)}
              placeholder={provider?.placeholder}
              spellCheck={false}
              aria-invalid={Boolean(baseUrlError)}
              className="font-mono text-[12.5px]"
            />
          </Field>
          <Field label="Endpoint label" htmlFor={`${model.id}-endpoint`} optional>
            <Input
              id={`${model.id}-endpoint`}
              value={model.endpointName}
              onChange={(event) => update('endpointName', event.target.value)}
              placeholder="primary"
              spellCheck={false}
              className="font-mono text-[12.5px]"
            />
          </Field>
          <Field label="Access key" htmlFor={`${model.id}-key`} optional>
            <InputGroup
              id={`${model.id}-key`}
              type="password"
              value={model.accessKey}
              onChange={(event) => update('accessKey', event.target.value)}
              placeholder="API key, when the provider needs one"
              autoComplete="off"
              leading={<KeyRound />}
              className="pr-3"
            />
          </Field>
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void runTest()}
              loading={test.state === 'testing'}
              disabled={!model.baseUrl.trim() || Boolean(baseUrlError)}
            >
              {test.state === 'testing' ? null : <PlugZap />}
              Test connection
            </Button>
            {test.state === 'ok' ? (
              <span
                className="flex items-center gap-1.5 text-xs text-success-foreground"
                role="status"
              >
                <CircleCheck className="size-3.5" />
                Reachable · {test.models.length} {test.models.length === 1 ? 'model' : 'models'}{' '}
                listed
              </span>
            ) : test.state === 'idle' ? (
              <span className="text-xs text-muted-foreground">
                Lists the models the endpoint serves. Nothing is saved.
              </span>
            ) : null}
          </div>
          {test.state === 'error' ? (
            <Alert tone="warning" title="The endpoint did not answer">
              {test.message} You can still continue; the router checks the backend again once it
              runs.
            </Alert>
          ) : null}
          {test.state === 'ok' && test.models.length ? (
            <div className="flex flex-wrap gap-1.5">
              {test.models.slice(0, 12).map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => update('name', name)}
                  className={cn(
                    'inline-flex h-6 items-center rounded-md border px-2 font-mono text-[11.5px] outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                    name === model.name.trim()
                      ? 'border-primary bg-primary-soft text-primary-soft-foreground'
                      : 'border-border bg-surface-subtle hover:border-border-strong',
                  )}
                >
                  {name}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </article>
  )
}

export function ModelStepPanel({
  models,
  defaultModelId,
  shouldShowStepOneIssues,
  stepOneErrors,
  stepOneAttempted,
  draftBuildError,
  removedModel,
  onAddModel,
  onUpdateModel,
  onRemoveModel,
  onUndoRemoveModel,
  onSelectDefaultModel,
}: ModelStepPanelProps) {
  const [modelQuery, setModelQuery] = useState('')
  const [modelPageNumber, setModelPageNumber] = useState(1)
  const [pendingRemoveModel, setPendingRemoveModel] = useState<ModelDraft | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)
  const validationFocusHandledRef = useRef(false)
  const fieldErrorsByModelId = getModelDraftFieldErrors(models)
  const filteredModels = filterSetupModels(models, modelQuery)
  const modelPage = paginateSetupModels(filteredModels, modelPageNumber)

  useEffect(() => {
    if (modelPage.page !== modelPageNumber) setModelPageNumber(modelPage.page)
  }, [modelPage.page, modelPageNumber])

  useEffect(() => {
    const shouldFocusValidation =
      shouldShowStepOneIssues || Boolean(stepOneAttempted && draftBuildError)
    if (!shouldFocusValidation) {
      validationFocusHandledRef.current = false
      return
    }
    if (validationFocusHandledRef.current) return
    validationFocusHandledRef.current = true
    const currentFieldErrors = getModelDraftFieldErrors(models)
    const firstInvalidModelIndex = models.findIndex((model) => {
      const errors = currentFieldErrors[model.id]
      return Boolean(errors?.name || errors?.baseUrl)
    })
    if (firstInvalidModelIndex >= 0) {
      setModelQuery('')
      setModelPageNumber(Math.floor(firstInvalidModelIndex / SETUP_MODELS_PER_PAGE) + 1)
    }
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        panelRef.current?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus()
      })
    })
  }, [draftBuildError, models, shouldShowStepOneIssues, stepOneAttempted])

  const handleAddModel = () => {
    setModelQuery('')
    setModelPageNumber(Math.max(1, Math.ceil((models.length + 1) / SETUP_MODELS_PER_PAGE)))
    onAddModel()
  }

  return (
    <div ref={panelRef} className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-subtle-foreground" />
          <input
            type="search"
            aria-label="Find a model"
            value={modelQuery}
            onChange={(event) => {
              setModelQuery(event.target.value)
              setModelPageNumber(1)
            }}
            placeholder="Find a model…"
            className="h-8 w-56 rounded-md border border-input bg-surface pl-8 pr-2 text-[13px] shadow-xs outline-none placeholder:text-subtle-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
          />
        </div>
        <span className="text-xs text-muted-foreground" aria-live="polite">
          {modelQuery.trim()
            ? `${filteredModels.length} of ${models.length} models`
            : `${models.length} model${models.length === 1 ? '' : 's'}`}
        </span>
        <Button variant="outline" size="sm" className="ml-auto" onClick={handleAddModel}>
          <Plus />
          Add model
        </Button>
      </div>

      {removedModel ? (
        <Alert
          tone="neutral"
          actions={
            <Button size="sm" variant="outline" onClick={onUndoRemoveModel}>
              Undo
            </Button>
          }
        >
          Removed{' '}
          <strong className="font-medium text-foreground">
            {removedModel.name || 'model draft'}
          </strong>
          .
        </Alert>
      ) : null}

      <div className="flex flex-col gap-3" aria-label="Connected models">
        {modelPage.items.map((model) => {
          const index = models.findIndex((candidate) => candidate.id === model.id)
          const fieldErrors = fieldErrorsByModelId[model.id] ?? {}
          return (
            <ModelCard
              key={model.id}
              model={model}
              index={index}
              isDefault={defaultModelId === model.id}
              canRemove={models.length > 1}
              nameError={shouldShowStepOneIssues ? fieldErrors.name : undefined}
              baseUrlError={shouldShowStepOneIssues ? fieldErrors.baseUrl : undefined}
              onUpdate={(field, value) => onUpdateModel(model.id, field, value)}
              onRemove={() => setPendingRemoveModel(model)}
              onMakeDefault={() => onSelectDefaultModel(model.id)}
            />
          )
        })}
        {modelPage.total === 0 ? (
          <p
            className="rounded-lg border border-dashed border-border px-4 py-6 text-center text-[13px] text-muted-foreground"
            role="status"
          >
            No models match “{modelQuery.trim()}”. Clear the search to see every model.
          </p>
        ) : null}
      </div>

      {modelPage.pageCount > 1 ? (
        <nav
          className="flex items-center justify-between text-xs text-muted-foreground"
          aria-label="Model pages"
        >
          <Button
            variant="ghost"
            size="xs"
            onClick={() => setModelPageNumber((page) => Math.max(1, page - 1))}
            disabled={modelPage.page === 1}
          >
            Previous
          </Button>
          <span aria-live="polite">
            Page {modelPage.page} of {modelPage.pageCount}
          </span>
          <Button
            variant="ghost"
            size="xs"
            onClick={() => setModelPageNumber((page) => Math.min(modelPage.pageCount, page + 1))}
            disabled={modelPage.page === modelPage.pageCount}
          >
            Next models
          </Button>
        </nav>
      ) : null}

      {shouldShowStepOneIssues || (stepOneAttempted && draftBuildError) ? (
        <div role="alert">
          <Alert tone="danger" title="Finish the model setup before continuing">
            <ul className="list-disc pl-4">
              {shouldShowStepOneIssues
                ? stepOneErrors.map((error) => <li key={error}>{error}</li>)
                : null}
              {stepOneAttempted && draftBuildError ? <li>{draftBuildError}</li> : null}
            </ul>
          </Alert>
        </div>
      ) : null}

      <ConfirmDialog
        isOpen={Boolean(pendingRemoveModel)}
        eyebrow="Model inventory"
        title="Remove this model?"
        description={
          <p>
            {pendingRemoveModel?.name || 'This model draft'} will be removed from the setup
            configuration. You can undo right after.
          </p>
        }
        confirmLabel="Remove model"
        tone="danger"
        onCancel={() => setPendingRemoveModel(null)}
        onConfirm={() => {
          if (pendingRemoveModel) onRemoveModel(pendingRemoveModel.id)
          setPendingRemoveModel(null)
        }}
      />
    </div>
  )
}

interface RoutingStarterPanelProps {
  routingMode: SetupRoutingMode
  remoteConfigUrl: string
  remoteImportState: RemoteImportState
  remoteImportError: string | null
  importedConfig: ImportedSetupConfig | null
  counts: SetupConfigCounts
  presets: PresetInfo[]
  presetCatalogState: PresetCatalogState
  presetCatalogError: string | null
  selectedPresetId: string | null
  presetRequestState: PresetRequestState
  presetDelta: PresetDelta | null
  presetImportedConfig: ImportedSetupConfig | null
  presetError: string | null
  onSelectRoutingMode: (mode: SetupRoutingMode) => void
  onChangeRemoteConfigUrl: (value: string) => void
  onImportRemoteConfig: () => void
  onSelectPreset: (id: string) => void
  onImportPresetConfig: () => void
  onRetryPresets: () => void
}

function ChoiceCard({
  active,
  icon,
  title,
  meta,
  description,
  onSelect,
}: {
  active: boolean
  icon: typeof Layers
  title: string
  meta: string
  description: string
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      data-choice
      onClick={onSelect}
      className={cn(
        'relative flex flex-col gap-3 rounded-xl border bg-card p-4 text-left shadow-xs outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/40',
        active
          ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
          : 'border-border hover:border-border-strong',
      )}
    >
      <span className="flex items-start gap-3">
        <IconTile icon={icon} tone="blue" />
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-sm font-semibold tracking-[-0.01em]">{title}</span>
          <span className="truncate text-xs text-subtle-foreground">{meta}</span>
        </span>
      </span>
      <span className="text-[13px] leading-relaxed text-muted-foreground">{description}</span>
      {active ? (
        <motion.span
          initial={{ scale: 0.4, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', bounce: 0.4, duration: 0.35 }}
          className="absolute -right-1.5 -top-1.5 grid size-5 place-items-center rounded-full bg-primary text-primary-foreground shadow-sm"
        >
          <Check className="size-3" strokeWidth={3} />
        </motion.span>
      ) : null}
    </button>
  )
}

export function RoutingStarterPanel({
  routingMode,
  remoteConfigUrl,
  remoteImportState,
  remoteImportError,
  importedConfig,
  counts,
  presets,
  presetCatalogState,
  presetCatalogError,
  selectedPresetId,
  presetRequestState,
  presetDelta,
  presetImportedConfig,
  presetError,
  onSelectRoutingMode,
  onChangeRemoteConfigUrl,
  onImportRemoteConfig,
  onSelectPreset,
  onImportPresetConfig,
  onRetryPresets,
}: RoutingStarterPanelProps) {
  const isPresetMode = routingMode === 'preset'
  const isRemoteMode = routingMode === 'remote'
  const isImporting = remoteImportState === 'importing'

  return (
    <div className="flex flex-col gap-5">
      <div
        role="radiogroup"
        aria-label="Routing options"
        className="grid grid-cols-1 gap-3 md:grid-cols-2"
      >
        <ChoiceCard
          active={routingMode === 'scratch'}
          icon={Layers}
          title="From scratch"
          meta="Default catch-all"
          description="One route sends every request to the default model. Grow the routing tree after activation."
          onSelect={() => onSelectRoutingMode('scratch')}
        />
        {presets.map((preset) => (
          <ChoiceCard
            key={preset.id}
            active={isPresetMode && selectedPresetId === preset.id}
            icon={Sparkles}
            title={preset.label}
            meta={`${preset.required_models.length} model${preset.required_models.length === 1 ? '' : 's'}`}
            description={preset.summary}
            onSelect={() => {
              onSelectRoutingMode('preset')
              onSelectPreset(preset.id)
            }}
          />
        ))}
        {presetCatalogState === 'loading'
          ? [0, 1].map((key) => <Skeleton key={key} className="h-[132px] rounded-xl" />)
          : null}
        <ChoiceCard
          active={isRemoteMode}
          icon={CloudDownload}
          title="From remote"
          meta={
            importedConfig
              ? `${counts.models} models · ${counts.decisions} decisions`
              : 'Import config.yaml'
          }
          description="Fetch a config.yaml from a URL and reuse its routing graph instead of a blank baseline."
          onSelect={() => onSelectRoutingMode('remote')}
        />
      </div>

      {presetCatalogState === 'loading' ? (
        <p className="text-xs text-muted-foreground" role="status">
          Loading starter architectures…
        </p>
      ) : null}

      {presetCatalogState === 'error' ? (
        <div role="alert">
          <Alert
            tone="warning"
            title="Starter architectures unavailable"
            actions={
              <Button size="sm" variant="outline" onClick={onRetryPresets}>
                Retry presets
              </Button>
            }
          >
            {presetCatalogError}
          </Alert>
        </div>
      ) : null}

      {isPresetMode && presetRequestState === 'loading' ? (
        <Card className="flex flex-col gap-2 p-4" role="status">
          <span className="text-[13px] font-medium">
            Checking the architecture against your models…
          </span>
          <Skeleton className="h-1.5 w-full" />
        </Card>
      ) : null}

      {isPresetMode && selectedPresetId && presetRequestState === 'idle' && !presetDelta ? (
        <Alert
          tone="info"
          actions={
            <Button size="sm" variant="outline" onClick={() => onSelectPreset(selectedPresetId)}>
              Recheck preset
            </Button>
          }
        >
          The model inventory changed. Recheck this preset before continuing.
        </Alert>
      ) : null}

      {isPresetMode && presetDelta ? (
        <SetupWizardPresetChecklist
          presetDelta={presetDelta}
          presetImportedConfig={presetImportedConfig}
          counts={counts}
          presetError={presetError}
          presetRequestState={presetRequestState}
          onImportPresetConfig={onImportPresetConfig}
        />
      ) : null}

      {isPresetMode && !presetDelta && presetError ? (
        <div role="alert">
          <Alert
            tone="danger"
            title="The preset could not be prepared"
            actions={
              selectedPresetId ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onSelectPreset(selectedPresetId)}
                >
                  Retry selected preset
                </Button>
              ) : null
            }
          >
            {presetError}
          </Alert>
        </div>
      ) : null}

      {isRemoteMode ? (
        <Card className="flex flex-col gap-3 p-4" aria-busy={isImporting}>
          <Field
            label="Remote config URL"
            htmlFor="setup-remote-url"
            error={remoteImportError ?? undefined}
            hint="A direct YAML link. The wizard fetches and parses it, and the review step validates it."
          >
            <div className="flex gap-2">
              <Input
                id="setup-remote-url"
                value={remoteConfigUrl}
                onChange={(event) => onChangeRemoteConfigUrl(event.target.value)}
                placeholder={DEFAULT_REMOTE_SETUP_CONFIG_URL}
                aria-invalid={Boolean(remoteImportError)}
                spellCheck={false}
                className="font-mono text-[12.5px]"
              />
              <Button variant="outline" onClick={onImportRemoteConfig} loading={isImporting}>
                {isImporting
                  ? 'Importing…'
                  : remoteImportState === 'error'
                    ? 'Retry import'
                    : 'Import'}
              </Button>
            </div>
          </Field>
          {isImporting ? (
            <p className="text-xs text-muted-foreground" role="status">
              Fetching and parsing the remote configuration…
            </p>
          ) : null}
          {importedConfig ? (
            <div role="status">
              <Alert tone="success" title="Remote config ready">
                <span className="block">
                  {counts.models} models · {counts.decisions} decisions · {counts.signals} signals
                </span>
                <span className="block truncate font-mono text-[12px]">
                  {importedConfig.sourceUrl}
                </span>
              </Alert>
            </div>
          ) : null}
        </Card>
      ) : null}

      <div className="grid grid-cols-3 gap-3">
        <MiniStat label="Models ready" value={counts.models} />
        <MiniStat label="Decisions" value={counts.decisions} />
        <MiniStat label="Signals" value={counts.signals} />
      </div>
    </div>
  )
}
