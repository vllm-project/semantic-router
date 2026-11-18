import { useId, useRef } from 'react'
import {
  ArrowLeft,
  Check,
  KeyRound,
  Link2,
  ListChecks,
  Plus,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Sparkles,
  X,
} from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  Checkbox,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Field,
  Input,
  InputGroup,
  SectionLabel,
  cn,
} from '../ui'
import {
  focusInitialDialogControl,
  keepOpenForToastInteraction,
} from '../components/dialogFocusSupport'
import ConfigPageConnectModelAdvanced from './ConfigPageConnectModelAdvanced'
import type { ConnectModelsDialogController } from './configPageConnectModelsDialogController'
import { requestedConnectedModelName } from './configPageConnectModelSupport'
import { CatalogProviderLogo } from './ConfigPageModelLogo'
import type { ModelProviderPreset } from './modelProviderCatalog'

const providerCategories = ['Start here', 'Model APIs', 'Private runtimes'] as const

interface Props {
  isOpen: boolean
  controller: ConnectModelsDialogController
  reasoningFamilies: string[]
  onClose: () => void
  onManualSetup: () => void
}

export default function ConfigPageConnectModelsDialogView({
  isOpen,
  controller,
  reasoningFamilies,
  onClose,
  onManualSetup,
}: Props) {
  const contentRef = useRef<HTMLDivElement>(null)
  const choosingModels = controller.stage === 'models' && controller.provider
  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open && !controller.busy) onClose()
      }}
    >
      <DialogContent
        ref={contentRef}
        size="xl"
        hideClose
        aria-busy={controller.busy}
        className="flex h-[min(780px,calc(100vh-2rem))] flex-col"
        onOpenAutoFocus={(event) => focusInitialDialogControl(event, contentRef.current)}
        onInteractOutside={keepOpenForToastInteraction}
      >
        <DialogHeader className="flex-row items-center gap-3 border-b border-border pb-4">
          {choosingModels ? (
            <>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                className="-ml-1.5"
                onClick={() => controller.setStage('provider')}
                disabled={controller.busy}
                aria-label="Choose another provider"
              >
                <ArrowLeft />
              </Button>
              <CatalogProviderLogo provider={controller.provider!.id} size={22} framed />
            </>
          ) : null}
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            <DialogTitle>
              {controller.stage === 'provider'
                ? 'Add models'
                : (controller.provider?.name ?? 'Connect provider')}
            </DialogTitle>
            <DialogDescription>
              {controller.stage === 'provider'
                ? 'Choose where your models run.'
                : 'Connect once, then import one or many models.'}
            </DialogDescription>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="-mr-2"
            onClick={onClose}
            disabled={controller.busy}
            aria-label="Close"
          >
            <X />
          </Button>
        </DialogHeader>
        {controller.error ? (
          <div className="px-6 pt-4">
            <Alert tone="danger">{controller.error}</Alert>
          </div>
        ) : null}
        <DialogBody className="flex min-h-0 flex-1 flex-col gap-5 py-5">
          {controller.stage === 'provider' ? (
            <ProviderStage controller={controller} />
          ) : controller.provider ? (
            <ModelsStage controller={controller} reasoningFamilies={reasoningFamilies} />
          ) : null}
        </DialogBody>
        <ConnectFooter controller={controller} onClose={onClose} onManualSetup={onManualSetup} />
      </DialogContent>
    </Dialog>
  )
}

function ProposedModelNotice({ controller }: Pick<Props, 'controller'>) {
  const model = controller.proposedModel
  if (!model) return null
  const serving = controller.servingProviders.length
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-primary/25 bg-primary-soft/40 px-3.5 py-2.5">
      <Sparkles className="size-4 shrink-0 text-primary" />
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="text-[13px] font-medium">Adding {model.display_name}</span>
        <span className="text-xs text-muted-foreground">
          {serving > 0
            ? `${serving} ${serving === 1 ? 'provider serves' : 'providers serve'} it. Choose where it runs.`
            : 'No catalog provider lists it. Choose a provider and enter its model ID.'}
        </span>
      </div>
      {serving > 0 ? (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={() => controller.setShowEveryProvider((current) => !current)}
        >
          {controller.showEveryProvider ? 'Only providers that serve it' : 'Show every provider'}
        </Button>
      ) : null}
    </div>
  )
}

function ProviderStage({ controller }: Pick<Props, 'controller'>) {
  const filteredToServing =
    controller.servingProviders.length > 0 &&
    !controller.showEveryProvider &&
    !controller.search.trim()
  return (
    <>
      <ProposedModelNotice controller={controller} />
      <div className="flex items-center gap-3">
        <InputGroup
          type="search"
          value={controller.search}
          onChange={(event) => controller.setSearch(event.target.value)}
          placeholder="Search providers"
          aria-label="Search providers"
          leading={<Search />}
          containerClassName="flex-1"
          data-dialog-initial-focus
        />
        <span className="shrink-0 text-xs tabular text-muted-foreground">
          {controller.visibleProviders.length} providers
        </span>
      </div>
      {filteredToServing ? (
        <ProviderGrid
          providers={controller.visibleProviders}
          onChoose={controller.chooseProvider}
        />
      ) : (
        providerCategories.map((category) => {
          const providers = controller.visibleProviders.filter((item) => item.category === category)
          if (providers.length === 0) return null
          return (
            <section key={category} className="flex flex-col gap-2">
              <SectionLabel>{category}</SectionLabel>
              <ProviderGrid providers={providers} onChoose={controller.chooseProvider} />
            </section>
          )
        })
      )}
      {!filteredToServing && !controller.search.trim() && controller.hiddenProviderCount > 0 ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="self-center"
          aria-expanded={controller.showAllProviders}
          onClick={() => controller.setShowAllProviders((current) => !current)}
        >
          {controller.showAllProviders
            ? 'Show featured providers'
            : `More providers (${controller.hiddenProviderCount})`}
        </Button>
      ) : null}
      {controller.visibleProviders.length === 0 ? (
        <p className="py-8 text-center text-[13px] text-muted-foreground">
          No provider matches “{controller.search.trim()}”. Use Add manually for a custom backend.
        </p>
      ) : null}
    </>
  )
}

function ProviderGrid({
  providers,
  onChoose,
}: {
  providers: ModelProviderPreset[]
  onChoose: (provider: ModelProviderPreset) => void
}) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {providers.map((provider) => (
        <button
          key={provider.id}
          type="button"
          onClick={() => onChoose(provider)}
          className="group flex min-w-0 items-start gap-3 rounded-lg border border-border bg-card px-3 py-2.5 text-left shadow-xs outline-none transition-[border-color,background-color,box-shadow] hover:border-border-strong hover:bg-surface-subtle focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
        >
          <CatalogProviderLogo provider={provider.id} size={18} framed />
          <span className="flex min-w-0 flex-col gap-0.5">
            <strong className="truncate text-[13px] font-medium">{provider.name}</strong>
            <small className="line-clamp-2 text-xs leading-snug text-muted-foreground">
              {provider.description}
            </small>
          </span>
        </button>
      ))}
    </div>
  )
}

function ModelsStage({
  controller,
  reasoningFamilies,
}: Pick<Props, 'controller' | 'reasoningFamilies'>) {
  return (
    <>
      <ConnectionPanel controller={controller} />
      <ManualModelRow controller={controller} />
      {controller.models.length > 0 ? <ModelPicker controller={controller} /> : null}
      <ConfigPageConnectModelAdvanced
        value={controller.advanced}
        reasoningFamilies={reasoningFamilies}
        onChange={controller.setAdvanced}
      />
    </>
  )
}

function ConnectionPanel({ controller }: Pick<Props, 'controller'>) {
  const provider = controller.provider!
  const baseUrlId = useId()
  const apiKeyId = useId()
  return (
    <section className="flex flex-col gap-4 rounded-xl border border-border bg-surface-subtle/50 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-[13px] font-medium">Connection</span>
          <span className="text-xs text-muted-foreground">
            {provider.supportsModelDiscovery
              ? 'Credentials stay private and are only used for this provider.'
              : controller.catalogModels.size > 0
                ? 'Choose from the built-in inventory or enter a custom model ID below.'
                : 'This provider does not expose catalog-backed discovery; enter a model ID below.'}
          </span>
        </div>
        {provider.supportsModelDiscovery ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void controller.discover()}
            disabled={controller.busy || !controller.baseUrl.trim()}
            loading={controller.discovering}
          >
            {controller.discovering ? null : controller.models.length > 0 ? (
              <RefreshCw />
            ) : (
              <ListChecks />
            )}
            {controller.discovering
              ? 'Connecting…'
              : controller.models.length > 0
                ? 'Refresh'
                : 'List models'}
          </Button>
        ) : null}
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {!provider.baseUrl ? (
          <Field label="Base URL" htmlFor={baseUrlId}>
            <Input
              id={baseUrlId}
              type="url"
              value={controller.baseUrl}
              onChange={(event) => controller.setBaseUrl(event.target.value)}
              placeholder="https://api.example.com/v1"
              className="font-mono text-[12.5px]"
              data-dialog-initial-focus
            />
          </Field>
        ) : (
          <div className="grid content-start gap-1.5">
            <span className="text-[13px] font-medium leading-5">API endpoint</span>
            <span className="flex h-9 min-w-0 items-center gap-2 rounded-md border border-border bg-surface px-3 text-muted-foreground">
              <Link2 className="size-3.5 shrink-0" />
              <code className="truncate font-mono text-[12.5px] text-foreground/85">
                {provider.baseUrl}
              </code>
            </span>
          </div>
        )}
        {provider.authStrategy !== 'none' ? (
          <Field
            label={
              <>
                API key
                <span className="font-normal text-subtle-foreground">
                  {provider.baseUrl ? 'Required' : 'Optional'}
                </span>
              </>
            }
            htmlFor={apiKeyId}
          >
            <InputGroup
              id={apiKeyId}
              type="password"
              value={controller.apiKey}
              onChange={(event) => controller.setAPIKey(event.target.value)}
              placeholder="Paste your key"
              autoComplete="new-password"
              leading={<KeyRound />}
              data-dialog-initial-focus={provider.baseUrl ? true : undefined}
            />
          </Field>
        ) : (
          <div className="grid content-start gap-1.5">
            <span className="text-[13px] font-medium leading-5">Authentication</span>
            <span className="flex h-9 items-center gap-2 rounded-md border border-border bg-surface px-3 text-[13px] text-muted-foreground">
              <Check className="size-3.5 text-success" />
              No API key required
            </span>
          </div>
        )}
      </div>
    </section>
  )
}

function ManualModelRow({ controller }: Pick<Props, 'controller'>) {
  return (
    <div className="flex items-center gap-2">
      <InputGroup
        value={controller.manualModel}
        onChange={(event) => controller.setManualModel(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault()
            controller.addManualModel()
          }
        }}
        placeholder="Or enter a model ID"
        aria-label="Model ID"
        leading={<Plus />}
        containerClassName="flex-1"
        className="font-mono text-[12.5px]"
      />
      <Button
        type="button"
        variant="outline"
        onClick={controller.addManualModel}
        disabled={!controller.manualModel.trim()}
      >
        Add
      </Button>
    </div>
  )
}

function ModelPicker({ controller }: Pick<Props, 'controller'>) {
  const allVisibleSelected =
    controller.visibleModels.length > 0 &&
    controller.visibleModels.every((model) => controller.selected.has(model))
  const toggleVisible = () => {
    controller.setSelected((current) => {
      const next = new Set(current)
      controller.visibleModels.forEach((model) =>
        allVisibleSelected ? next.delete(model) : next.add(model),
      )
      return next
    })
  }
  return (
    <section className="flex flex-col gap-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <SectionLabel>Models</SectionLabel>
          <Badge size="sm" tone={controller.selected.size ? 'brand' : 'neutral'}>
            {controller.selected.size} selected
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <InputGroup
            type="search"
            value={controller.modelSearch}
            onChange={(event) => controller.setModelSearch(event.target.value)}
            placeholder="Filter models"
            aria-label="Filter models"
            leading={<Search />}
            className="h-8 w-48"
          />
          <Button type="button" variant="ghost" size="sm" onClick={toggleVisible}>
            <Check />
            {allVisibleSelected ? 'Clear visible' : 'Select all'}
          </Button>
        </div>
      </div>
      <div className="flex max-h-72 flex-col divide-y divide-border overflow-y-auto rounded-lg border border-border bg-card">
        {controller.visibleModels.map((model) => (
          <ModelOption key={model} controller={controller} model={model} />
        ))}
        {controller.visibleModels.length === 0 ? (
          <p className="px-3 py-6 text-center text-[13px] text-muted-foreground">
            No models match this filter.
          </p>
        ) : null}
      </div>
    </section>
  )
}

export function ModelOption({ controller, model }: Pick<Props, 'controller'> & { model: string }) {
  const providerModelIDField = useId()
  const checkboxId = useId()
  const logicalName = controller.resolvedModelNames.get(model) ?? model
  const renamed = logicalName !== requestedConnectedModelName(controller.advanced.namePrefix, model)
  const catalogID = controller.catalogModels.get(model)
  const providerModelIDKind = controller.providerModelIdRequirements.get(model)
  const needsProviderModelID = Boolean(providerModelIDKind)
  const selected = controller.selected.has(model)
  const toggle = () =>
    controller.setSelected((current) => {
      const next = new Set(current)
      if (next.has(model)) next.delete(model)
      else next.add(model)
      return next
    })
  return (
    <div className={cn('flex flex-col gap-2 px-3 py-2.5', selected && 'bg-primary-soft/40')}>
      <label htmlFor={checkboxId} className="flex cursor-pointer items-center gap-3">
        <Checkbox id={checkboxId} checked={selected} onCheckedChange={toggle} />
        <span className="flex min-w-0 flex-1 flex-col">
          <strong className="truncate font-mono text-[12.5px] font-medium">{logicalName}</strong>
          {catalogID ? (
            <small className="truncate text-xs text-muted-foreground">
              Built-in · {controller.modelDisplayNames.get(catalogID)}
            </small>
          ) : renamed ? (
            <small className="text-xs text-muted-foreground">
              Named to avoid a public model conflict
            </small>
          ) : null}
        </span>
      </label>
      {needsProviderModelID ? (
        <div className="ml-7 grid gap-1.5">
          <label
            htmlFor={providerModelIDField}
            className="text-xs font-medium text-muted-foreground"
          >
            {providerModelIDKind === 'deployment_name' ? 'Deployment name' : 'Provider model ID'}
          </label>
          <Input
            id={providerModelIDField}
            type="text"
            value={controller.providerModelIds.get(model) ?? ''}
            onChange={(event) => controller.setProviderModelId(model, event.target.value)}
            placeholder="Enter the deployed model name"
            required={selected}
            disabled={!selected}
            className="h-8 font-mono text-[12.5px]"
          />
        </div>
      ) : null}
    </div>
  )
}

function ConnectFooter({
  controller,
  onClose,
  onManualSetup,
}: Pick<Props, 'controller' | 'onClose' | 'onManualSetup'>) {
  return (
    <DialogFooter className="justify-between">
      {controller.stage === 'provider' ? (
        <Button
          type="button"
          variant="ghost"
          onClick={() => {
            onClose()
            onManualSetup()
          }}
          disabled={controller.busy}
        >
          <SlidersHorizontal />
          Add manually
        </Button>
      ) : (
        <span className="text-xs text-muted-foreground">
          {controller.selected.size > 0
            ? `${controller.selected.size} ${controller.selected.size === 1 ? 'model' : 'models'} will be added to providers.models`
            : 'Select or enter at least one model.'}
        </span>
      )}
      <div className="flex items-center gap-2">
        <Button type="button" variant="ghost" onClick={onClose} disabled={controller.busy}>
          Cancel
        </Button>
        {controller.stage === 'models' ? (
          <Button
            type="button"
            onClick={() => void controller.submit()}
            disabled={controller.busy || controller.selected.size === 0}
            loading={controller.saving}
          >
            {controller.saving ? null : <Plus />}
            {controller.saving
              ? 'Adding…'
              : `Add ${controller.selected.size ? `${controller.selected.size} ` : ''}model${controller.selected.size === 1 ? '' : 's'}`}
          </Button>
        ) : null}
      </div>
    </DialogFooter>
  )
}
