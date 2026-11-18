import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  BrainCircuit,
  ChartNoAxesColumn,
  Copy,
  Layers,
  MoreHorizontal,
  Network,
  Package,
  Pencil,
  Radar,
  Split,
  Trash2,
} from 'lucide-react'

import { configCollectionItemYaml } from '../lib/configYaml'
import {
  Alert,
  Badge,
  Button,
  CodeBlock,
  CopyField,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  IconTile,
  KeyValueGrid,
  MiniStat,
  ProviderStack,
  SectionLabel,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  toast,
  useCopy,
} from '../ui'
import ConfigPageResolutionFlow from './ConfigPageResolutionFlow'
import { collectDecisionTargetModels } from './configPageEntrypointsRecipesSupport'
import { ModelLogo } from './ConfigPageModelLogo'
import { modelProviderIds, providerLogoKey } from './configPageModelPresentation'
import {
  apiLabel,
  decisionLanes,
  publicModelResolution,
  recipeDisplayName,
  usageExamples,
  type EntrypointSheetView,
  type PublicModelCard,
} from './configPageRecipesPageSupport'
import { EntityYamlPanel } from './configPageRoutingScopeViews'
import type { ConfigData, NormalizedModel } from './configPageSupport'

interface ConfigPageEntrypointSheetProps {
  card: PublicModelCard | null
  open: boolean
  view: EntrypointSheetView
  onViewChange: (view: EntrypointSheetView) => void
  onOpenChange: (open: boolean) => void
  config: ConfigData
  models: NormalizedModel[]
  isReadonly: boolean
  routerHealthy: boolean
  onEdit: () => void
  onDelete: () => void
  onOpenRecipe: (recipe: string) => void
}

function DecisionLanes({
  card,
  onOpenModel,
}: {
  card: PublicModelCard
  onOpenModel: (name: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const lanes = decisionLanes(card.recipe)
  const visible = expanded ? lanes : lanes.slice(0, 6)
  if (lanes.length === 0) return null
  return (
    <section className="flex flex-col gap-2">
      <SectionLabel>Decision lanes · {lanes.length}</SectionLabel>
      <div className="flex flex-col divide-y divide-border rounded-lg border border-border">
        {visible.map(({ decision, targets }) => (
          <div key={decision.name} className="flex flex-col gap-1.5 px-3 py-2.5">
            <div className="flex items-center gap-2">
              <Split className="size-3.5 shrink-0 text-chart-3" />
              <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] font-medium">
                {decision.name}
              </span>
              <span className="font-mono text-[11px] tabular text-subtle-foreground">
                P{decision.priority ?? 0}
              </span>
            </div>
            {targets.length ? (
              <span className="flex flex-wrap gap-1 pl-5">
                {targets.map((target) => (
                  <button
                    key={target}
                    type="button"
                    onClick={() => onOpenModel(target)}
                    className="inline-flex h-6 items-center rounded-md border border-border bg-surface px-1.5 font-mono text-[11.5px] transition-colors hover:border-border-strong hover:bg-surface-subtle"
                  >
                    {target}
                  </button>
                ))}
              </span>
            ) : (
              <span className="pl-5 text-xs text-muted-foreground">
                Responds without calling a model
              </span>
            )}
          </div>
        ))}
      </div>
      {lanes.length > 6 ? (
        <Button
          variant="ghost"
          size="xs"
          className="w-fit"
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? 'Show fewer' : `Show all ${lanes.length} decisions`}
        </Button>
      ) : null}
    </section>
  )
}

export default function ConfigPageEntrypointSheet({
  card,
  open,
  view,
  onViewChange,
  onOpenChange,
  config,
  models,
  isReadonly,
  routerHealthy,
  onEdit,
  onDelete,
  onOpenRecipe,
}: ConfigPageEntrypointSheetProps) {
  const navigate = useNavigate()
  const { copy } = useCopy()
  if (!card) return null
  const name = card.key
  const recipeName = recipeDisplayName(config, card.entrypoint.recipe)
  const resolution = publicModelResolution(card)
  const targetModels = card.targets.map((target) => ({
    name: target,
    model: models.find((candidate) => candidate.name === target),
  }))
  const providers = [
    ...new Set(
      targetModels.flatMap(({ model }) =>
        model
          ? modelProviderIds(model)
              .slice(0, 1)
              .map((id) => providerLogoKey(id))
          : [],
      ),
    ),
  ]
  const strategy = String(card.recipe?.routing.strategy ?? 'priority')
  const editable = !isReadonly && card.api === 'chat'
  const deletable = editable && !card.derived
  const origin = window.location.origin.replace(/\/$/, '')
  const yamlEntry = { ...card.entrypoint, recipe: recipeName }
  const openModel = (model: string) => navigate(`/models/backends?m=${encodeURIComponent(model)}`)

  return (
    <DetailSheet open={open} onOpenChange={onOpenChange} size="xl">
      <DetailSheetHeader
        icon={<IconTile icon={Layers} tone="blue" size="lg" />}
        title={<span className="font-mono text-[15px]">{name}</span>}
        badges={
          <>
            {resolution !== 'resolving' ? (
              <Badge tone={resolution === 'missing' ? 'danger' : 'warning'} dot>
                {resolution === 'missing' ? 'Recipe missing' : 'No decisions'}
              </Badge>
            ) : routerHealthy ? (
              <Badge tone="success" dot>
                Live
              </Badge>
            ) : (
              <Badge tone="neutral" dot>
                Configured
              </Badge>
            )}
            {card.entrypoint.recipe === 'default' && card.api === 'chat' ? (
              <Badge tone="brand">default</Badge>
            ) : null}
            {card.api === 'systemone' ? <Badge tone="violet">System One</Badge> : null}
          </>
        }
        description={
          <>
            <span>
              recipe <span className="font-mono">{recipeName}</span>
            </span>
            <span className="text-subtle-foreground">·</span>
            <span>{apiLabel(card)} API</span>
            <span className="text-subtle-foreground">·</span>
            <span>
              {card.targets.length} {card.targets.length === 1 ? 'backend' : 'backends'}
            </span>
          </>
        }
        actions={
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label="Public model actions">
                <MoreHorizontal />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem
                onSelect={() => {
                  copy(name)
                  toast.success('Model name copied')
                }}
              >
                <Copy />
                Copy model name
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={() => navigate(`/observe/evaluations?model=${encodeURIComponent(name)}`)}
              >
                <ChartNoAxesColumn />
                Evaluate
              </DropdownMenuItem>
              {deletable ? (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem variant="destructive" onSelect={onDelete}>
                    <Trash2 />
                    Delete
                  </DropdownMenuItem>
                </>
              ) : null}
            </DropdownMenuContent>
          </DropdownMenu>
        }
      />

      <Tabs
        value={view}
        onValueChange={(next) => onViewChange(next as EntrypointSheetView)}
        className="min-h-0 flex-1 gap-0"
      >
        <TabsList className="px-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="backends">
            Backends
            <Badge size="sm" tone="neutral" className="tabular">
              {card.targets.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="connect">Connect</TabsTrigger>
          <TabsTrigger value="yaml">YAML</TabsTrigger>
        </TabsList>
        <DetailSheetBody>
          <TabsContent value="overview" className="flex flex-col gap-5 px-5 py-5">
            {resolution === 'missing' ? (
              <Alert tone="danger" title={`The recipe ${recipeName} does not exist`}>
                Requests to this name fail until it points to an existing recipe. Edit the public
                model to choose one.
              </Alert>
            ) : resolution === 'empty' ? (
              <Alert tone="warning" title={`${recipeName} has no decisions yet`}>
                Add a decision to the recipe so requests to this name reach a backend.
              </Alert>
            ) : null}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MiniStat
                label="Signals"
                value={card.signals}
                hint={`${card.signalFamilies} families`}
              />
              <MiniStat label="Decisions" value={card.decisions.length} hint={`by ${strategy}`} />
              <MiniStat label="Projections" value={card.projections} />
              <MiniStat label="Backends" value={card.targets.length} hint="candidate models" />
            </div>

            <section className="flex flex-col gap-2">
              <SectionLabel>How a request resolves</SectionLabel>
              <ConfigPageResolutionFlow
                className="overflow-x-auto pb-1"
                steps={[
                  {
                    key: 'model',
                    icon: (
                      <IconTile
                        icon={card.api === 'systemone' ? BrainCircuit : Layers}
                        tone="blue"
                        size="xs"
                      />
                    ),
                    label: 'Public model',
                    value: <span className="font-mono">{name.replace(/^vllm-sr\//, '')}</span>,
                    hint: `${apiLabel(card)} API`,
                    grow: 1.6,
                  },
                  {
                    key: 'recipe',
                    icon: <IconTile icon={Package} tone="blue" size="xs" />,
                    label: 'Recipe',
                    value: <span className="font-mono">{recipeName}</span>,
                    hint: <span className="capitalize">{strategy}</span>,
                    grow: recipeName.length > 10 ? 1.3 : 1,
                    onClick: card.recipe ? () => onOpenRecipe(card.entrypoint.recipe) : undefined,
                  },
                  {
                    key: 'signals',
                    icon: <IconTile icon={Radar} tone="gold" size="xs" />,
                    label: 'Signals',
                    value: `${card.signals} ${card.signals === 1 ? 'signal' : 'signals'}`,
                    hint: `${card.signalFamilies} ${card.signalFamilies === 1 ? 'family' : 'families'}`,
                    onClick: card.recipe
                      ? () =>
                          navigate(
                            `/routing/signals?recipe=${encodeURIComponent(card.entrypoint.recipe)}`,
                          )
                      : undefined,
                  },
                  {
                    key: 'decisions',
                    icon: <IconTile icon={Split} tone="violet" size="xs" />,
                    label: 'Decisions',
                    value: `${card.decisions.length} ${card.decisions.length === 1 ? 'decision' : 'decisions'}`,
                    hint: `by ${strategy}`,
                    onClick: card.recipe
                      ? () =>
                          navigate(
                            `/routing/decisions?recipe=${encodeURIComponent(card.entrypoint.recipe)}`,
                          )
                      : undefined,
                  },
                  {
                    key: 'backends',
                    icon: providers.length ? (
                      <span className="flex h-6 items-center">
                        <ProviderStack providers={providers} size={14} max={3} />
                      </span>
                    ) : (
                      <IconTile icon={Network} tone="green" size="xs" />
                    ),
                    label: 'Backends',
                    value: `${card.targets.length} ${card.targets.length === 1 ? 'model' : 'models'}`,
                    hint:
                      card.targets.length > 1
                        ? `${card.targets[0]} +${card.targets.length - 1}`
                        : (card.targets[0] ?? 'none'),
                    onClick: () => onViewChange('backends'),
                  },
                ]}
              />
              <p className="text-xs leading-relaxed text-muted-foreground">
                The Router extracts the recipe&apos;s signals once per request. The{' '}
                {strategy === 'confidence' ? 'most confident' : 'highest-priority'} decision that
                matches picks a backend with its algorithm.
              </p>
            </section>

            <DecisionLanes card={card} onOpenModel={openModel} />

            <section className="flex flex-col gap-2">
              <SectionLabel>Details</SectionLabel>
              <KeyValueGrid
                items={[
                  {
                    label: 'Model names',
                    value: (
                      <span className="flex flex-wrap gap-1">
                        {card.entrypoint.model_names.map((modelName) => (
                          <Badge key={modelName} tone="outline" className="font-mono">
                            {modelName}
                          </Badge>
                        ))}
                      </span>
                    ),
                    span: true,
                  },
                  {
                    label: 'Protocols',
                    value:
                      card.api === 'systemone'
                        ? 'System One typed questions'
                        : 'Chat Completions, Responses, Anthropic Messages',
                  },
                  { label: 'Recipe', value: recipeName, mono: true },
                  {
                    label: 'Review model',
                    value: card.entrypoint.review_model ?? 'Session model',
                    mono: Boolean(card.entrypoint.review_model),
                  },
                  {
                    label: 'Memory model',
                    value: card.entrypoint.memory_model ?? 'Session model',
                    mono: Boolean(card.entrypoint.memory_model),
                  },
                  ...(card.recipe?.description
                    ? [{ label: 'Recipe description', value: card.recipe.description, span: true }]
                    : []),
                ]}
              />
              {card.derived ? (
                <p className="text-xs text-muted-foreground">
                  No entrypoint names the default recipe, so the Router serves it as{' '}
                  <span className="font-mono">vllm-sr/auto</span>. Editing it adds that entrypoint
                  to config.yaml.
                </p>
              ) : null}
            </section>
          </TabsContent>

          <TabsContent value="backends" className="flex flex-col gap-2 px-5 py-5">
            <p className="mb-1 text-[13px] text-muted-foreground">
              Decisions in this recipe can send traffic to these models.
            </p>
            {targetModels.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border-strong px-3 py-6 text-center text-xs text-muted-foreground">
                No decision in this recipe names a model yet.
              </p>
            ) : (
              targetModels.map(({ name: target, model }) => {
                const lanes = card.decisions.filter((decision) =>
                  collectDecisionTargetModels(decision).includes(target),
                ).length
                return (
                  <button
                    key={target}
                    type="button"
                    onClick={() => openModel(target)}
                    className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-border-strong hover:bg-surface-subtle"
                  >
                    {model ? (
                      <ModelLogo model={model} size={18} framed />
                    ) : (
                      <IconTile icon={Network} tone="slate" size="sm" />
                    )}
                    <span className="flex min-w-0 flex-1 flex-col">
                      <span className="truncate font-mono text-[12.5px] font-medium">{target}</span>
                      <span className="truncate text-xs text-subtle-foreground">
                        {model
                          ? model.provider_model_id || model.catalog || 'Backend model'
                          : 'Not a configured backend model (a LoRA adapter or a missing model)'}
                      </span>
                    </span>
                    <span className="shrink-0 text-xs tabular text-muted-foreground">
                      {lanes} {lanes === 1 ? 'decision' : 'decisions'}
                    </span>
                  </button>
                )
              })
            )}
          </TabsContent>

          <TabsContent value="connect" className="flex flex-col gap-4 px-5 py-5">
            {card.api === 'systemone' ? (
              <Alert tone="info" title="A System One public model">
                It answers typed questions on the native System One API. Grant the name on a
                listener to publish it; chat clients cannot call it.
              </Alert>
            ) : (
              <p className="text-[13px] text-muted-foreground">
                OpenAI-compatible and streaming ready. Point any OpenAI client at the router and
                send this name in the model field.
              </p>
            )}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <SectionLabel>Base URL</SectionLabel>
                <CopyField value={`${origin}/v1`} />
              </div>
              <div className="flex flex-col gap-1.5">
                <SectionLabel>Model</SectionLabel>
                <CopyField value={name} />
              </div>
            </div>
            {card.api === 'chat' ? (
              <>
                <p className="text-xs text-muted-foreground">No API key required</p>
                <CodeBlock tabs={usageExamples(origin, name)} />
              </>
            ) : null}
          </TabsContent>

          <TabsContent value="yaml" className="px-5 py-5">
            <EntityYamlPanel
              path={`entrypoints[${name}]`}
              yaml={configCollectionItemYaml(['entrypoints'], yamlEntry)}
              note={
                card.derived
                  ? 'Derived default; config.yaml does not state it yet.'
                  : 'The entrypoint as config.yaml states it.'
              }
            />
          </TabsContent>
        </DetailSheetBody>
      </Tabs>

      <DetailSheetFooter>
        <span className="min-w-0 truncate text-xs text-muted-foreground">
          {card.api === 'systemone'
            ? 'Edit System One public models in the config editor.'
            : isReadonly
              ? 'Read-only: your account can view this configuration.'
              : 'The Router validates every change when you save.'}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              navigate(`/routing/topology?scope=${encodeURIComponent(card.entrypoint.recipe)}`)
            }
          >
            <Network />
            Open topology
          </Button>
          {editable ? (
            <Button size="sm" onClick={onEdit}>
              <Pencil />
              Edit
            </Button>
          ) : card.api === 'systemone' && !isReadonly ? (
            <Button size="sm" onClick={() => navigate('/routing/editor')}>
              <Pencil />
              Open config editor
            </Button>
          ) : null}
        </div>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
