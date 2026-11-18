import { useNavigate } from 'react-router-dom'
import {
  Braces,
  Copy,
  Layers,
  MoreHorizontal,
  Network,
  Package,
  Pencil,
  Plus,
  Power,
  Split,
  Trash2,
  TriangleAlert,
} from 'lucide-react'

import { configCollectionItemYaml } from '../lib/configYaml'
import {
  Badge,
  Button,
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
  EmptyState,
  IconTile,
  KeyValueGrid,
  MiniStat,
  SectionLabel,
  StatusLabel,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  toast,
  useCopy,
} from '../ui'
import { countSignalsInProfile, type RoutingProfileLike } from '../utils/routingScopes'
import { ModelLogo } from './ConfigPageModelLogo'
import {
  decisionLanes,
  recipeRowStatus,
  recipeStatusLabel,
  type BuiltInRecipeRow,
  type ConfigRecipeRow,
  type PackageRecipeRow,
  type RecipeRow,
} from './configPageRecipesPageSupport'
import { EntityYamlPanel } from './configPageRoutingScopeViews'
import type { NormalizedModel } from './configPageSupport'

export const RECIPE_SHEET_VIEWS = ['overview', 'decisions', 'yaml'] as const
export type RecipeSheetView = (typeof RECIPE_SHEET_VIEWS)[number]

const signalTypeLabel = (type: string) => type.replace(/_/g, ' ')

interface ConfigPageRecipeSheetProps {
  row: RecipeRow | null
  open: boolean
  view: RecipeSheetView
  onViewChange: (view: RecipeSheetView) => void
  onOpenChange: (open: boolean) => void
  models: NormalizedModel[]
  pendingRestart: ReadonlySet<string>
  isReadonly: boolean
  canDeploy: boolean
  onEdit: (row: ConfigRecipeRow) => void
  onDelete: (row: ConfigRecipeRow) => void
  onNewPublicModel: (row: ConfigRecipeRow | BuiltInRecipeRow) => void
  onOpenPublicModel: (name: string) => void
  onToggle: (row: PackageRecipeRow, mode: 'activate' | 'deactivate') => void
}

function ConfigRecipeOverview({
  row,
  models,
  isReadonly,
  onNewPublicModel,
  onOpenPublicModel,
}: {
  row: ConfigRecipeRow
  models: NormalizedModel[]
  isReadonly: boolean
  onNewPublicModel: () => void
  onOpenPublicModel: (name: string) => void
}) {
  const navigate = useNavigate()
  const families = Object.entries(
    countSignalsInProfile(row.recipe.routing as RoutingProfileLike).byType,
  )
  const projections = [
    ...(row.recipe.routing.projections?.partitions ?? []).map((item) => item.name),
    ...(row.recipe.routing.projections?.scores ?? []).map((item) => item.name),
    ...(row.recipe.routing.projections?.mappings ?? []).map((item) => item.name),
  ]
  const scope = encodeURIComponent(row.id)
  return (
    <div className="flex flex-col gap-5">
      {row.description ? (
        <p className="text-[13px] leading-relaxed text-muted-foreground">{row.description}</p>
      ) : null}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MiniStat label="Decisions" value={row.decisions} hint={`by ${row.strategy}`} />
        <MiniStat label="Signals" value={row.signals} />
        <MiniStat label="Projections" value={row.projections} />
        <MiniStat label="Target models" value={row.targets.length} />
      </div>

      <section className="flex flex-col gap-2">
        <SectionLabel>Public models</SectionLabel>
        {row.publicModels.length ? (
          <div className="flex flex-col gap-1.5">
            {row.publicModels.map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => onOpenPublicModel(name)}
                className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-border-strong hover:bg-surface-subtle"
              >
                <Layers className="size-4 text-chart-1" />
                <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] font-medium">
                  {name}
                </span>
              </button>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-border-strong">
            <EmptyState
              compact
              icon={Layers}
              tone="blue"
              title="No public model uses this recipe"
              description="Publish a model name for it so apps can call it."
              actions={
                isReadonly ? undefined : (
                  <Button variant="outline" size="sm" onClick={onNewPublicModel}>
                    <Plus />
                    New public model
                  </Button>
                )
              }
            />
          </div>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionLabel>Signal families</SectionLabel>
        {families.length ? (
          <div className="flex flex-wrap gap-1.5">
            {families.map(([type, count]) => (
              <button
                key={type}
                type="button"
                onClick={() =>
                  navigate(`/routing/signals?recipe=${scope}&type=${encodeURIComponent(type)}`)
                }
                className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-surface px-2 text-xs capitalize transition-colors hover:border-border-strong hover:bg-surface-subtle"
              >
                <span className="size-1.5 rounded-full bg-chart-2" />
                {signalTypeLabel(type)}
                <span className="font-mono text-[11px] text-subtle-foreground">{count}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="text-[13px] text-muted-foreground">This recipe defines no signals.</p>
        )}
      </section>

      {projections.length ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Projections</SectionLabel>
          <div className="flex flex-wrap gap-1.5">
            {projections.map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => navigate(`/routing/projections?recipe=${scope}`)}
                className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-surface px-2 font-mono text-xs transition-colors hover:border-border-strong hover:bg-surface-subtle"
              >
                <span className="size-1.5 rounded-full bg-chart-4" />
                {name}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <section className="flex flex-col gap-2">
        <SectionLabel>Configured targets</SectionLabel>
        {row.targets.length ? (
          <div className="flex flex-wrap gap-1.5">
            {row.targets.map((target) => {
              const model = models.find((candidate) => candidate.name === target)
              return (
                <button
                  key={target}
                  type="button"
                  onClick={() => navigate(`/models/backends?m=${encodeURIComponent(target)}`)}
                  className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-surface px-2 font-mono text-xs transition-colors hover:border-border-strong hover:bg-surface-subtle"
                >
                  {model ? <ModelLogo model={model} size={12} /> : null}
                  {target}
                </button>
              )
            })}
          </div>
        ) : (
          <p className="text-[13px] text-muted-foreground">No target models</p>
        )}
      </section>
    </div>
  )
}

function DecisionsList({ row }: { row: ConfigRecipeRow }) {
  const navigate = useNavigate()
  const lanes = decisionLanes(row.recipe)
  if (!lanes.length) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong">
        <EmptyState
          compact
          icon={Split}
          tone="violet"
          title="No decisions yet"
          description="A decision matches request signals and sends the request to its candidate models."
          actions={
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate(`/routing/decisions?recipe=${encodeURIComponent(row.id)}`)}
            >
              Open decisions
            </Button>
          }
        />
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-1.5">
      {lanes.map(({ decision, targets }) => (
        <button
          key={decision.name}
          type="button"
          onClick={() =>
            navigate(
              `/routing/decisions?recipe=${encodeURIComponent(row.id)}&id=${encodeURIComponent(decision.name)}`,
            )
          }
          className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-border-strong hover:bg-surface-subtle"
        >
          <span className="flex items-center gap-2">
            <IconTile icon={Split} tone="violet" size="xs" />
            <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] font-medium">
              {decision.name}
            </span>
            <span className="font-mono text-[11px] tabular text-subtle-foreground">
              P{decision.priority ?? 0}
            </span>
          </span>
          {decision.description ? (
            <span className="line-clamp-2 pl-8 text-xs text-muted-foreground">
              {decision.description}
            </span>
          ) : null}
          <span className="flex flex-wrap gap-1 pl-8">
            {targets.length ? (
              targets.map((target) => (
                <Badge key={target} size="sm" tone="outline" className="font-mono">
                  {target}
                </Badge>
              ))
            ) : (
              <span className="text-[11px] text-subtle-foreground">No model</span>
            )}
          </span>
        </button>
      ))}
    </div>
  )
}

function PackageDetails({ row }: { row: PackageRecipeRow }) {
  const { summary } = row
  return (
    <div className="flex flex-col gap-5">
      {summary.description ? (
        <p className="text-[13px] leading-relaxed text-muted-foreground">{summary.description}</p>
      ) : null}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MiniStat label="Public models" value={summary.counts.unified_models} />
        <MiniStat label="Recipes" value={summary.counts.recipes} />
        <MiniStat label="Decisions" value={summary.counts.decisions} />
        <MiniStat label="Probes" value={summary.counts.probes} />
      </div>
      {summary.warnings.length ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Warnings · {summary.warnings.length}</SectionLabel>
          {summary.warnings.map((warning) => (
            <div
              key={`${warning.code}:${warning.path}`}
              className="flex items-start gap-2 rounded-md bg-warning-soft px-2.5 py-2 text-xs"
            >
              <TriangleAlert className="mt-px size-3.5 shrink-0 text-warning-foreground" />
              <span className="flex min-w-0 flex-col">
                <span className="text-foreground/90">{warning.message}</span>
                <span className="font-mono text-[11px] text-muted-foreground">
                  {warning.code} · {warning.path}
                </span>
              </span>
            </div>
          ))}
        </section>
      ) : null}
      <section className="flex flex-col gap-3">
        <SectionLabel>Identity</SectionLabel>
        <KeyValueGrid
          items={[
            { label: 'Package', value: summary.id, mono: true },
            { label: 'Version', value: summary.version, mono: true },
            {
              label: 'Installed',
              value: summary.installed_at ? new Date(summary.installed_at).toLocaleString() : '—',
            },
            {
              label: 'Archive',
              value: summary.archive_verified ? 'Verified checksum' : 'Not verified',
            },
          ]}
        />
        <div className="flex flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">Recipe digest</span>
          <CopyField value={summary.recipe_digest} />
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">Config digest</span>
          <CopyField value={summary.config_digest} />
        </div>
      </section>
    </div>
  )
}

function BuiltInDetails({ row }: { row: BuiltInRecipeRow }) {
  return (
    <div className="flex flex-col gap-5">
      {row.description ? (
        <p className="text-[13px] leading-relaxed text-muted-foreground">{row.description}</p>
      ) : null}
      <KeyValueGrid
        items={[
          { label: 'Bundle', value: row.bundle, mono: true },
          { label: 'Catalog version', value: row.catalogVersion || '—', mono: true },
        ]}
      />
      <section className="flex flex-col gap-2">
        <SectionLabel>Decisions · {row.recipe.decisions.length}</SectionLabel>
        <div className="flex flex-col divide-y divide-border rounded-lg border border-border">
          {row.recipe.decisions.map((decision) => (
            <div key={decision.name} className="flex flex-col gap-0.5 px-3 py-2.5">
              <span className="flex items-center gap-2">
                <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] font-medium">
                  {decision.name}
                </span>
                <Badge size="sm" tone="outline">
                  {decision.minimum_candidates > 0
                    ? `${decision.minimum_candidates}+ ${decision.minimum_candidates === 1 ? 'model' : 'models'}`
                    : 'No model'}
                </Badge>
              </span>
              <span className="text-xs text-muted-foreground">
                {decision.description}
                {decision.algorithm ? ` · ${decision.algorithm.replace(/_/g, ' ')}` : ''}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default function ConfigPageRecipeSheet({
  row,
  open,
  view,
  onViewChange,
  onOpenChange,
  models,
  pendingRestart,
  isReadonly,
  canDeploy,
  onEdit,
  onDelete,
  onNewPublicModel,
  onOpenPublicModel,
  onToggle,
}: ConfigPageRecipeSheetProps) {
  const navigate = useNavigate()
  const { copy } = useCopy()
  if (!row) return null
  const status = recipeRowStatus(row, pendingRestart)
  const label = recipeStatusLabel[status]
  const configRow = row.kind === 'config' ? row : null
  const yaml = configRow
    ? configCollectionItemYaml(['recipes'], {
        name: configRow.name,
        ...(configRow.description ? { description: configRow.description } : {}),
        ...configRow.recipe.routing,
      })
    : ''

  return (
    <DetailSheet open={open} onOpenChange={onOpenChange} size="xl">
      <DetailSheetHeader
        icon={<IconTile icon={Package} tone="blue" size="lg" />}
        title={<span className="font-mono">{row.name}</span>}
        badges={
          <>
            <Badge
              tone={
                label.tone === 'success'
                  ? 'success'
                  : label.tone === 'warning'
                    ? 'warning'
                    : 'neutral'
              }
              dot
            >
              {label.label}
            </Badge>
            {row.kind === 'builtin' ? <Badge tone="outline">built-in</Badge> : null}
            {row.kind === 'package' ? <Badge tone="outline">package</Badge> : null}
            {configRow?.id === 'default' ? <Badge tone="brand">default</Badge> : null}
          </>
        }
        description={
          row.kind === 'config' ? (
            <>
              <span>
                {row.strategy.charAt(0).toUpperCase()}
                {row.strategy.slice(1)} strategy
              </span>
              <span className="text-subtle-foreground">·</span>
              <span>
                {row.decisions} {row.decisions === 1 ? 'decision' : 'decisions'}
              </span>
              <span className="text-subtle-foreground">·</span>
              <span>config.yaml</span>
            </>
          ) : row.kind === 'package' ? (
            <>
              <span className="font-mono">{row.summary.version}</span>
              <span className="text-subtle-foreground">·</span>
              <span className="font-mono">{row.summary.recipe_digest.slice(0, 19)}…</span>
            </>
          ) : (
            <>
              <span className="font-mono">{row.bundle}</span>
              <span className="text-subtle-foreground">·</span>
              <span>installed with vllm-sr</span>
            </>
          )
        }
        actions={
          configRow ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon-sm" aria-label="Recipe actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                <DropdownMenuItem
                  onSelect={() =>
                    navigate(`/routing/topology?scope=${encodeURIComponent(configRow.id)}`)
                  }
                >
                  <Network />
                  View topology
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() =>
                    navigate(`/routing/decisions?recipe=${encodeURIComponent(configRow.id)}`)
                  }
                >
                  <Split />
                  Open decisions
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => {
                    copy(yaml)
                    toast.success('Recipe YAML copied')
                  }}
                >
                  <Copy />
                  Copy YAML
                </DropdownMenuItem>
                {!isReadonly && !configRow.synthetic && configRow.id !== 'default' ? (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem variant="destructive" onSelect={() => onDelete(configRow)}>
                      <Trash2 />
                      Delete
                    </DropdownMenuItem>
                  </>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null
        }
      />
      {configRow ? (
        <Tabs
          value={view}
          onValueChange={(next) => onViewChange(next as RecipeSheetView)}
          className="min-h-0 flex-1 gap-0"
        >
          <TabsList className="px-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="decisions">
              Decisions
              <Badge size="sm" tone="neutral" className="tabular">
                {configRow.decisions}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="yaml">YAML</TabsTrigger>
          </TabsList>
          <DetailSheetBody>
            <TabsContent value="overview" className="px-5 py-5">
              <ConfigRecipeOverview
                row={configRow}
                models={models}
                isReadonly={isReadonly}
                onNewPublicModel={() => onNewPublicModel(configRow)}
                onOpenPublicModel={onOpenPublicModel}
              />
            </TabsContent>
            <TabsContent value="decisions" className="px-5 py-5">
              <DecisionsList row={configRow} />
            </TabsContent>
            <TabsContent value="yaml" className="px-5 py-5">
              <EntityYamlPanel
                path={`recipes[${configRow.name}]`}
                yaml={yaml}
                note="The recipe as config.yaml states it."
              />
            </TabsContent>
          </DetailSheetBody>
        </Tabs>
      ) : (
        <DetailSheetBody className="px-5 py-5">
          {row.kind === 'package' ? <PackageDetails row={row} /> : null}
          {row.kind === 'builtin' ? <BuiltInDetails row={row} /> : null}
        </DetailSheetBody>
      )}
      <DetailSheetFooter highlight={status === 'pending'}>
        <StatusLabel tone={label.tone}>
          <span className="text-xs">
            {status === 'pending'
              ? 'Saved · run vllm-sr serve to apply'
              : status === 'active'
                ? 'The Router serves this package'
                : status === 'loaded'
                  ? 'Part of the running configuration'
                  : status === 'inactive'
                    ? 'Not loaded by the Router'
                    : 'Publish it as a public model to use it'}
          </span>
        </StatusLabel>
        <div className="ml-auto flex items-center gap-2">
          {configRow ? (
            <>
              <Button variant="outline" size="sm" onClick={() => navigate('/routing/editor')}>
                <Braces />
                Open in config editor
              </Button>
              {!isReadonly && !configRow.synthetic ? (
                <Button size="sm" onClick={() => onEdit(configRow)}>
                  <Pencil />
                  Edit recipe
                </Button>
              ) : null}
            </>
          ) : null}
          {row.kind === 'package' && !isReadonly ? (
            row.summary.active ? (
              <Button
                variant="destructive-ghost"
                size="sm"
                disabled={!canDeploy || status === 'pending'}
                onClick={() => onToggle(row, 'deactivate')}
              >
                <Power />
                Deactivate
              </Button>
            ) : (
              <Button
                size="sm"
                disabled={!canDeploy || status === 'pending'}
                onClick={() => onToggle(row, 'activate')}
              >
                <Power />
                Activate
              </Button>
            )
          ) : null}
          {row.kind === 'builtin' && !isReadonly ? (
            <Button size="sm" onClick={() => onNewPublicModel(row)}>
              <Plus />
              Create public model
            </Button>
          ) : null}
        </div>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
