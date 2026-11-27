import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, Link2, Plus } from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  Card,
  DataToolbar,
  EmptyState,
  FilterPill,
  ProviderLogo,
  Segmented,
  Tooltip,
  cn,
} from '../ui'
import type { DecisionDeploymentRow } from './decisionDeploymentRows'
import { CapabilityChips, ModalityIcons } from './DecisionModelChips'
import {
  DECISION_CATALOG_MODELS,
  DECISION_DEVICE_LABELS,
  formatParameters,
  type DecisionDeviceClass,
  type DecisionModelOption,
} from './decisionModelSupport'
import { CPU_RECOMMENDED_DECISION_MODELS } from './decisionModelRecommendation'
import { QUESTION_TYPE_ORDER, questionTypeLabel } from './decisionQuestionTypes'
import { DECISION_PROVIDERS } from './decisionRuntimeCatalog'

const BACKBONES: Record<string, string> = {
  qwen3: 'Qwen3',
  qwen3_5_text: 'Qwen3.5 text',
  modernbert: 'ModernBERT',
  ultrabert: 'UltraBERT',
  ultravbert: 'UltraVBERT',
}

const normalize = (value: string) =>
  value
    .toLowerCase()
    .replace(/[-\s]+/g, ' ')
    .trim()

function familySummary(models: readonly DecisionModelOption[]): string {
  const modalities = [...new Set(models.flatMap((model) => model.modalities))]
  const servable = models.filter((model) => model.servable).length
  return [
    `${models.length} ${models.length === 1 ? 'model' : 'models'}`,
    servable < models.length ? `${servable} servable` : '',
    modalities.length ? `reads ${modalities.join(', ')}` : '',
  ]
    .filter(Boolean)
    .join(' · ')
}

function ModelCard({
  model,
  index,
  deployedAs,
  isDefault,
  recommended,
  writable,
  onDeploy,
  onOpen,
}: {
  model: DecisionModelOption
  index: number
  deployedAs: string[]
  isDefault: boolean
  recommended: boolean
  writable: boolean
  onDeploy: () => void
  onOpen: (name: string) => void
}) {
  const publisher = DECISION_PROVIDERS[model.provider]
  return (
    <motion.article
      aria-label={model.label}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 8) * 0.03, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
    >
      <Card
        className={cn(
          'h-full gap-0 p-4 transition-[border-color,box-shadow] hover:border-border-strong hover:shadow-sm',
          recommended && 'border-chart-3/40',
          !model.servable && 'bg-surface-subtle/50',
        )}
      >
        <div className="flex items-start gap-3">
          <span title={publisher?.name ?? model.provider}>
            <ProviderLogo provider={publisher?.logo ?? model.provider} size={18} framed />
          </span>
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate font-mono text-[13px] font-semibold">{model.name}</span>
            <span className="truncate text-xs text-subtle-foreground">
              {[
                model.parameters === null ? '' : formatParameters(model.parameters),
                model.device ? DECISION_DEVICE_LABELS[model.device] : '',
              ]
                .filter(Boolean)
                .join(' · ') || model.family}
            </span>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1">
            {isDefault ? (
              <Badge tone="brand" size="sm">
                Default
              </Badge>
            ) : null}
            {recommended ? (
              <Badge tone="violet" size="sm">
                <Cpu />
                Recommended for CPU
              </Badge>
            ) : null}
            {!model.servable ? (
              <Badge tone="outline" size="sm">
                Not servable yet
              </Badge>
            ) : null}
          </div>
        </div>
        <p className="mt-3 line-clamp-3 min-h-[3.9em] text-xs leading-relaxed text-muted-foreground">
          {model.summary}
        </p>
        <CapabilityChips supported={model.questionTypes} className="mt-3" />
        <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-border pt-3">
          {[
            {
              label: 'Parameters',
              value: model.parameters === null ? null : formatParameters(model.parameters),
              title:
                model.parameters === null
                  ? undefined
                  : `${model.parameters.toLocaleString()} parameters`,
            },
            {
              label: 'Memory',
              value: model.minMemoryGiB === null ? null : `${model.minMemoryGiB} GiB`,
              title: 'Device memory floor',
            },
            {
              label: 'Backbone',
              value: model.backbone ? (BACKBONES[model.backbone] ?? model.backbone) : null,
            },
          ].map((fact) => (
            <div key={fact.label} className="flex min-w-0 flex-col gap-0.5">
              <dt className="text-[11px] text-subtle-foreground">{fact.label}</dt>
              <dd
                className={cn(
                  'truncate font-mono text-[12.5px] font-medium tabular',
                  fact.value === null && 'text-subtle-foreground',
                )}
                title={fact.value === null ? 'Not published yet' : fact.title}
              >
                {fact.value ?? '—'}
              </dd>
            </div>
          ))}
        </dl>
        <div className="mt-3 flex min-h-7 items-center gap-2">
          <ModalityIcons modalities={model.modalities} />
          {deployedAs.length ? (
            <Button
              variant="ghost"
              size="xs"
              className="ml-auto max-w-[60%]"
              onClick={() => onOpen(deployedAs[0])}
              aria-label={`Open deployment ${deployedAs[0]}`}
            >
              <Link2 />
              <span className="truncate font-mono">{deployedAs[0]}</span>
              {deployedAs.length > 1 ? (
                <span className="text-subtle-foreground">+{deployedAs.length - 1}</span>
              ) : null}
            </Button>
          ) : model.servable ? (
            <Tooltip
              content={
                writable
                  ? `Deploy ${model.name}`
                  : 'Configuration write access is required to deploy.'
              }
            >
              <span className="ml-auto">
                <Button
                  variant="outline"
                  size="xs"
                  disabled={!writable}
                  onClick={onDeploy}
                  aria-label={`Deploy ${model.label}`}
                >
                  <Plus />
                  Deploy
                </Button>
              </span>
            </Tooltip>
          ) : (
            <span className="ml-auto text-xs text-subtle-foreground">Answers no questions yet</span>
          )}
        </div>
      </Card>
    </motion.article>
  )
}

export default function DecisionModelCatalog({
  rows,
  recommended,
  defaultArtifact,
  writable,
  onDeploy,
  onOpenDeployment,
}: {
  rows: DecisionDeploymentRow[]
  recommended: string | null
  defaultArtifact?: string
  writable: boolean
  onDeploy: (model: DecisionModelOption) => void
  onOpenDeployment: (name: string) => void
}) {
  const [search, setSearch] = useState('')
  const [family, setFamily] = useState('all')
  const [capabilities, setCapabilities] = useState<string[]>([])
  const [devices, setDevices] = useState<string[]>([])
  const families = useMemo(
    () => [...new Set(DECISION_CATALOG_MODELS.map((model) => model.family))],
    [],
  )
  const query = normalize(search)
  const matches = DECISION_CATALOG_MODELS.filter(
    (model) =>
      (family === 'all' || model.family === family) &&
      capabilities.every((capability) => model.questionTypes.includes(capability)) &&
      (devices.length === 0 || (model.device !== null && devices.includes(model.device))) &&
      normalize(`${model.label} ${model.name} ${model.artifact} ${model.family}`).includes(query),
  )
  const groups = families
    .map((name) => ({
      name,
      all: DECISION_CATALOG_MODELS.filter((model) => model.family === name),
      models: matches
        .filter((model) => model.family === name)
        .sort(
          (left, right) => Number(right.name === recommended) - Number(left.name === recommended),
        ),
    }))
    .filter((group) => group.models.length)
  const deployedAs = (artifact: string) =>
    rows.filter((row) => row.artifact === artifact).map((row) => row.name)
  const clear = () => {
    setSearch('')
    setFamily('all')
    setCapabilities([])
    setDevices([])
  }
  let index = 0
  return (
    <section
      id="decision-catalog"
      className="flex scroll-mt-6 flex-col gap-3"
      aria-labelledby="decision-catalog-title"
    >
      <div className="flex flex-col gap-0.5">
        <h2 id="decision-catalog-title" className="text-[15px] font-semibold tracking-[-0.012em]">
          Catalog
        </h2>
        <p className="text-[13px] text-muted-foreground">
          Every decision model the runtime registry pins, grouped by family. Deploy one to give some
          questions their own model, or make it the default.
        </p>
      </div>
      {recommended ? (
        <Alert tone="info" title="Decision models run on a CPU on this instance" icon={<Cpu />}>
          Recommended for CPU: <span className="font-mono text-foreground">{recommended}</span>.
          {recommended !== CPU_RECOMMENDED_DECISION_MODELS[0]
            ? ` The recommendation moves to ${CPU_RECOMMENDED_DECISION_MODELS[0]} once the runtime serves it.`
            : null}
        </Alert>
      ) : null}
      <DataToolbar
        className="mb-1"
        search={search}
        onSearchChange={setSearch}
        placeholder="Search decision models…"
        filters={
          <>
            <Segmented
              aria-label="Model family"
              value={family}
              onValueChange={setFamily}
              options={[
                { value: 'all', label: 'All' },
                ...families.map((value) => ({ value, label: value })),
              ]}
            />
            <FilterPill
              label="Answers"
              selected={capabilities}
              onChange={setCapabilities}
              options={QUESTION_TYPE_ORDER.map((type) => ({
                value: type,
                label: questionTypeLabel(type),
                count: DECISION_CATALOG_MODELS.filter((model) => model.questionTypes.includes(type))
                  .length,
              }))}
            />
            <FilterPill
              label="Device"
              selected={devices}
              onChange={setDevices}
              options={(['cpu', 'flexible', 'gpu'] as DecisionDeviceClass[]).map((device) => ({
                value: device,
                label: DECISION_DEVICE_LABELS[device],
                count: DECISION_CATALOG_MODELS.filter((model) => model.device === device).length,
              }))}
            />
          </>
        }
        actions={
          <span className="text-xs text-muted-foreground tabular">
            {matches.length} of {DECISION_CATALOG_MODELS.length} models
          </span>
        }
      />
      {groups.length ? (
        <div className="flex flex-col gap-6">
          {groups.map((group) => (
            <section
              key={group.name}
              className="flex flex-col gap-2.5"
              aria-label={`${group.name} models`}
            >
              <div className="flex items-baseline gap-2">
                <h3 className="text-[13px] font-semibold">{group.name}</h3>
                <span className="text-xs text-subtle-foreground">{familySummary(group.all)}</span>
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                {group.models.map((model) => (
                  <ModelCard
                    key={model.artifact}
                    model={model}
                    index={index++}
                    deployedAs={deployedAs(model.artifact)}
                    isDefault={model.artifact === defaultArtifact}
                    recommended={model.name === recommended}
                    writable={writable}
                    onDeploy={() => onDeploy(model)}
                    onOpen={onOpenDeployment}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-border-strong">
          <EmptyState
            compact
            icon={Cpu}
            tone="violet"
            title="No decision model matches"
            description="Set questions need Vela 2.0 and Span questions a Vela model; clear a filter to see every family."
            actions={
              <Button variant="outline" size="sm" onClick={clear}>
                Clear filters
              </Button>
            }
          />
        </div>
      )}
    </section>
  )
}
