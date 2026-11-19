import { useNavigate } from 'react-router-dom'
import { Activity, Crown, Link2, MessagesSquare } from 'lucide-react'
import {
  Badge,
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  KeyValueGrid,
  MiniStat,
  ProviderLogo,
  SectionLabel,
  useRetainedValue,
} from '../ui'
import type { DecisionDeploymentRow } from './decisionDeploymentRows'
import { CapabilityChips } from './DecisionModelChips'
import { questionTypeLabel } from './decisionQuestionTypes'
import DecisionReplicaPanel from './DecisionReplicaPanel'
import { decisionStageStatus } from './decisionModelManagement'
import DecisionStatusTrack from './DecisionStatusTrack'
import type { DecisionModelManagement } from './useDecisionModelManagement'

const shown = (value?: string | number) =>
  value === undefined || value === '' ? 'Not reported' : String(value)

export default function DecisionDeploymentSheet({
  row: current,
  model,
  writable,
  onOpenChange,
  onBind,
  onMakeDefault,
}: {
  row: DecisionDeploymentRow | undefined
  model: DecisionModelManagement
  writable: boolean
  onOpenChange: (open: boolean) => void
  onBind: (row: DecisionDeploymentRow) => void
  onMakeDefault: (row: DecisionDeploymentRow) => void
}) {
  const navigate = useNavigate()
  const row = useRetainedValue(current)
  if (!row) return null
  const status = decisionStageStatus(row.progress)
  const observed = row.observed
  return (
    <DetailSheet open={Boolean(current)} onOpenChange={onOpenChange} size="lg">
      <DetailSheetHeader
        icon={<ProviderLogo provider="vllm" size={22} framed />}
        title={<span className="font-mono">{row.name}</span>}
        badges={
          <>
            {row.isDefault ? (
              <Badge tone="brand" size="sm">
                Default
              </Badge>
            ) : null}
            <Badge tone={status.tone} size="sm" dot>
              {status.label}
            </Badge>
          </>
        }
        description={
          <>
            <span className="font-mono">{row.modelName}</span>
            {row.artifact && row.artifact !== row.modelName ? (
              <>
                <span className="text-subtle-foreground">·</span>
                <span className="font-mono">{row.artifact}</span>
              </>
            ) : null}
            <span className="text-subtle-foreground">·</span>
            <span>
              {row.kind === 'systemone'
                ? 'System One API'
                : row.kind === 'observed'
                  ? 'observed only'
                  : 'model runtime'}
            </span>
          </>
        }
      />
      <DetailSheetBody className="flex flex-col gap-6 px-5 py-5">
        <DecisionStatusTrack progress={row.progress} />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MiniStat
            label="Readiness"
            value={
              row.replicas.ready === null ? (
                <span className="text-muted-foreground">unknown</span>
              ) : (
                `${row.replicas.ready}/${row.replicas.desired}`
              )
            }
            hint="replicas ready"
          />
          <MiniStat
            label="Device"
            value={<span className="font-mono text-base">{row.device || '—'}</span>}
          />
          <MiniStat
            label="Engine"
            value={<span className="font-mono text-base">{observed?.engine || '—'}</span>}
          />
          <MiniStat label="Restarts" value={observed?.restarts ?? '—'} />
        </div>
        <section className="flex flex-col gap-2">
          <SectionLabel>Serves</SectionLabel>
          {row.isDefault || row.consumers.length || row.taskBindings.length ? (
            <ul className="overflow-hidden rounded-lg border border-border">
              {row.isDefault ? (
                <li className="flex items-center gap-3 border-b border-border px-3 py-2 text-[13px] last:border-0">
                  <span className="flex-1">Every question without its own binding</span>
                  <Badge tone="brand" size="sm">
                    Default
                  </Badge>
                </li>
              ) : null}
              {row.consumers.map((consumer) => (
                <li
                  key={consumer.id}
                  className="flex items-center gap-3 border-b border-border px-3 py-2 last:border-0"
                >
                  <span className="flex min-w-0 flex-1 items-center gap-1.5">
                    {consumer.recipe ? (
                      <span className="shrink-0 text-xs text-subtle-foreground">
                        {consumer.recipe} ·
                      </span>
                    ) : null}
                    <span className="truncate font-mono text-[12.5px]">{consumer.name}</span>
                  </span>
                  <Badge tone="outline" size="sm">
                    {consumer.kind === 'question'
                      ? questionTypeLabel(consumer.questionType)
                      : 'Selector'}
                  </Badge>
                </li>
              ))}
              {row.taskBindings.map((binding) => (
                <li
                  key={`${binding.recipe}:${binding.consumer}:${binding.task_id}`}
                  className="flex items-center gap-3 border-b border-border px-3 py-2 last:border-0"
                >
                  <span className="flex min-w-0 flex-1 items-center gap-1.5">
                    <span className="shrink-0 text-xs text-subtle-foreground">
                      {binding.recipe} ·
                    </span>
                    <span className="truncate font-mono text-[12.5px]">{binding.consumer}</span>
                  </span>
                  <Badge tone="outline" size="sm">
                    Task binding
                  </Badge>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-muted-foreground">
              Not bound: nothing starts until a signal, a decision or the default binding uses it.
            </p>
          )}
        </section>
        {row.questionTypes ? (
          <section className="flex flex-col gap-2">
            <SectionLabel>Answers</SectionLabel>
            <CapabilityChips supported={row.questionTypes} />
          </section>
        ) : null}
        <DecisionReplicaPanel key={row.name} model={model} row={row} writable={writable} />
        <section className="flex flex-col gap-3">
          <SectionLabel>Runtime details</SectionLabel>
          {observed?.reason ? (
            <p className="text-xs text-muted-foreground">{observed.reason}</p>
          ) : null}
          <KeyValueGrid
            items={[
              { label: 'Deployment', value: row.name, mono: true },
              { label: 'Repository', value: shown(observed?.repo ?? row.artifact), mono: true },
              {
                label: 'Revision',
                value: shown(observed?.revision ?? row.declaration?.revision),
                mono: true,
              },
              { label: 'Profile', value: shown(observed?.profile) },
              { label: 'Family', value: shown(observed?.family ?? row.model?.familyId) },
              { label: 'Process', value: shown(observed?.process), mono: true },
              { label: 'Served model', value: shown(observed?.served_name), mono: true },
              {
                label: 'Ownership',
                value: observed
                  ? observed.managed
                    ? 'Managed worker'
                    : 'External deployment'
                  : 'Not reported',
              },
              {
                label: 'Supported APIs',
                value: observed?.surfaces?.length ? observed.surfaces.join(', ') : 'Not reported',
              },
              {
                label: 'Model heads',
                value: observed?.heads?.length
                  ? observed.heads.map((head) => head.name).join(', ')
                  : 'Not reported',
              },
            ]}
          />
        </section>
      </DetailSheetBody>
      <DetailSheetFooter>
        <Button
          variant="ghost"
          size="sm"
          disabled={!row.progress.stage || row.progress.stage !== 'ready'}
          onClick={() =>
            navigate(
              `/models/decision?tab=playground&target=${encodeURIComponent(`deployment:${row.name}`)}`,
            )
          }
        >
          <MessagesSquare />
          Try it
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() =>
            navigate(`/models/decision?tab=monitoring&deployment=${encodeURIComponent(row.name)}`)
          }
        >
          <Activity />
          Monitoring
        </Button>
        <div className="ml-auto flex items-center gap-2">
          {!row.isDefault && row.declaration ? (
            <Button
              variant="outline"
              size="sm"
              disabled={!writable}
              onClick={() => onMakeDefault(row)}
            >
              <Crown />
              Make default
            </Button>
          ) : null}
          {row.declaration ? (
            <Button size="sm" disabled={!writable} onClick={() => onBind(row)}>
              <Link2 />
              Bind consumers
            </Button>
          ) : null}
        </div>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
