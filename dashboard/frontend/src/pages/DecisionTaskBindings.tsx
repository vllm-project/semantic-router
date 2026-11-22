import { useState } from 'react'
import { ListChecks, Pencil } from 'lucide-react'
import ConfigSaveWarnings from '../components/ConfigSaveWarnings'
import {
  Alert,
  Badge,
  Button,
  DataTable,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DropdownMenuItem,
  EmptyState,
  Field,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  StatusLabel,
  toast,
  type Column,
} from '../ui'
import { configSaveWarnings, type RouterConfigWarning } from '../utils/configSaveWarnings'
import type { Observation } from '../utils/dashboardObservation'
import type { RouterConfig } from './dashboardPageTypes'
import DecisionObservationStatus from './DecisionObservationStatus'
import { isSignalBindingKey, isUnsafeBindingKey } from './modelBindings'
import type { DecisionTaskBinding, DecisionTasks } from './useDecisionTasks'

const GLOBAL_BINDINGS_PATH = ['global', 'model_catalog', 'bindings']

const SOURCE_LABELS: Record<DecisionTaskBinding['source'], string> = {
  default: 'Default model',
  recipe: 'Recipe override',
  global: 'Global binding',
  rule: 'Decision algorithm',
}

/** The key a task's model is bound under: its signal type, else the Router's consumer name. */
function taskConsumer(binding: DecisionTaskBinding): string {
  const path = binding.path ?? []
  return path[path.length - 1] || binding.consumer
}

// The task catalog reports the binding's recipe; the default recipe is the
// routing profile of the config view.
function routingPath(config: RouterConfig, binding: DecisionTaskBinding): string[] {
  const written = binding.path ?? []
  const path = written.slice(0, -1)
  if (
    path[0] !== 'recipes' ||
    path[path.length - 1] !== 'model_bindings' ||
    written.some(isUnsafeBindingKey)
  )
    throw new Error('This task does not expose an editable binding.')
  const index = config.recipes?.findIndex((recipe) => recipe.name === binding.recipe) ?? -1
  if (index >= 0) return ['recipes', String(index), 'routing']
  if (binding.recipe === (config.default_recipe?.name ?? 'default')) return ['routing']
  throw new Error('This recipe was removed. Refresh the task list.')
}

function objectAt(root: unknown, path: readonly string[]): Record<string, unknown> {
  let target = root as Record<string, unknown>
  for (const part of path) {
    if (target[part] === undefined) target[part] = {}
    if (!target[part] || typeof target[part] !== 'object')
      throw new Error('The binding configuration changed. Refresh before editing.')
    target = target[part] as Record<string, unknown>
  }
  return target
}

/**
 * Names the deployment that answers a task, or removes it (null) so the task
 * inherits: the binding of its signal type, in the row's recipe or, for a
 * row the global block binds, there.
 */
export function withDecisionTaskBinding(
  config: RouterConfig,
  binding: DecisionTaskBinding,
  deployment: string | null,
): RouterConfig {
  const next = structuredClone(config)
  const key = taskConsumer(binding)
  if (!isSignalBindingKey(key)) throw new Error('This task does not expose an editable binding.')
  const target = objectAt(
    next,
    binding.source === 'global'
      ? GLOBAL_BINDINGS_PATH
      : [...routingPath(next, binding), 'model_bindings'],
  )
  if (deployment === null) delete target[key]
  else target[key] = deployment
  return next
}

const bindingId = (binding: DecisionTaskBinding) =>
  `${binding.recipe}:${binding.consumer}:${binding.task_id}`

function BindingDialog({
  binding,
  data,
  onOpenChange,
  save,
}: {
  binding: DecisionTaskBinding | null
  data: DecisionTasks
  onOpenChange: (open: boolean) => void
  save: (binding: DecisionTaskBinding, deployment: string | null) => Promise<unknown>
}) {
  const [selected, setSelected] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [opened, setOpened] = useState<string | null>(null)
  if (binding && opened !== bindingId(binding)) {
    setOpened(bindingId(binding))
    setSelected(binding.deployment)
    setError(null)
  }
  if (!binding) return null
  const task = data.tasks.find((item) => item.id === binding.task_id)
  const title = task?.title ?? binding.task_id
  const inheritLabel =
    binding.source === 'recipe'
      ? 'Use inherited binding'
      : binding.source === 'global'
        ? 'Remove global binding'
        : undefined
  const options = data.deployments
    .filter((item) =>
      item.tasks.some(
        (capability) => capability.task_id === binding.task_id && capability.supported,
      ),
    )
    .map((item) => ({
      value: item.deployment,
      label: item.model || item.deployment,
      description: [
        item.model && item.model !== item.deployment ? item.deployment : '',
        item.ready ? 'Ready' : 'Starts when deployed',
      ]
        .filter(Boolean)
        .join(' · '),
    }))
  const apply = async (deployment: string | null) => {
    setBusy(true)
    setError(null)
    try {
      await save(binding, deployment)
      onOpenChange(false)
      toast.success(
        deployment === null
          ? `${title} now inherits its binding`
          : `${title} now uses ${deployment}`,
      )
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The task binding could not be saved.')
    } finally {
      setBusy(false)
    }
  }
  return (
    <Dialog open onOpenChange={(open) => (busy ? undefined : onOpenChange(open))}>
      <DialogContent size="md">
        <DialogHeader>
          <DialogTitle>Change the model for {title}</DialogTitle>
          <DialogDescription>
            <span className="font-mono text-xs">
              {binding.recipe || 'default'} · {taskConsumer(binding)}
            </span>
          </DialogDescription>
        </DialogHeader>
        <DialogBody className="flex flex-col gap-4 pb-5">
          <Field
            label="Deployment"
            hint={
              binding.source === 'global'
                ? 'Changes the global binding, which every recipe without its own binding uses.'
                : 'Only deployments that support this task are listed.'
            }
          >
            <Select value={selected} onValueChange={setSelected} disabled={busy}>
              <SelectTrigger className="w-full" aria-label={`Model for ${title}`}>
                <SelectValue placeholder="Choose a deployment" />
              </SelectTrigger>
              <SelectContent>
                {options.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    <span className="font-mono text-[12.5px]">{option.label}</span>
                    <span className="ml-2 text-xs text-subtle-foreground">
                      {option.description}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          {error ? (
            <div role="alert">
              <Alert tone="danger" title="The binding was not saved">
                {error}
              </Alert>
            </div>
          ) : null}
        </DialogBody>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          {inheritLabel ? (
            <Button variant="outline" onClick={() => void apply(null)} disabled={busy}>
              {inheritLabel}
            </Button>
          ) : null}
          <Button
            onClick={() => void apply(selected)}
            loading={busy}
            disabled={!options.some((option) => option.value === selected)}
          >
            Apply
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default function DecisionTaskBindings({
  data,
  error,
  observation,
  writable,
  save,
}: {
  data: DecisionTasks | null
  error: string | null
  observation: Observation<DecisionTasks>
  writable: boolean
  save: (binding: DecisionTaskBinding, deployment: string | null) => Promise<unknown>
}) {
  const [warnings, setWarnings] = useState<RouterConfigWarning[]>([])
  const [editing, setEditing] = useState<DecisionTaskBinding | null>(null)
  const saveBinding = async (binding: DecisionTaskBinding, deployment: string | null) => {
    setWarnings([])
    const result = await save(binding, deployment)
    setWarnings(configSaveWarnings(result))
    return result
  }
  const editable = (binding: DecisionTaskBinding) => writable && binding.editable
  const columns: Column<DecisionTaskBinding>[] = [
    {
      id: 'task',
      header: 'Task',
      sortValue: (binding) =>
        data?.tasks.find((item) => item.id === binding.task_id)?.title ?? binding.task_id,
      cell: (binding) => (
        <div className="flex min-w-0 flex-col">
          <span className="truncate text-[13px] font-medium">
            {data?.tasks.find((item) => item.id === binding.task_id)?.title ?? binding.task_id}
          </span>
          <span className="truncate font-mono text-[11.5px] text-subtle-foreground">
            {binding.recipe || 'default'} · {taskConsumer(binding)}
          </span>
        </div>
      ),
    },
    {
      id: 'model',
      header: 'Actual model',
      cell: (binding) => (
        <div className="flex min-w-0 flex-col">
          <span className="truncate font-mono text-[12.5px]">
            {binding.model || binding.deployment || 'Unbound'}
          </span>
          {binding.model && binding.deployment && binding.model !== binding.deployment ? (
            <span className="truncate font-mono text-[11.5px] text-subtle-foreground">
              {binding.deployment}
            </span>
          ) : null}
        </div>
      ),
    },
    {
      id: 'source',
      header: 'Configuration source',
      hideBelow: 'md',
      cell: (binding) => (
        <Badge tone="outline" size="sm">
          {SOURCE_LABELS[binding.source]}
        </Badge>
      ),
    },
    {
      id: 'runtime',
      header: 'Runtime',
      cell: (binding) => (
        <StatusLabel tone={binding.ready ? 'success' : 'neutral'}>
          {binding.ready ? 'Ready' : 'Not ready'}
        </StatusLabel>
      ),
    },
  ]
  return (
    <section className="flex flex-col gap-3" aria-labelledby="task-bindings-title">
      <div className="flex flex-col gap-0.5">
        <h2 id="task-bindings-title" className="text-[15px] font-semibold tracking-[-0.012em]">
          Task bindings
        </h2>
        <p className="text-[13px] text-muted-foreground">
          The model each built-in task uses. Override a task, or restore its inherited binding.
        </p>
      </div>
      <DecisionObservationStatus label="Task bindings" observation={observation} />
      <ConfigSaveWarnings warnings={warnings} />
      <DataTable
        data={data?.bindings ?? []}
        columns={columns}
        loading={!data && observation.loading}
        getRowId={bindingId}
        onRowClick={(binding) => (editable(binding) ? setEditing(binding) : undefined)}
        rowActions={
          writable && data?.bindings.some((binding) => binding.editable)
            ? (binding) => (
                <DropdownMenuItem
                  disabled={!editable(binding)}
                  onSelect={() => setEditing(binding)}
                >
                  <Pencil />
                  Change model
                </DropdownMenuItem>
              )
            : undefined
        }
        empty={
          <EmptyState
            compact
            icon={ListChecks}
            tone="violet"
            title={
              data
                ? 'No active task bindings in this configuration.'
                : error
                  ? 'Task bindings are unavailable.'
                  : 'Loading task bindings…'
            }
            description={
              data
                ? 'Built-in signals that ask a decision model appear here once a recipe uses them.'
                : error
                  ? 'Refresh to try again; the Router reports task bindings when it is running.'
                  : undefined
            }
          />
        }
      />
      {data ? (
        <BindingDialog
          binding={editing}
          data={data}
          onOpenChange={(open) => (open ? undefined : setEditing(null))}
          save={saveBinding}
        />
      ) : null}
    </section>
  )
}
