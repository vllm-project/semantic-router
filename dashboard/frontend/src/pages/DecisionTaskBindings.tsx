import { useState } from 'react'
import ConfigSaveWarnings from '../components/ConfigSaveWarnings'
import { configSaveWarnings, type RouterConfigWarning } from '../utils/configSaveWarnings'
import type { RouterConfig } from './dashboardPageTypes'
import { isSignalBindingKey, isUnsafeBindingKey } from './modelBindings'
import type { DecisionTaskBinding, DecisionTasks } from './useDecisionTasks'
import type { Observation } from '../utils/dashboardObservation'
import DecisionObservationStatus from './DecisionObservationStatus'
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionModelPage.module.css'

const GLOBAL_BINDINGS_PATH = ['global', 'model_catalog', 'bindings']

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

function BindingRow({
  binding,
  data,
  writable,
  save,
}: {
  binding: DecisionTaskBinding
  data: DecisionTasks
  writable: boolean
  save: (binding: DecisionTaskBinding, deployment: string | null) => Promise<unknown>
}) {
  const [editing, setEditing] = useState(false)
  const [selected, setSelected] = useState(binding.deployment)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const task = data.tasks.find((item) => item.id === binding.task_id)
  const inheritLabel =
    binding.source === 'recipe'
      ? 'Use inherited binding'
      : binding.source === 'global'
        ? 'Remove global binding'
        : undefined
  const apply = async (deployment: string | null) => {
    setBusy(true)
    setError(null)
    try {
      await save(binding, deployment)
      setEditing(false)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The task binding could not be saved.')
    } finally {
      setBusy(false)
    }
  }
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
  return (
    <tr>
      <td>
        <strong>{task?.title ?? binding.task_id}</strong>
        <small className={styles.bindingMeta}>
          {binding.recipe || 'default'} · {taskConsumer(binding)}
        </small>
      </td>
      <td>
        {editing ? (
          <>
            <SystemOneSelect
              label={`Model for ${task?.title ?? binding.task_id}`}
              value={selected}
              onChange={setSelected}
              options={options}
              disabled={busy}
            />
            {binding.source === 'global' && (
              <small className={styles.bindingMeta}>
                Changes the global binding, which every recipe without its own binding uses.
              </small>
            )}
          </>
        ) : (
          <>
            <span>{binding.model || binding.deployment || 'Unbound'}</span>
            {binding.model && binding.deployment && binding.model !== binding.deployment && (
              <small className={styles.bindingMeta}>{binding.deployment}</small>
            )}
          </>
        )}
      </td>
      <td>
        <span className={styles.sourcePill}>
          {
            {
              default: 'Default model',
              recipe: 'Recipe override',
              global: 'Global binding',
              rule: 'Decision algorithm',
            }[binding.source]
          }
        </span>
      </td>
      <td>
        <span className={binding.ready ? styles.readyText : styles.muted}>
          {binding.ready ? 'Ready' : 'Not ready'}
        </span>
      </td>
      <td>
        <div className={styles.bindingActions}>
          {writable &&
            binding.editable &&
            (editing ? (
              <>
                <button
                  type="button"
                  onClick={() => void apply(selected)}
                  disabled={busy || !options.some((item) => item.value === selected)}
                >
                  Apply
                </button>
                {inheritLabel && (
                  <button type="button" onClick={() => void apply(null)} disabled={busy}>
                    {inheritLabel}
                  </button>
                )}
                <button type="button" onClick={() => setEditing(false)} disabled={busy}>
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={() => {
                  setSelected(binding.deployment)
                  setEditing(true)
                }}
              >
                Change
              </button>
            ))}
        </div>
        {error && (
          <p role="alert" className={styles.notice}>
            {error}
          </p>
        )}
      </td>
    </tr>
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
  const saveBinding = async (binding: DecisionTaskBinding, deployment: string | null) => {
    setWarnings([])
    const result = await save(binding, deployment)
    setWarnings(configSaveWarnings(result))
    return result
  }
  return (
    <section className={styles.panel} aria-labelledby="task-bindings-title">
      <h2 id="task-bindings-title">Task bindings</h2>
      <p className={styles.muted}>
        See which model each task uses. Override a task here, or restore its default binding.
      </p>
      <DecisionObservationStatus label="Task bindings" observation={observation} />
      <ConfigSaveWarnings warnings={warnings} />
      {data?.bindings.length ? (
        <div className={styles.tableScroll}>
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Actual model</th>
                <th>Configuration source</th>
                <th>Runtime</th>
                <th>Manage</th>
              </tr>
            </thead>
            <tbody>
              {data.bindings.map((binding) => (
                <BindingRow
                  key={`${binding.recipe}:${binding.consumer}:${binding.task_id}`}
                  binding={binding}
                  data={data}
                  writable={writable}
                  save={saveBinding}
                />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className={styles.muted}>
          {data
            ? 'No active task bindings in this configuration.'
            : error
              ? 'Task bindings are unavailable.'
              : 'Loading task bindings…'}
        </p>
      )}
    </section>
  )
}
