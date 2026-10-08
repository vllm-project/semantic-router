import { useState } from 'react'
import type { RouterConfig } from './dashboardPageTypes'
import type { DecisionTaskBinding, DecisionTasks } from './useDecisionTasks'
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionModelPage.module.css'

export function withDecisionTaskBinding(
  config: RouterConfig,
  binding: DecisionTaskBinding,
  deployment: string | null,
): RouterConfig {
  const next = structuredClone(config)
  const path = [...binding.path]
  if (path[0] === 'recipes') {
    const index = next.recipes?.findIndex((recipe) => recipe.name === binding.recipe) ?? -1
    if (index < 0) throw new Error('This recipe was removed. Refresh the task list.')
    path[1] = String(index)
  }
  if (
    !['routing', 'recipes', 'global'].includes(path[0]) ||
    !path.includes('model_bindings') ||
    path.some((part) => ['__proto__', 'constructor', 'prototype'].includes(part))
  )
    throw new Error('This task does not expose an editable binding.')
  let target: Record<string, unknown> = next as unknown as Record<string, unknown>
  for (const part of path.slice(0, -1)) {
    if (target[part] === undefined) target[part] = {}
    if (!target[part] || typeof target[part] !== 'object')
      throw new Error('The binding configuration changed. Refresh before editing.')
    target = target[part] as Record<string, unknown>
  }
  const name = path[path.length - 1]
  if (deployment === null) delete target[name]
  // The chooser offers deployments that implement this task through the
  // decision contract. Specialist heads/adapters belong to the old binding.
  else target[name] = { deployment, contract: 'decision.v1' }
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
      description: `${item.deployment} · ${item.ready ? 'Ready' : 'Starts when deployed'}`,
    }))
  return (
    <tr>
      <td>
        <strong>{task?.title ?? binding.task_id}</strong>
        <small className={styles.bindingMeta}>
          {binding.recipe || 'default'} · {binding.consumer}
        </small>
      </td>
      <td>
        {editing ? (
          <SystemOneSelect
            label={`Model for ${task?.title ?? binding.task_id}`}
            value={selected}
            onChange={setSelected}
            options={options}
            disabled={busy}
          />
        ) : (
          <>
            <span>{binding.model || binding.deployment || 'Unbound'}</span>
            <small className={styles.bindingMeta}>{binding.deployment}</small>
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
              module: 'Specialist binding',
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
                {binding.source === 'recipe' && (
                  <button type="button" onClick={() => void apply(null)} disabled={busy}>
                    Use inherited binding
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
  writable,
  save,
}: {
  data: DecisionTasks | null
  error: string | null
  writable: boolean
  save: (binding: DecisionTaskBinding, deployment: string | null) => Promise<unknown>
}) {
  return (
    <section className={styles.panel} aria-labelledby="task-bindings-title">
      <h2 id="task-bindings-title">Task bindings</h2>
      <p className={styles.muted}>
        See which model each task uses. Override a task here, or restore its default binding.
      </p>
      {error && (
        <p className={styles.notice} role="alert">
          {error}
        </p>
      )}
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
                  save={save}
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
