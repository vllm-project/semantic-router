import { useState } from 'react'
import { DECISION_MODEL_OPTIONS } from './decisionModelSupport'
import { DECISION_PROVIDERS } from './decisionRuntimeCatalog'
import type { useDecisionModelManagement } from './useDecisionModelManagement'
import type { DecisionTasks } from './useDecisionTasks'
import type { InstanceMode } from './useInstanceDeployment'
import SystemOneSelect from './SystemOneSelect'
import styles from './DecisionModelCatalog.module.css'
import pageStyles from './DecisionModelPage.module.css'

interface Props {
  model: ReturnType<typeof useDecisionModelManagement>
  writable: boolean
  mode: InstanceMode
  onModeChange: (mode: InstanceMode) => void
  canSwitch: boolean
  busy: boolean
  onDeploy: () => void
  tasks: DecisionTasks | null
}
const pageSize = 6
export default function DecisionModelCatalog({
  model,
  writable,
  mode,
  onModeChange,
  canSwitch,
  busy,
  onDeploy,
  tasks,
}: Props) {
  const [search, setSearch] = useState('')
  const [family, setFamily] = useState('all')
  const [capability, setCapability] = useState('all')
  const [provider, setProvider] = useState('all')
  const [page, setPage] = useState(1)
  const query = search.trim().toLowerCase()
  const options = DECISION_MODEL_OPTIONS.filter(
    (option) =>
      (family === 'all' || family === option.family) &&
      (provider === 'all' || provider === option.provider) &&
      (capability === 'all' || option.questionTypes.includes(capability)) &&
      `${option.label} ${option.provider} ${option.family}`.toLowerCase().includes(query),
  )
  const totalPages = Math.max(1, Math.ceil(options.length / pageSize))
  const currentPage = Math.min(page, totalPages)
  const selected = DECISION_MODEL_OPTIONS.find((option) => option.name === model.selectedModel)
  const disabled = !writable || busy || model.deploying || !model.global
  const projection = tasks?.models?.find((item) => item.model === selected?.artifact)
  const updateFilter = (setter: (value: string) => void) => (value: string) => {
    setter(value)
    setPage(1)
  }
  return (
    <div className={styles.catalog}>
      <div className={styles.catalogHeader}>
        <div>
          <span className={styles.eyebrow}>Model library</span>
          <h2 id="decision-model-choose-title">Choose a decision model</h2>
          <p className={styles.help}>
            Select a model and instance mode. Explicit task bindings remain unchanged.
          </p>
        </div>
        <span className={styles.catalogCount}>{options.length} models</span>
      </div>
      <div className={styles.modePicker} role="group" aria-label="Instance mode">
        {(['router', 'engine'] as const).map((value) => (
          <button
            type="button"
            key={value}
            aria-pressed={mode === value}
            disabled={!canSwitch || disabled}
            onClick={() => onModeChange(value)}
          >
            <strong>{value === 'router' ? 'Router mode' : 'Engine mode'}</strong>
            <span>
              {value === 'router'
                ? 'Route Chat requests and serve System One questions.'
                : 'Serve System One questions. Retain routing configuration for later.'}
            </span>
          </button>
        ))}
      </div>
      <div className={styles.filters}>
        <label className={styles.searchField}>
          <span>Search models</span>
          <input
            placeholder="Search models or providers…"
            value={search}
            onChange={(event) => updateFilter(setSearch)(event.target.value)}
          />
        </label>
        <SystemOneSelect
          label="Model provider"
          value={provider}
          onChange={updateFilter(setProvider)}
          options={[
            { value: 'all', label: 'All providers' },
            ...Object.entries(DECISION_PROVIDERS).map(([value, item]) => ({
              value,
              label: item.name,
            })),
          ]}
        />
        <SystemOneSelect
          label="Question capability"
          value={capability}
          onChange={updateFilter(setCapability)}
          options={[
            { value: 'all', label: 'All capabilities' },
            ...['choice', 'score', 'noul', 'span', 'set'].map((value) => ({ value, label: value })),
          ]}
        />
      </div>
      <div className={styles.familyTabs} aria-label="Model families">
        {['all', ...new Set(DECISION_MODEL_OPTIONS.map((option) => option.family))].map((value) => (
          <button
            key={value}
            type="button"
            aria-pressed={family === value}
            onClick={() => updateFilter(setFamily)(value)}
          >
            {value === 'all' ? 'All families' : value}
          </button>
        ))}
      </div>
      <fieldset className={styles.cards} disabled={disabled}>
        <legend className={pageStyles.srOnly}>Decision model selection</legend>
        {options.slice((currentPage - 1) * pageSize, currentPage * pageSize).map((option) => {
          const publisher = DECISION_PROVIDERS[option.provider]
          return (
            <label
              key={option.name}
              className={`${styles.card} ${model.selectedModel === option.name ? styles.selected : ''}`}
            >
              <div className={styles.cardTop}>
                {publisher ? (
                  <img className={styles.logo} src={publisher.logo} alt={publisher.name} />
                ) : (
                  <span className={styles.logoFallback}>{option.provider.slice(0, 2)}</span>
                )}
                <input
                  type="radio"
                  name="decision-model"
                  value={option.name}
                  checked={model.selectedModel === option.name}
                  onChange={() => model.selectModel(option.name)}
                />
              </div>
              <strong className={styles.modelName}>{option.label}</strong>
              <span className={styles.description}>{option.summary}</span>
              <div className={styles.capabilities}>
                {option.questionTypes.map((kind) => (
                  <span key={kind}>{kind}</span>
                ))}
              </div>
              <span className={styles.hardware}>{option.hardware}</span>
              <span className={styles.cardState}>
                {model.savedModel === option.name
                  ? 'Saved selection'
                  : model.selectedModel === option.name
                    ? 'Selected · not deployed'
                    : 'Available'}
              </span>
            </label>
          )
        })}
      </fieldset>
      {!options.length && <p className={styles.help}>No models match these filters.</p>}
      <nav className={styles.pagination} aria-label="Decision model pagination">
        <span>
          {options.length
            ? `${(currentPage - 1) * pageSize + 1}–${Math.min(currentPage * pageSize, options.length)}`
            : '0'}{' '}
          of {options.length}
        </span>
        <div>
          <button
            type="button"
            disabled={currentPage <= 1}
            onClick={() => setPage(currentPage - 1)}
          >
            Previous
          </button>
          <span>
            Page {currentPage} of {totalPages}
          </span>
          <button
            type="button"
            disabled={currentPage >= totalPages}
            onClick={() => setPage(currentPage + 1)}
          >
            Next
          </button>
        </div>
      </nav>
      <div className={styles.selectedPreview}>
        <div>
          <span className={styles.eyebrow}>Ready to deploy</span>
          <strong>{selected?.label ?? model.selectedModel ?? 'Choose a model'}</strong>
          <p className={styles.help}>
            {mode === 'router' ? 'Router' : 'Engine'} mode · Native questions:{' '}
            {selected?.questionTypes.join(', ') || 'Not reported'}
          </p>
        </div>
        {projection && (
          <div className={styles.taskPreview}>
            {projection.tasks.map((task) => (
              <span key={task.task_id} title={task.reason} data-supported={task.supported}>
                {tasks?.tasks.find((item) => item.id === task.task_id)?.title ?? task.task_id}
                {task.implementation === 'composed_noul' ? ' · composed' : ''}
                {!task.supported ? ' · unavailable' : ''}
              </span>
            ))}
          </div>
        )}
        <p className={styles.help}>
          Capability support does not imply evaluated accuracy. Test your tasks in Decision
          Playground.
        </p>
      </div>
      <div className={styles.selectionBar}>
        <span>
          {writable ? (
            <>
              <strong>{selected?.label ?? model.selectedModel}</strong>
              <small>
                {mode === 'engine'
                  ? 'Chat routing pauses; your Router configuration is retained. Native API access follows the configured listener model grants.'
                  : 'Tasks using the default will use this model.'}
              </small>
            </>
          ) : (
            'Configuration write access is required to deploy.'
          )}
        </span>
        <button
          type="button"
          className={pageStyles.primary}
          disabled={disabled || !selected}
          onClick={onDeploy}
        >
          {busy || model.deploying ? 'Deploying…' : 'Deploy selection'}
        </button>
      </div>
    </div>
  )
}
