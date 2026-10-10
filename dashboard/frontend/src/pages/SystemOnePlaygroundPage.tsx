import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { canRunEvaluation } from '../utils/accessControl'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import SystemOneResults, { JSONInspector } from './SystemOneResults'
import SystemOneSelect from './SystemOneSelect'
import {
  buildSystemOneRequest,
  EXAMPLE_STATES,
  exampleDrafts,
  newQuestion,
  QUESTION_TYPES,
  type QuestionDraft,
  type QuestionType,
  type SystemOneQuestion,
} from './systemOnePlayground'
import { useSystemOnePlayground } from './useSystemOnePlayground'
import { useDecisionTasks } from './useDecisionTasks'
import { systemOneDeploymentOption } from './systemOneDeploymentPresentation'
import styles from './SystemOnePlaygroundPage.module.css'

function QuestionEditor({
  draft,
  supported,
  onChange,
}: {
  draft: QuestionDraft
  supported: string[] | undefined
  onChange: (next: QuestionDraft) => void
}) {
  const question = draft.question
  const update = (patch: Partial<SystemOneQuestion>) =>
    onChange({ ...draft, question: { ...question, ...patch } })
  const options = question.type === 'choice' ? question.choices : question.labels
  const optionsField = question.type === 'choice' ? 'choices' : 'labels'
  const optionNoun = question.type === 'choice' ? 'option' : 'label'
  return (
    <div className={styles.questionEditor}>
      <div className={styles.typeSelector} aria-label="Question type">
        {QUESTION_TYPES.map(({ type, label, symbol }) => (
          <button
            type="button"
            key={type}
            aria-pressed={question.type === type}
            disabled={supported !== undefined && !supported.includes(type)}
            title={
              supported !== undefined && !supported.includes(type)
                ? 'This runtime does not support this question type.'
                : undefined
            }
            onClick={() => {
              if (question.type === type) return
              onChange({
                ...draft,
                variants: { ...draft.variants, [question.type]: question },
                question: draft.variants?.[type] ?? newQuestion(type).question,
              })
            }}
          >
            <span aria-hidden="true">{symbol}</span>
            {label}
          </button>
        ))}
      </div>
      <p className={styles.typeDescription}>
        {QUESTION_TYPES.find((item) => item.type === question.type)?.detail}
      </p>
      <label className={styles.field}>
        Question name
        <input
          value={draft.name}
          onChange={(event) => onChange({ ...draft, name: event.target.value })}
          placeholder="e.g. task"
          spellCheck={false}
        />
      </label>
      <label className={styles.field}>
        Instructions
        <textarea
          rows={3}
          value={question.instructions}
          onChange={(event) => update({ instructions: event.target.value })}
          placeholder="What should the model determine?"
        />
      </label>
      {options && (
        <div className={styles.optionsEditor}>
          <div className={styles.fieldHeading}>
            <span>{question.type === 'choice' ? 'Choices' : 'Labels'}</span>
            <span>Key and optional description</span>
          </div>
          {options.map((option, index) => (
            <div className={styles.optionRow} key={index}>
              <input
                aria-label={`${optionNoun} ${index + 1} key`}
                value={option.key}
                placeholder="key"
                spellCheck={false}
                onChange={(event) =>
                  update({
                    [optionsField]: options.map((item, i) =>
                      i === index ? { ...item, key: event.target.value } : item,
                    ),
                  })
                }
              />
              <input
                aria-label={`${optionNoun} ${index + 1} description`}
                value={option.description ?? ''}
                placeholder="Describe this option"
                onChange={(event) =>
                  update({
                    [optionsField]: options.map((item, i) =>
                      i === index ? { ...item, description: event.target.value } : item,
                    ),
                  })
                }
              />
              <button
                type="button"
                className={styles.removeButton}
                aria-label={`Remove ${optionNoun} ${index + 1}`}
                disabled={options.length <= (question.type === 'choice' ? 2 : 1)}
                onClick={() => update({ [optionsField]: options.filter((_, i) => i !== index) })}
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            className={styles.subtleButton}
            disabled={options.length >= 255}
            onClick={() =>
              update({
                [optionsField]: [
                  ...options,
                  { key: `${optionNoun}_${options.length + 1}`, description: '' },
                ],
              })
            }
          >
            + Add {optionNoun}
          </button>
        </div>
      )}
      {question.type === 'score' && (
        <div className={styles.optionsEditor}>
          <div className={styles.fieldHeading}>
            <span>Score levels</span>
            <span>Lowest → highest</span>
          </div>
          {(question.levels ?? []).map((level, index) => (
            <div className={styles.levelRow} key={index}>
              <span>{index}</span>
              <input
                aria-label={`Level ${index}`}
                value={level}
                onChange={(event) =>
                  update({
                    levels: question.levels?.map((item, i) =>
                      i === index ? event.target.value : item,
                    ),
                  })
                }
              />
              <button
                type="button"
                className={styles.removeButton}
                disabled={(question.levels?.length ?? 0) <= 2}
                aria-label={`Remove level ${index}`}
                onClick={() => update({ levels: question.levels?.filter((_, i) => i !== index) })}
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            className={styles.subtleButton}
            disabled={(question.levels?.length ?? 0) >= 10}
            onClick={() => update({ levels: [...(question.levels ?? []), ''] })}
          >
            + Add level
          </button>
        </div>
      )}
      {question.type === 'noul' && (
        <div className={styles.noulHint}>
          <span aria-hidden="true">◐</span>
          <p>
            <strong>A continuous truth estimate</strong>The answer is P(true), from 0 to 1. Ask one
            clear question; the model supplies its default false/true criteria.
          </p>
        </div>
      )}
      <details className={styles.questionSettings}>
        <summary>Question settings</summary>
        <label className={styles.field}>
          State field <span className={styles.optional}>Optional</span>
          <input
            value={question.over ?? ''}
            onChange={(event) => update({ over: event.target.value })}
            placeholder="Model default · or e.g. request, answer"
          />
        </label>
        {(question.type === 'set' || question.type === 'span') && (
          <label className={styles.field}>
            Selection threshold <span className={styles.optional}>Optional</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={question.threshold ?? ''}
              placeholder="Model default"
              onChange={(event) =>
                update({
                  threshold: event.target.value === '' ? undefined : event.target.valueAsNumber,
                })
              }
            />
          </label>
        )}
        <p className={styles.caption}>
          Leave these fields empty to use the model’s defaults. A span reads one state field.
        </p>
      </details>
    </div>
  )
}

export default function SystemOnePlaygroundPage() {
  const { user } = useAuth()
  const permitted = canRunEvaluation(user)
  const runtime = useSystemOnePlayground()
  const taskCatalog = useDecisionTasks()
  const [source, setSource] = useState(EXAMPLE_STATES[0].state)
  const [format, setFormat] = useState<'text' | 'json'>('text')
  const [drafts, setDrafts] = useState<QuestionDraft[]>(() => exampleDrafts(EXAMPLE_STATES[0]))
  const [selectedExample, setSelectedExample] = useState(EXAMPLE_STATES[0].id)
  const [templateBaseline, setTemplateBaseline] = useState<{
    source: string
    format: 'text' | 'json'
    questions: string
  } | null>(null)
  const [activeIndex, setActiveIndex] = useState(0)
  const [addOpen, setAddOpen] = useState(false)
  const questionTabs = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const container = questionTabs.current
    const active = container?.children[activeIndex]
    if (!container || !(active instanceof HTMLElement)) return
    const containerBounds = container.getBoundingClientRect()
    const activeBounds = active.getBoundingClientRect()
    if (activeBounds.right > containerBounds.right)
      container.scrollLeft += activeBounds.right - containerBounds.right + 16
    else if (activeBounds.left < containerBounds.left)
      container.scrollLeft -= containerBounds.left - activeBounds.left + 16
  }, [activeIndex, drafts.length])
  const built = useMemo(
    () => buildSystemOneRequest(source, format, drafts),
    [source, format, drafts],
  )
  const selected = runtime.selected
  const supported = selected?.question_types
  const unsupported = supported
    ? drafts.filter((draft) => !supported.includes(draft.question.type))
    : []
  const unavailable = !selected
    ? 'No deployed runtime is available.'
    : !selected.ready
      ? selected.unavailable_reason || 'This runtime is not ready.'
      : !selected.surfaces.includes('decisions')
        ? 'This runtime does not expose the System One decisions API.'
        : unsupported.length
          ? `This runtime does not support ${[...new Set(unsupported.map((draft) => draft.question.type))].join(', ')} questions.`
          : null
  const canRun =
    permitted &&
    !runtime.loading &&
    !runtime.capabilityError &&
    !unavailable &&
    !built.error &&
    !runtime.running
  const resultIsPrevious =
    runtime.result &&
    (runtime.result.deployment !== runtime.selectedId ||
      JSON.stringify(runtime.result.request) !== JSON.stringify(built.request))

  function loadExample(id: string) {
    const example = EXAMPLE_STATES.find((item) => item.id === id)
    if (!example) {
      const task = taskCatalog.data?.tasks.find((item) => `task:${item.id}` === id)
      if (!task) return
      const text =
        typeof task.template.state === 'string'
          ? task.template.state
          : JSON.stringify(task.template.state, null, 2)
      const nextFormat = typeof task.template.state === 'string' ? 'text' : 'json'
      const nextDrafts: QuestionDraft[] = Object.entries(task.template.questions).map(
        ([name, native]) => {
          const { criteria, ...question } = structuredClone(native)
          if (criteria !== undefined) {
            if (question.type === 'score' && Array.isArray(criteria))
              question.levels = criteria as string[]
            else if (criteria && typeof criteria === 'object' && !Array.isArray(criteria)) {
              const options = Object.entries(criteria).map(([key, description]) => ({
                key,
                description: typeof description === 'string' ? description : '',
              }))
              if (question.type === 'span' || question.type === 'set') question.labels = options
              else question.choices = options
            }
          }
          return { id: crypto.randomUUID(), name, question }
        },
      )
      setSource(text)
      setFormat(nextFormat)
      setDrafts(nextDrafts)
      setActiveIndex(0)
      setSelectedExample(id)
      setTemplateBaseline({
        source: text,
        format: nextFormat,
        questions: JSON.stringify(
          Object.fromEntries(nextDrafts.map(({ name, question }) => [name, question])),
        ),
      })
      return
    }
    setTemplateBaseline(null)
    setSource(example.state)
    setFormat('text')
    setSelectedExample(id)
    setDrafts(exampleDrafts(example))
    setActiveIndex(0)
  }

  const example = EXAMPLE_STATES.find((item) => item.id === selectedExample)
  const templateEdited = Boolean(
    templateBaseline &&
      (source !== templateBaseline.source ||
        format !== templateBaseline.format ||
        JSON.stringify(Object.fromEntries(drafts.map(({ name, question }) => [name, question]))) !==
          templateBaseline.questions),
  )
  const exampleEdited =
    templateEdited ||
    Boolean(
      example &&
        (source !== example.state ||
          format !== 'text' ||
          JSON.stringify(
            Object.fromEntries(drafts.map(({ name, question }) => [name, question])),
          ) !== JSON.stringify(example.questions)),
    )

  function addQuestion(type: QuestionType) {
    let index = 1
    let next = newQuestion(type, index)
    while (drafts.some((draft) => draft.name === next.name)) next = newQuestion(type, ++index)
    setDrafts([...drafts, next])
    setActiveIndex(drafts.length)
    setAddOpen(false)
  }

  function changeFormat(next: 'text' | 'json') {
    if (next === format) return
    if (next === 'json') setSource(JSON.stringify({ request: source }, null, 2))
    else {
      try {
        const parsed = JSON.parse(source) as { request?: unknown }
        if (typeof parsed?.request === 'string') setSource(parsed.request)
      } catch {
        /* Preserve the user's input. */
      }
    }
    setFormat(next)
  }

  return (
    <ConfigPageManagerLayout
      eyebrow="Build / System One"
      title="Decision Playground"
      description="Explore the System One API. Ask typed questions, compare probabilities, and inspect exactly what your decision model sees."
    >
      <div className={styles.page}>
        <section className={styles.runtimeBar} aria-label="Runtime target">
          <div className={styles.runtimeIcon} aria-hidden="true">
            ⌘
          </div>
          <SystemOneSelect
            className={styles.runtimeSelect}
            label="Runtime target"
            value={runtime.selectedId}
            onChange={runtime.setSelectedId}
            placeholder={runtime.loading ? 'Discovering models…' : 'No runtime available'}
            disabled={runtime.loading || runtime.running}
            options={(runtime.capabilities?.deployments ?? []).map(systemOneDeploymentOption)}
          />
          {selected && (
            <span className={selected.ready ? styles.successPill : styles.warningPill}>
              <i />
              {selected.ready ? 'Ready' : 'Unavailable'}
            </span>
          )}
          <div className={styles.runtimeActions}>
            <button
              type="button"
              disabled={runtime.loading || runtime.running}
              onClick={runtime.refresh}
            >
              {runtime.loading ? 'Refreshing…' : 'Refresh'}
            </button>
            <Link to="/decision-model">Decision Models ↗</Link>
            <Link to="/decision-model/monitoring">Decision Monitoring ↗</Link>
          </div>
        </section>
        {runtime.capabilityError && (
          <div role="alert" className={styles.error}>
            {runtime.capabilityError}
          </div>
        )}
        <div className={styles.workspace}>
          <div className={styles.composer}>
            <fieldset className={styles.composerFieldset} disabled={runtime.running}>
              <legend className={styles.srOnly}>Test request</legend>
              <section className={styles.panel} aria-labelledby="state-heading">
                <div className={styles.panelHeading}>
                  <div>
                    <span className={styles.step}>01</span>
                    <h2 id="state-heading">Input context</h2>
                  </div>
                  <div className={styles.segmented}>
                    <button
                      type="button"
                      aria-pressed={format === 'text'}
                      onClick={() => changeFormat('text')}
                    >
                      Text
                    </button>
                    <button
                      type="button"
                      aria-pressed={format === 'json'}
                      onClick={() => changeFormat('json')}
                    >
                      Structured
                    </button>
                  </div>
                </div>
                <label className={styles.srOnly} htmlFor="system-one-state">
                  Input context
                </label>
                <textarea
                  id="system-one-state"
                  className={`${styles.stateInput} ${format === 'json' ? styles.codeInput : ''}`}
                  rows={7}
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  spellCheck={format === 'text'}
                  placeholder={
                    format === 'text'
                      ? 'Paste a request, conversation, or passage to analyze…'
                      : '{ "request": "…", "answer": "…" }'
                  }
                />
                <div className={styles.contextFooter}>
                  <SystemOneSelect
                    className={styles.exampleSelect}
                    label="Load example"
                    value={selectedExample}
                    onChange={loadExample}
                    placeholder="Choose a starting point"
                    disabled={runtime.running}
                    options={[
                      ...EXAMPLE_STATES.map((example) => ({
                        value: example.id,
                        label: example.label,
                        description: example.description,
                      })),
                      ...(taskCatalog.data?.tasks ?? []).map((task) => ({
                        value: `task:${task.id}`,
                        label: task.title,
                        description: `${task.stage} task · ${task.description}`,
                      })),
                    ]}
                  />
                  <span>
                    {exampleEdited && <strong>Modified · </strong>}
                    {Array.from(source).length.toLocaleString()} characters
                  </span>
                </div>
              </section>
              <section className={styles.panel} aria-labelledby="questions-heading">
                <div className={styles.panelHeading}>
                  <div>
                    <span className={styles.step}>02</span>
                    <h2 id="questions-heading">Questions</h2>
                    <span className={styles.count}>{drafts.length}</span>
                  </div>
                  <button
                    type="button"
                    className={styles.subtleButton}
                    aria-expanded={addOpen}
                    onClick={() => setAddOpen(!addOpen)}
                  >
                    + Add question
                  </button>
                </div>
                {addOpen && (
                  <div className={styles.addMenu} aria-label="Add a question type">
                    {QUESTION_TYPES.map(({ type, label, symbol, detail }) => (
                      <button
                        type="button"
                        key={type}
                        disabled={supported !== undefined && !supported.includes(type)}
                        onClick={() => addQuestion(type)}
                      >
                        <span>{symbol}</span>
                        <div>
                          <strong>{label}</strong>
                          <small>{detail}</small>
                        </div>
                        <span>+</span>
                      </button>
                    ))}
                  </div>
                )}
                <div
                  ref={questionTabs}
                  className={styles.questionTabs}
                  aria-label="Questions in this request"
                >
                  {drafts.map((draft, index) => (
                    <div
                      key={draft.id}
                      className={index === activeIndex ? styles.activeQuestionTab : undefined}
                    >
                      <button
                        type="button"
                        aria-pressed={index === activeIndex}
                        onClick={() => setActiveIndex(index)}
                      >
                        {draft.name || 'Untitled'}
                        <small>{draft.question.type}</small>
                      </button>
                      {drafts.length > 1 && (
                        <button
                          type="button"
                          aria-label={`Remove question ${draft.name}`}
                          onClick={() => {
                            setDrafts(drafts.filter((_, i) => i !== index))
                            setActiveIndex(
                              Math.max(0, activeIndex >= index ? activeIndex - 1 : activeIndex),
                            )
                          }}
                        >
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                </div>
                <QuestionEditor
                  key={drafts[activeIndex].id}
                  draft={drafts[activeIndex]}
                  supported={supported}
                  onChange={(next) =>
                    setDrafts(drafts.map((draft, index) => (index === activeIndex ? next : draft)))
                  }
                />
              </section>
            </fieldset>
            <div className={styles.runBar}>
              <div className={styles.runHint}>
                {!permitted
                  ? 'Your account needs evaluation.run permission to send a test.'
                  : unavailable ||
                    built.error ||
                    'Runs inference on your selected runtime. No routing configuration is changed.'}
              </div>
              <div className={styles.runActions}>
                {runtime.running ? (
                  <>
                    <span role="status">Running · {(runtime.elapsed / 1000).toFixed(1)} s</span>
                    <button type="button" className={styles.cancelButton} onClick={runtime.cancel}>
                      Cancel request
                    </button>
                  </>
                ) : (
                  <button
                    type="button"
                    className={styles.runButton}
                    disabled={!canRun}
                    onClick={() => built.request && void runtime.run(built.request)}
                  >
                    <span aria-hidden="true">▶</span> Run{' '}
                    {drafts.length > 1 ? `${drafts.length} questions` : 'test'}
                  </button>
                )}
              </div>
            </div>
            {built.request && (
              <JSONInspector
                title="Preview request"
                value={{ deployment: runtime.selectedId, request: built.request }}
              />
            )}
          </div>
          <section
            className={styles.output}
            aria-labelledby="results-heading"
            aria-busy={runtime.running}
          >
            <div className={styles.outputHeading}>
              <div>
                <span className={styles.step}>03</span>
                <h2 id="results-heading">Results</h2>
              </div>
              <span>Live model inference</span>
            </div>
            {runtime.error && (
              <div role="alert" className={styles.error}>
                {runtime.error}
              </div>
            )}
            {runtime.notice && (
              <div role="status" className={styles.notice}>
                {runtime.notice}
              </div>
            )}
            {resultIsPrevious && (
              <div className={styles.notice}>
                Showing the last submitted request. Run again to see results for your changes.
              </div>
            )}
            {runtime.result ? (
              <SystemOneResults run={runtime.result} />
            ) : (
              <div className={styles.emptyState}>
                <div
                  className={`${styles.emptyGraphic} ${runtime.running ? styles.working : ''}`}
                  aria-hidden="true"
                >
                  <span>◉</span>
                  <span>▥</span>
                  <span>◐</span>
                  <span>⌁</span>
                  <span>⊞</span>
                </div>
                <h3>{runtime.running ? 'Your model is thinking' : 'From context to a decision'}</h3>
                <p>
                  {runtime.running
                    ? 'The selected runtime is answering your questions. Results will appear together when the request finishes.'
                    : 'Run a test to see probability distributions, scores, selected labels, and extracted spans.'}
                </p>
                {!runtime.running && (
                  <span className={styles.emptyFootnote}>
                    Real responses from your deployed model
                  </span>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </ConfigPageManagerLayout>
  )
}
