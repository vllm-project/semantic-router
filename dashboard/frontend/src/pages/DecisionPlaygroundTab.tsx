import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { FlaskConical, Play, Plus, RefreshCw, Square } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
  CodeView,
  CopyButton,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
  EmptyState,
  Kbd,
  Segmented,
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
  Skeleton,
  StatusDot,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Textarea,
  cn,
} from '../ui'
import { canRunEvaluation } from '../utils/accessControl'
import SystemOneQuestionEditor from './SystemOneQuestionEditor'
import { QUESTION_ICONS } from './decisionQuestionTypes'
import SystemOneResults, { SystemOneRoutingOutcome } from './SystemOneResults'
import {
  buildSystemOneRequest,
  EXAMPLE_STATES,
  exampleDrafts,
  finiteNumber,
  newQuestion,
  QUESTION_TYPES,
  type QuestionDraft,
  type QuestionType,
} from './systemOnePlayground'
import { systemOneTargetOptions } from './systemOneTargets'
import { useDecisionTasks } from './useDecisionTasks'
import { useSystemOnePlayground } from './useSystemOnePlayground'

type View = 'answers' | 'request' | 'response'

function formatElapsed(milliseconds: number): string {
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} ms`
    : `${(milliseconds / 1000).toFixed(2)} s`
}

export default function DecisionPlaygroundTab() {
  const { user } = useAuth()
  const permitted = canRunEvaluation(user)
  const runtime = useSystemOnePlayground()
  const taskCatalog = useDecisionTasks()
  const [params] = useSearchParams()
  const targetParam = params.get('target')
  const { setSelectedId } = runtime
  useEffect(() => {
    if (targetParam) setSelectedId(targetParam)
  }, [targetParam, setSelectedId])
  const [source, setSource] = useState(EXAMPLE_STATES[0].state)
  const [format, setFormat] = useState<'text' | 'json'>('text')
  const [drafts, setDrafts] = useState<QuestionDraft[]>(() => exampleDrafts(EXAMPLE_STATES[0]))
  const [selectedExample, setSelectedExample] = useState(EXAMPLE_STATES[0].id)
  const [templateBaseline, setTemplateBaseline] = useState<{
    source: string
    format: 'text' | 'json'
    questions: string
  } | null>(null)
  const [view, setView] = useState<View>('answers')
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
    ? 'Choose an active auto route or a deployed model.'
    : !selected.ready
      ? selected.unavailable_reason || 'This runtime is not ready.'
      : !selected.surfaces.includes('decisions')
        ? 'This runtime does not expose the System One decisions API.'
        : unsupported.length
          ? `This runtime does not support ${[...new Set(unsupported.map((draft) => draft.question.type))].join(', ')} questions.`
          : null
  const canRun = permitted && !runtime.loading && !unavailable && !built.error && !runtime.running
  const result = runtime.result
  const resultIsPrevious =
    result &&
    ((result.targetKey ?? result.deployment) !== runtime.selectedId ||
      JSON.stringify(result.request) !== JSON.stringify(built.request))
  const names = drafts.map((draft) => draft.name.trim())

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
  }

  const example = EXAMPLE_STATES.find((item) => item.id === selectedExample)
  const currentQuestions = JSON.stringify(
    Object.fromEntries(drafts.map(({ name, question }) => [name, question])),
  )
  const templateEdited = Boolean(
    templateBaseline &&
      (source !== templateBaseline.source ||
        format !== templateBaseline.format ||
        currentQuestions !== templateBaseline.questions),
  )
  const exampleEdited =
    templateEdited ||
    Boolean(
      example &&
        (source !== example.state ||
          format !== 'text' ||
          currentQuestions !== JSON.stringify(example.questions)),
    )

  function addQuestion(type: QuestionType) {
    let index = 1
    let next = newQuestion(type, index)
    while (drafts.some((draft) => draft.name === next.name)) next = newQuestion(type, ++index)
    setDrafts([...drafts, next])
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

  const execute = () => {
    if (!canRun || !built.request) return
    setView('answers')
    void runtime.run(built.request)
  }

  const targetOptions = systemOneTargetOptions(runtime.targets)
  const groups = ['Automatic routes', 'Direct models']
    .map((group) => ({ group, options: targetOptions.filter((option) => option.group === group) }))
    .filter((group) => group.options.length)
  const submitted = result
    ? result.targetKind === 'route'
      ? { model: result.deployment, request: result.request }
      : { deployment: result.deployment, request: result.request }
    : built.request
      ? selected?.kind === 'route'
        ? { model: selected.id, request: built.request }
        : { deployment: selected?.id ?? runtime.selectedId, request: built.request }
      : null
  const responseJSON = result ? JSON.stringify(result.response, null, 2) : ''
  const hint = !permitted
    ? 'Your account needs evaluation.run permission to send a test.'
    : unavailable ||
      built.error ||
      'Runs inference on the selected runtime. No routing configuration changes.'

  return (
    <div className="flex flex-col gap-4">
      <div className="-mb-1 flex flex-wrap items-center gap-3">
        <span className="text-xs text-muted-foreground" role="status">
          {runtime.refreshing
            ? 'Discovering runtimes and auto routes…'
            : `${runtime.targets.length} ${runtime.targets.length === 1 ? 'target' : 'targets'} discovered`}
        </span>
        <Button
          variant="outline"
          size="sm"
          className="ml-auto"
          disabled={runtime.refreshing || runtime.running}
          onClick={runtime.refresh}
        >
          <RefreshCw className={cn(runtime.refreshing && 'animate-spin')} />
          Refresh
        </Button>
      </div>
      {runtime.capabilityError ? (
        <div role="alert">
          <Alert tone="danger" title="Model discovery failed">
            {runtime.capabilityError}
          </Alert>
        </div>
      ) : null}
      {runtime.routeError ? (
        <Alert tone="warning" title="Automatic routes are unavailable; direct models still work">
          {runtime.routeError}
        </Alert>
      ) : null}
      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.08fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Request</CardTitle>
            <CardDescription>A state and the typed questions to ask about it</CardDescription>
            <CardAction>
              <Select
                value={runtime.selectedId || undefined}
                onValueChange={runtime.setSelectedId}
                disabled={runtime.running || !targetOptions.length}
              >
                <SelectTrigger size="sm" className="max-w-72 gap-2" aria-label="Runtime target">
                  {selected ? <StatusDot tone={selected.ready ? 'success' : 'warning'} /> : null}
                  <SelectValue
                    placeholder={runtime.loading ? 'Discovering targets…' : 'No target available'}
                  >
                    {targetOptions.find((option) => option.value === runtime.selectedId)?.label}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent align="end" className="max-w-[min(28rem,calc(100vw-2rem))]">
                  {groups.map(({ group, options }) => (
                    <SelectGroup key={group}>
                      <SelectLabel>{group}</SelectLabel>
                      {options.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          <span className="font-mono text-[12.5px]">{option.label}</span>
                          {option.description ? (
                            <span className="ml-2 text-xs text-subtle-foreground">
                              {option.description}
                            </span>
                          ) : null}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
            </CardAction>
          </CardHeader>
          <div className="flex flex-col gap-5 px-5 pb-5 pt-4">
            {selected?.kind === 'route' ? (
              <p className="text-xs text-muted-foreground">
                Runs the <span className="font-medium text-foreground">{selected.recipe}</span>{' '}
                recipe across its model pool. Signals and the selected algorithm share up to{' '}
                {selected.execution_timeout_ms / 1000} seconds. The result shows the selected stage
                and model.
              </p>
            ) : null}
            {!runtime.refreshing && runtime.routes && runtime.routes.routes.length === 0 ? (
              <p className="text-xs text-subtle-foreground">
                Publish a System One route in the active configuration to test auto alongside direct
                models.
              </p>
            ) : null}
            <fieldset disabled={runtime.running} className="contents">
              <legend className="sr-only">Test request</legend>
              <section className="flex flex-col gap-1.5" aria-labelledby="playground-state-label">
                <div className="flex flex-wrap items-center gap-2">
                  <label
                    id="playground-state-label"
                    htmlFor="playground-state"
                    className="text-[13px] font-medium"
                  >
                    State
                  </label>
                  <Segmented
                    size="sm"
                    aria-label="State format"
                    value={format}
                    onValueChange={changeFormat}
                    options={[
                      { value: 'text', label: 'Text' },
                      { value: 'json', label: 'Structured' },
                    ]}
                  />
                  <Select
                    value={selectedExample}
                    onValueChange={loadExample}
                    disabled={runtime.running}
                  >
                    <SelectTrigger size="sm" className="ml-auto max-w-56" aria-label="Load example">
                      <SelectValue placeholder="Load an example">
                        {EXAMPLE_STATES.find((item) => item.id === selectedExample)?.label ??
                          taskCatalog.data?.tasks.find(
                            (task) => `task:${task.id}` === selectedExample,
                          )?.title}
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent align="end" className="max-w-[min(26rem,calc(100vw-2rem))]">
                      <SelectGroup>
                        <SelectLabel>Examples</SelectLabel>
                        {EXAMPLE_STATES.map((item) => (
                          <SelectItem key={item.id} value={item.id}>
                            {item.label}
                            <span className="ml-2 text-xs text-subtle-foreground">
                              {item.description}
                            </span>
                          </SelectItem>
                        ))}
                      </SelectGroup>
                      {taskCatalog.data?.tasks.length ? (
                        <SelectGroup>
                          <SelectLabel>Built-in tasks</SelectLabel>
                          {taskCatalog.data.tasks.map((task) => (
                            <SelectItem key={task.id} value={`task:${task.id}`}>
                              {task.title}
                              <span className="ml-2 text-xs text-subtle-foreground">
                                {task.stage} task
                              </span>
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      ) : null}
                    </SelectContent>
                  </Select>
                </div>
                <Textarea
                  id="playground-state"
                  rows={6}
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                      event.preventDefault()
                      execute()
                    }
                  }}
                  spellCheck={format === 'text'}
                  className={cn('text-[13px]', format === 'json' && 'font-mono text-[12.5px]')}
                  placeholder={
                    format === 'text'
                      ? 'Paste a request, conversation or passage to analyze…'
                      : '{ "request": "…", "answer": "…" }'
                  }
                />
                <div className="flex items-center gap-2 text-xs text-subtle-foreground">
                  <span>The conversation, tool output or document the questions are about.</span>
                  <span className="ml-auto shrink-0 tabular">
                    {exampleEdited ? (
                      <span className="font-medium text-muted-foreground">Modified · </span>
                    ) : null}
                    {Array.from(source).length.toLocaleString()} characters
                  </span>
                </div>
              </section>
              <section className="flex flex-col gap-2" aria-labelledby="playground-questions-label">
                <div className="flex items-center gap-2">
                  <span id="playground-questions-label" className="text-[13px] font-medium">
                    Questions
                  </span>
                  <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
                    {drafts.length}
                  </span>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="xs" className="ml-auto">
                        <Plus />
                        Add question
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-64">
                      <DropdownMenuLabel>Question type</DropdownMenuLabel>
                      {QUESTION_TYPES.map(({ type, label, detail }) => {
                        const Icon = QUESTION_ICONS[type]
                        const disabled = supported !== undefined && !supported.includes(type)
                        return (
                          <DropdownMenuItem
                            key={type}
                            disabled={disabled}
                            onSelect={() => addQuestion(type)}
                          >
                            <Icon />
                            <span className="flex flex-col">
                              <span>{label}</span>
                              <span className="text-xs text-subtle-foreground">
                                {disabled
                                  ? `Not supported by ${selected?.model ?? 'this runtime'}`
                                  : detail}
                              </span>
                            </span>
                          </DropdownMenuItem>
                        )
                      })}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <div className="flex flex-col gap-2">
                  <AnimatePresence initial={false}>
                    {drafts.map((draft, index) => (
                      <SystemOneQuestionEditor
                        key={draft.id}
                        draft={draft}
                        index={index}
                        supported={supported}
                        duplicate={Boolean(names[index]) && names.indexOf(names[index]) !== index}
                        onChange={(next) =>
                          setDrafts((current) =>
                            current.map((item) => (item.id === draft.id ? next : item)),
                          )
                        }
                        onRemove={
                          drafts.length > 1
                            ? () =>
                                setDrafts((current) =>
                                  current.filter((item) => item.id !== draft.id),
                                )
                            : undefined
                        }
                      />
                    ))}
                  </AnimatePresence>
                </div>
              </section>
            </fieldset>
          </div>
          <div className="flex flex-wrap items-center gap-3 border-t border-border px-5 py-3">
            <span
              className={cn(
                'min-w-0 flex-1 text-xs',
                permitted && !unavailable && !built.error
                  ? 'text-subtle-foreground'
                  : 'text-warning-foreground',
              )}
            >
              {hint}
            </span>
            {runtime.running ? (
              <>
                <span role="status" className="font-mono text-xs tabular text-muted-foreground">
                  Running · {(runtime.elapsed / 1000).toFixed(1)} s
                </span>
                <Button variant="outline" size="sm" onClick={runtime.cancel}>
                  <Square />
                  Cancel request
                </Button>
              </>
            ) : (
              <Button onClick={execute} disabled={!canRun}>
                <Play />
                Run {drafts.length > 1 ? `${drafts.length} questions` : 'test'}
                <Kbd className="border-primary-foreground/25 bg-primary-foreground/15 text-primary-foreground shadow-none">
                  ⌘↵
                </Kbd>
              </Button>
            )}
          </div>
        </Card>

        <Card className="overflow-hidden xl:sticky xl:top-6" aria-busy={runtime.running}>
          <CardHeader className="pb-3">
            <CardTitle>Answers</CardTitle>
            <CardDescription className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              {result ? (
                <>
                  <span className="font-mono">{result.response.model}</span>
                  <span className="tabular">{formatElapsed(result.elapsed)}</span>
                  <span className="tabular">
                    {result.response.usage.input_tokens.toLocaleString()} →{' '}
                    {result.response.usage.output_tokens.toLocaleString()} tokens
                  </span>
                  {finiteNumber(result.response.meta?.compute_ms) ? (
                    <span className="tabular">
                      compute {result.response.meta.compute_ms.toFixed(1)} ms
                    </span>
                  ) : null}
                  {result.response.meta?.profile ? (
                    <span>profile {result.response.meta.profile}</span>
                  ) : null}
                  <span>
                    on <span className="font-mono">{result.deployment}</span>
                  </span>
                </>
              ) : (
                <span>Live inference on your deployed model</span>
              )}
            </CardDescription>
            <CardAction>
              {result ? (
                <>
                  <Badge tone="success" size="sm" dot>
                    Response received
                  </Badge>
                  <CopyButton value={responseJSON} label="Copy response" />
                </>
              ) : null}
            </CardAction>
          </CardHeader>
          <Tabs value={view} onValueChange={(value) => setView(value as View)} className="gap-0">
            <TabsList className="px-5">
              <TabsTrigger value="answers">Answers</TabsTrigger>
              <TabsTrigger value="request">Request</TabsTrigger>
              <TabsTrigger value="response" disabled={!result}>
                Response
              </TabsTrigger>
            </TabsList>
            <TabsContent value="answers" className="flex flex-col gap-3 px-5 py-4">
              {runtime.error ? (
                <div role="alert">
                  <Alert tone="danger" title="The test failed">
                    {runtime.error}
                  </Alert>
                </div>
              ) : null}
              {runtime.notice ? <Alert tone="neutral">{runtime.notice}</Alert> : null}
              {resultIsPrevious && !runtime.running ? (
                <Alert tone="neutral">
                  Showing the last submitted request. Run again to see results for your changes.
                </Alert>
              ) : null}
              {runtime.running ? (
                <div
                  className="flex flex-col gap-5 py-2"
                  role="status"
                  aria-label="Your model is thinking"
                >
                  <span className="text-xs text-muted-foreground">Your model is thinking…</span>
                  {drafts.map((draft) => (
                    <div key={draft.id} className="flex flex-col gap-2.5">
                      <Skeleton className="h-3.5 w-48" />
                      <Skeleton className="h-2 w-full" />
                      <Skeleton className="h-2 w-4/5" />
                    </div>
                  ))}
                </div>
              ) : result ? (
                <motion.div
                  key={result.completedAt.getTime()}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                  className="flex flex-col gap-3"
                >
                  <SystemOneRoutingOutcome run={result} />
                  <SystemOneResults run={result} />
                </motion.div>
              ) : !runtime.error ? (
                <EmptyState
                  compact
                  icon={FlaskConical}
                  tone="violet"
                  title="From context to a decision"
                  description="Run a test to see probability distributions, scores, selected labels and extracted spans from your deployed model."
                />
              ) : null}
            </TabsContent>
            <TabsContent
              value="request"
              className="max-h-[640px] overflow-auto bg-surface-sunken/50"
            >
              <div className="flex items-center justify-between border-b border-border px-4 py-2 text-xs text-muted-foreground">
                <span>{result ? 'Submitted request' : 'Preview request'} · application/json</span>
                {submitted ? (
                  <CopyButton value={JSON.stringify(submitted, null, 2)} label="Copy request" />
                ) : null}
              </div>
              {submitted ? (
                <CodeView code={JSON.stringify(submitted, null, 2)} language="json" lineNumbers />
              ) : (
                <p className="px-4 py-6 text-xs text-subtle-foreground">
                  {built.error ?? 'Nothing to preview yet.'}
                </p>
              )}
            </TabsContent>
            <TabsContent
              value="response"
              className="max-h-[640px] overflow-auto bg-surface-sunken/50"
            >
              {result ? <CodeView code={responseJSON} language="json" lineNumbers /> : null}
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  )
}
