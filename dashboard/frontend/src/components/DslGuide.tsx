import * as React from 'react'
import type { LucideIcon } from 'lucide-react'
import {
  BookOpenText,
  Boxes,
  Calculator,
  ChevronRight,
  ClipboardList,
  Plus,
  Puzzle,
  Radar,
  Search,
  Split,
  X,
  Zap,
} from 'lucide-react'
import {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  PLUGIN_DESCRIPTIONS,
  PLUGIN_TYPES,
  SIGNAL_TYPES,
  getAlgorithmFieldSchema,
  getPluginFieldSchema,
  getSignalFieldSchema,
} from '@/lib/dslMutations'
import type { FieldSchema } from '@/lib/dslSchemaTypes'
import { Button, CopyButton, cn } from '../ui'

interface DslGuideProps {
  onInsertSnippet?: (snippet: string) => void
}

function Section({
  title,
  icon: Icon,
  defaultOpen = false,
  forceOpen = false,
  children,
}: {
  title: string
  icon: LucideIcon
  defaultOpen?: boolean
  forceOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = React.useState(defaultOpen)
  const regionId = React.useId()
  const expanded = open || forceOpen
  return (
    <section className="border-b border-border last:border-0">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={expanded}
        aria-controls={regionId}
        className="flex w-full items-center gap-2 px-4 py-3 text-left text-[13px] font-semibold outline-none transition-colors hover:bg-surface-subtle focus-visible:bg-surface-subtle"
      >
        <ChevronRight
          className={cn(
            'size-3.5 text-subtle-foreground transition-transform duration-200',
            expanded && 'rotate-90',
          )}
        />
        <Icon className="size-4 text-subtle-foreground" />
        {title}
      </button>
      {expanded ? (
        <div id={regionId} className="flex flex-col gap-3 px-4 pb-4">
          {children}
        </div>
      ) : null}
    </section>
  )
}

function Snippet({
  code,
  label,
  onInsert,
}: {
  code: string
  label?: string
  onInsert?: (code: string) => void
}) {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-surface-sunken/50 dark:bg-surface-sunken">
      <div className="flex h-8 items-center gap-1 border-b border-border pl-3 pr-1">
        <span className="flex-1 truncate text-xs font-medium text-muted-foreground">
          {label ?? 'Example'}
        </span>
        <CopyButton value={code} label={`Copy ${label ?? 'example'}`} />
        {onInsert ? (
          <Button variant="ghost" size="xs" onClick={() => onInsert(code)}>
            <Plus />
            Insert
          </Button>
        ) : null}
      </div>
      <pre className="overflow-x-auto px-3 py-2.5 font-mono text-[12px] leading-5 text-foreground/90">
        {code}
      </pre>
    </div>
  )
}

function Hint({ children }: { children: React.ReactNode }) {
  return <p className="text-[13px] leading-relaxed text-muted-foreground">{children}</p>
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded-[5px] border border-border bg-surface-subtle px-1 py-px font-mono text-[11.5px] text-foreground/90">
      {children}
    </code>
  )
}

function Table({ head, rows }: { head: [string, string, string]; rows: React.ReactNode[][] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <table className="w-full text-left text-xs">
        <thead className="bg-surface-subtle text-subtle-foreground">
          <tr>
            {head.map((cell) => (
              <th key={cell} className="px-2.5 py-1.5 font-medium">
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((row, index) => (
            <tr key={index} className="align-top">
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  className={cn(
                    'px-2.5 py-1.5',
                    cellIndex === 0 ? 'whitespace-nowrap' : 'text-muted-foreground',
                  )}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function FieldTable({ fields }: { fields: FieldSchema[] }) {
  if (fields.length === 0)
    return <p className="text-xs text-subtle-foreground">No configurable fields.</p>
  return (
    <Table
      head={['Field', 'Type', 'Info']}
      rows={fields.map((field) => [
        <span key="key" className="font-mono text-foreground">
          {field.key}
          {field.required ? <span className="text-destructive-foreground"> *</span> : null}
        </span>,
        <span key="type" className="font-mono">
          {field.type}
          {field.options?.length ? (
            <span className="text-subtle-foreground"> ({field.options.join(' | ')})</span>
          ) : null}
        </span>,
        field.description ?? '',
      ])}
    />
  )
}

function TypeDetail({
  name,
  description,
  fields,
}: {
  name: string
  description?: string
  fields: FieldSchema[]
}) {
  const [expanded, setExpanded] = React.useState(false)
  const regionId = React.useId()
  return (
    <div className="rounded-lg border border-border">
      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
        aria-controls={regionId}
        className="flex w-full items-start gap-2 px-2.5 py-2 text-left outline-none transition-colors hover:bg-surface-subtle focus-visible:bg-surface-subtle"
      >
        <ChevronRight
          className={cn(
            'mt-0.5 size-3.5 shrink-0 text-subtle-foreground transition-transform',
            expanded && 'rotate-90',
          )}
        />
        <span className="flex min-w-0 flex-col gap-0.5">
          <span className="font-mono text-[12px] font-medium text-foreground">{name}</span>
          {description ? (
            <span className="text-xs leading-relaxed text-muted-foreground">{description}</span>
          ) : null}
        </span>
      </button>
      {expanded ? (
        <div id={regionId} className="px-2.5 pb-2.5">
          <FieldTable fields={fields} />
        </div>
      ) : null}
    </div>
  )
}

const QUICK_START_SNIPPET = `# Models
MODEL "qwen2.5:3b" {
  capabilities: ["general", "reasoning"]
}

# Signals
SIGNAL keyword urgent_request {
  operator: "any"
  keywords: ["urgent", "asap", "emergency"]
  method: "regex"
  case_sensitive: false
}

SIGNAL domain ai_topics {
  description: "Computer science and machine learning"
  labels: ["computer science"]
}

# Plugins
PLUGIN replay router_replay {
  enabled: true
}

# Routes
ROUTE ai_route (description = "AI-related requests") {
  PRIORITY 100

  WHEN keyword("urgent_request") AND domain("ai_topics")

  MODEL "qwen2.5:3b" (reasoning = false)

  ALGORITHM multi_factor {
    weights: { quality: 0.6, cost: 0.4 }
  }

  PLUGIN replay
}`

const MODEL_SNIPPET = `MODEL "qwen2.5:3b" {
  param_size: "3b"
  capabilities: ["general", "reasoning"]
  modalities: { input: ["text", "image"], output: ["text"] }
}`

const SIGNAL_SNIPPET = `SIGNAL keyword my_signal {
  operator: "any"
  keywords: ["hello", "world"]
  method: "regex"
  case_sensitive: false
}`

const ROUTE_SNIPPET = `ROUTE my_route (description = "Description") {
  PRIORITY 100

  WHEN keyword("my_signal")

  MODEL "qwen2.5:3b" (reasoning = false)

  ALGORITHM multi_factor {
    weights: { quality: 0.6, cost: 0.4 }
  }
}`

const PLUGIN_SNIPPET = `PLUGIN my_plugin router_replay {
  enabled: true
}`

const SIGNAL_DESCRIPTIONS: Record<string, string> = {
  keyword: 'Match queries by keyword lists (regex, BM25, n-gram), or count how often they occur',
  domain: "Match the request's subject area against the rule's labels",
  fact_check: 'Flag queries requiring factual verification',
  user_feedback: 'Match the feedback a user turn gives about the previous answer',
  language: 'Detect query language',
  context: 'Context window length requirements',
  complexity: 'Match how difficult the request is: easy, medium or hard',
  modality: 'Match the output the request asks for, such as text, image, speech or video',
  authz: 'Authorization-based routing (RBAC)',
  jailbreak:
    'Detect prompt injection or jailbreak attempts in the request or, at the response stage, the answer',
  safety:
    'Detect harmful content in the request or, at the response stage, the answer; with labels, in those harm categories',
  hallucination:
    "Flag claims in the model's answer that its retrieved documents and tool results do not support (response stage)",
  refusal: 'Match an answer in which the model refused the request (response stage)',
  turn_relation:
    'Match how the current turn relates to the earlier ones: continue, reask, shift or return',
  pii: 'Detect personally identifiable information in queries',
  conversation: 'Count or detect conversation features: messages, tool definitions and tool calls',
  metadata: 'Match caller-provided request metadata by key (untrusted, never an identity)',
  input_modality: 'Match a request that contains text, image, audio or video input',
  agent: 'Match the agent a request belongs to: main or subagent, its harness and subagent type',
  systemone: 'Ask a System One model a named question and match its answer',
}

const MODEL_FIELDS: React.ReactNode[][] = [
  [
    <Code key="c">param_size</Code>,
    'string',
    <>
      Logical size hint such as <Code>3b</Code> or <Code>70b</Code>
    </>,
  ],
  [<Code key="c">description</Code>, 'string', 'Human-readable routing summary'],
  [<Code key="c">capabilities</Code>, 'string[]', 'Semantic tags used by selection or tooling'],
  [<Code key="c">context_window_size</Code>, 'number', 'Context window in tokens'],
  [<Code key="c">max_output_tokens</Code>, 'number', 'Longest answer the model writes'],
  [<Code key="c">tags</Code>, 'string[]', 'Free-form labels'],
  [<Code key="c">loras</Code>, 'object[]', 'LoRA adapters: name and description'],
  [
    <Code key="c">modalities</Code>,
    'object',
    <>
      What the model reads and writes, such as{' '}
      <Code>{'{ input: ["text", "image"], output: ["text"] }'}</Code>
    </>,
  ],
]

const ROUTE_CLAUSES: React.ReactNode[][] = [
  [<Code key="c">PRIORITY</Code>, 'Yes', 'Integer priority (higher = matched first)'],
  [<Code key="c">TIER</Code>, 'No', 'Integer tier grouping for route classification'],
  [<Code key="c">DESCRIPTION</Code>, 'No', 'Route description (can also be set in the header)'],
  [<Code key="c">WHEN</Code>, 'No', 'Boolean expression of signal references'],
  [<Code key="c">MODEL</Code>, 'Yes', 'One or more model references with attributes'],
  [<Code key="c">ALGORITHM</Code>, 'No', 'Model selection algorithm with its config'],
  [<Code key="c">PLUGIN</Code>, 'No', 'Plugin references, with optional inline overrides'],
]

const OPERATORS: React.ReactNode[][] = [
  [<Code key="c">NOT</Code>, 'Highest', <Code key="e">NOT domain(&quot;other&quot;)</Code>],
  [
    <Code key="c">AND</Code>,
    'Medium',
    <Code key="e">keyword(&quot;a&quot;) AND domain(&quot;b&quot;)</Code>,
  ],
  [
    <Code key="c">OR</Code>,
    'Lowest',
    <Code key="e">domain(&quot;math&quot;) OR domain(&quot;code&quot;)</Code>,
  ],
]

const MODEL_ATTRIBUTES: React.ReactNode[][] = [
  [
    <Code key="c">reasoning</Code>,
    'boolean',
    "Enable the model's catalog-defined reasoning behavior",
  ],
  [
    <Code key="c">effort</Code>,
    'string',
    <>
      <Code>low</Code> | <Code>medium</Code> | <Code>high</Code>
    </>,
  ],
  [
    <Code key="c">param_size</Code>,
    'string',
    <>
      e.g. <Code>3b</Code>, <Code>70b</Code>
    </>,
  ],
  [<Code key="c">weight</Code>, 'number', 'Routing weight (0–1)'],
  [<Code key="c">lora</Code>, 'string', 'LoRA adapter name'],
]

const CHEAT_SHEET: Array<[string, string]> = [
  ['MODEL <name> { ... }', 'Declare a routing model catalog entry'],
  ['SIGNAL <type> <name> { ... }', 'Declare a signal'],
  ['ROUTE <name> (description = "...") { ... }', 'Define a route'],
  ['DECISION_TREE <name> { IF ... ELSE ... }', 'Define if/else conditional routing logic'],
  ['PROJECTION <score|mapping|partition> <name> { ... }', 'Declare a signal projection'],
  ['PLUGIN <name> <type> { ... }', 'Declare a plugin template'],
  ['PRIORITY <number>', 'Route priority (higher = matched first)'],
  ['TIER <number>', 'Tier grouping (inside ROUTE)'],
  ['DESCRIPTION "text"', 'Route description (inside ROUTE body)'],
  ['WHEN <bool_expr>', 'Condition (inside ROUTE)'],
  ['MODEL "name" (attrs...)', 'Model reference (inside ROUTE)'],
  ['ALGORITHM <type> { ... }', 'Algorithm (inside ROUTE)'],
  [
    'providers / global live in config.yaml',
    'Deployment bindings and runtime defaults are outside the DSL',
  ],
  ['# comment', 'Line comment'],
  ['"string" | 123 | true | [a, b]', 'Value types'],
  ['{ key: value, ... }', 'Nested object'],
]

/** The Signal DSL reference: syntax, every signal, algorithm and plugin type, and snippets. */
export default function DslGuide({ onInsertSnippet }: DslGuideProps) {
  const [query, setQuery] = React.useState('')
  const needle = query.toLowerCase().trim()
  const matches = (text: string) => !needle || text.toLowerCase().includes(needle)
  const searching = Boolean(needle)

  const signals = SIGNAL_TYPES.filter((type) =>
    matches(`signal ${type} ${SIGNAL_DESCRIPTIONS[type] ?? ''}`),
  )
  const algorithms = ALGORITHM_TYPES.filter((type) =>
    matches(`algorithm ${type} ${ALGORITHM_DESCRIPTIONS[type] ?? ''}`),
  )
  const plugins = PLUGIN_TYPES.filter((type) =>
    matches(`plugin ${type} ${PLUGIN_DESCRIPTIONS[type] ?? ''}`),
  )
  const showQuickStart = matches('quick start template example snippet')
  const showModels = matches('model catalog reasoning capability modalities param_size')
  const showRoutes = matches('route when expression boolean model algorithm priority tier')
  const showCheat = matches('cheat sheet syntax grammar reference')
  const nothing =
    !showQuickStart &&
    !showModels &&
    !showRoutes &&
    !showCheat &&
    signals.length === 0 &&
    algorithms.length === 0 &&
    plugins.length === 0

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="shrink-0 border-b border-border px-4 py-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-subtle-foreground" />
          <input
            type="search"
            aria-label="Search the DSL guide"
            placeholder="Search keywords, types and fields…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="h-8 w-full rounded-md border border-input bg-surface pl-8 pr-7 text-[12.5px] shadow-xs outline-none transition-[border-color,box-shadow] placeholder:text-subtle-foreground hover:border-border-strong focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
          />
          {query ? (
            <button
              type="button"
              aria-label="Clear search"
              onClick={() => setQuery('')}
              className="absolute right-1.5 top-1/2 grid size-5 -translate-y-1/2 place-items-center rounded text-subtle-foreground hover:text-foreground"
            >
              <X className="size-3" />
            </button>
          ) : null}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {showQuickStart ? (
          <Section title="Quick start" icon={Zap} defaultOpen forceOpen={searching}>
            <Hint>
              A DSL file defines <strong>signals</strong> (what to detect), <strong>routes</strong>{' '}
              (how to decide), <strong>models</strong> (the semantic catalog) and{' '}
              <strong>plugins</strong> (reusable route helpers).
            </Hint>
            <Snippet code={QUICK_START_SNIPPET} label="Full example" onInsert={onInsertSnippet} />
            <div className="grid gap-2">
              <Snippet code={MODEL_SNIPPET} label="Model" onInsert={onInsertSnippet} />
              <Snippet code={SIGNAL_SNIPPET} label="Signal" onInsert={onInsertSnippet} />
              <Snippet code={ROUTE_SNIPPET} label="Route" onInsert={onInsertSnippet} />
              <Snippet code={PLUGIN_SNIPPET} label="Plugin" onInsert={onInsertSnippet} />
            </div>
          </Section>
        ) : null}
        {showModels ? (
          <Section title="Model catalog" icon={Boxes} forceOpen={searching}>
            <Hint>
              Top-level models define the semantic routing catalog. Syntax:{' '}
              <Code>{'MODEL <name> { fields }'}</Code>
            </Hint>
            <Table head={['Field', 'Type', 'Description']} rows={MODEL_FIELDS} />
            <Snippet
              code={MODEL_SNIPPET}
              label="Model catalog example"
              onInsert={onInsertSnippet}
            />
          </Section>
        ) : null}
        {signals.length ? (
          <Section
            title={`Signals · ${SIGNAL_TYPES.length} types`}
            icon={Radar}
            forceOpen={searching}
          >
            <Hint>
              Signals detect patterns in requests. Syntax:{' '}
              <Code>{'SIGNAL <type> <name> { fields }'}</Code>
            </Hint>
            {signals.map((type) => (
              <TypeDetail
                key={type}
                name={type}
                description={SIGNAL_DESCRIPTIONS[type]}
                fields={getSignalFieldSchema(type)}
              />
            ))}
          </Section>
        ) : null}
        {showRoutes ? (
          <Section title="Routes and WHEN expressions" icon={Split} forceOpen={searching}>
            <Hint>
              Routes define decision logic. Syntax:{' '}
              <Code>{'ROUTE <name> (description = "...") { ... }'}</Code>
            </Hint>
            <Table head={['Clause', 'Required', 'Description']} rows={ROUTE_CLAUSES} />
            <Hint>
              Combine signal references with <Code>AND</Code>, <Code>OR</Code>, <Code>NOT</Code> and
              parentheses.
            </Hint>
            <Table head={['Operator', 'Precedence', 'Example']} rows={OPERATORS} />
            <Snippet
              code={`WHEN keyword("urgent") AND (domain("math") OR domain("code")) AND NOT keyword("casual")`}
              label="Complex WHEN example"
            />
            <Table head={['Attribute', 'Type', 'Description']} rows={MODEL_ATTRIBUTES} />
            <Snippet
              code={`MODEL "qwen3:70b" (reasoning = true, effort = "high", param_size = "70b"),\n      "qwen2.5:3b" (reasoning = false, param_size = "3b")`}
              label="Multi-model example"
            />
          </Section>
        ) : null}
        {algorithms.length ? (
          <Section
            title={`Algorithms · ${ALGORITHM_TYPES.length} types`}
            icon={Calculator}
            forceOpen={searching}
          >
            <Hint>
              Algorithms choose among a route&apos;s models. Syntax:{' '}
              <Code>{'ALGORITHM <type> { fields }'}</Code> inside a ROUTE.
            </Hint>
            {algorithms.map((type) => (
              <TypeDetail
                key={type}
                name={type}
                description={ALGORITHM_DESCRIPTIONS[type]}
                fields={getAlgorithmFieldSchema(type)}
              />
            ))}
          </Section>
        ) : null}
        {plugins.length ? (
          <Section
            title={`Plugins · ${PLUGIN_TYPES.length} types`}
            icon={Puzzle}
            forceOpen={searching}
          >
            <Hint>
              Plugins add pre- and post-processing. Declare with{' '}
              <Code>{'PLUGIN <name> <type> { fields }'}</Code> and reference it in routes with{' '}
              <Code>{'PLUGIN <name>'}</Code>.
            </Hint>
            {plugins.map((type) => (
              <TypeDetail
                key={type}
                name={type}
                description={PLUGIN_DESCRIPTIONS[type]}
                fields={getPluginFieldSchema(type)}
              />
            ))}
          </Section>
        ) : null}
        {showCheat ? (
          <Section title="Cheat sheet" icon={ClipboardList} forceOpen={searching}>
            <dl className="grid gap-1.5">
              {CHEAT_SHEET.map(([syntax, meaning]) => (
                <div
                  key={syntax}
                  className="flex flex-col gap-0.5 rounded-md bg-surface-subtle px-2.5 py-1.5"
                >
                  <dt className="font-mono text-[11.5px] text-foreground">{syntax}</dt>
                  <dd className="text-xs text-muted-foreground">{meaning}</dd>
                </div>
              ))}
            </dl>
          </Section>
        ) : null}
        {nothing ? (
          <div className="flex flex-col items-center gap-2 px-6 py-12 text-center">
            <BookOpenText className="size-5 text-subtle-foreground" />
            <p className="text-[13px] text-muted-foreground">
              Nothing in the guide matches “{query}”.
            </p>
          </div>
        ) : null}
      </div>
    </div>
  )
}
