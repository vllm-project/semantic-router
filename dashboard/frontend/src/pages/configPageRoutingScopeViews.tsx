import * as React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { CornerDownLeft, Play, Radar, Split } from 'lucide-react'

import { diffLines } from '../lib/configYaml'
import {
  Alert,
  AnimatedNumber,
  Badge,
  Button,
  CodeView,
  CopyButton,
  HighlightedText,
  Kbd,
  Label,
  ScoreMeter,
  Segmented,
  Skeleton,
  StatusDot,
  Textarea,
  cn,
} from '../ui'
import { testQueryDryRun } from './topology/utils/api'
import type { TestQueryResult } from './topology/types'
import { signalTypeIcon, signalTypeLabel } from './configPageSignalCatalog'
import { keywordOccurrences, signalThreshold } from './configPageSignalEvaluator'

const ease = [0.16, 1, 0.3, 1] as const

/** A signal reference: its type icon in the signal tone and the name users copy. */
export function SignalChip({
  type,
  name,
  label,
  negated = false,
  className,
}: {
  type: string
  name: string
  label?: string
  negated?: boolean
  className?: string
}) {
  const Icon = signalTypeIcon(type)
  return (
    <span
      title={`${signalTypeLabel(type)} · ${name}${label ? ` · ${label}` : ''}`}
      className={cn(
        'inline-flex h-6 min-w-0 max-w-[220px] items-center gap-1 rounded-md border bg-surface-subtle px-1.5 text-xs',
        negated ? 'border-destructive/30' : 'border-border',
        className,
      )}
    >
      {negated ? (
        <span className="font-mono text-[10px] font-semibold text-destructive-foreground">NOT</span>
      ) : null}
      <Icon
        className={cn('size-3 shrink-0', type === 'projection' ? 'text-chart-4' : 'text-chart-2')}
      />
      <span className="truncate font-mono text-[11.5px]">{name || '?'}</span>
      {label ? (
        <span className="shrink-0 font-mono text-[11px] text-subtle-foreground">:{label}</span>
      ) : null}
    </span>
  )
}

function DiffView({ before, after }: { before: string; after: string }) {
  const changes = React.useMemo(() => diffLines(before, after), [before, after])
  const changed = changes.some((change) => change.kind !== 'same')
  if (!changed) {
    return (
      <p className="px-4 py-6 text-center text-[13px] text-muted-foreground">
        No changes from the saved configuration.
      </p>
    )
  }
  return (
    <pre className="overflow-x-auto py-3 font-mono text-[12.5px] leading-[1.7]">
      <code className="grid">
        {changes.map((change, index) => (
          <span
            key={index}
            className={cn(
              'flex whitespace-pre px-4',
              change.kind === 'add' && 'bg-success-soft text-foreground',
              change.kind === 'remove' && 'bg-destructive-soft text-foreground/80',
              change.kind === 'same' && 'text-foreground/70',
            )}
          >
            <span aria-hidden className="mr-3 w-3 shrink-0 select-none text-subtle-foreground">
              {change.kind === 'add' ? '+' : change.kind === 'remove' ? '−' : ' '}
            </span>
            {change.text || ' '}
          </span>
        ))}
      </code>
    </pre>
  )
}

/** Canonical YAML of an entity, with the changes a draft makes to the saved one. */
export function EntityYamlPanel({
  path,
  yaml,
  savedYaml,
  note,
}: {
  path: string
  yaml: string
  savedYaml?: string
  note?: React.ReactNode
}) {
  const dirty = savedYaml !== undefined && savedYaml !== yaml
  const [view, setView] = React.useState<'yaml' | 'changes'>('yaml')
  React.useEffect(() => {
    if (!dirty) setView('yaml')
  }, [dirty])
  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-hidden rounded-xl border border-border bg-surface-sunken/60">
        <div className="flex h-10 items-center justify-between gap-2 border-b border-border pl-3 pr-1.5 text-xs text-muted-foreground">
          <span className="min-w-0 truncate font-mono">{path}</span>
          <span className="flex shrink-0 items-center gap-1.5">
            {dirty ? (
              <Segmented
                size="sm"
                aria-label="YAML view"
                value={view}
                onValueChange={setView}
                options={[
                  { value: 'yaml', label: 'Draft' },
                  { value: 'changes', label: 'Changes' },
                ]}
              />
            ) : null}
            <CopyButton value={yaml} label="Copy YAML" />
          </span>
        </div>
        {view === 'changes' && savedYaml !== undefined ? (
          <DiffView before={savedYaml} after={yaml} />
        ) : (
          <CodeView code={yaml} language="yaml" lineNumbers />
        )}
      </div>
      {note ? <p className="text-xs text-muted-foreground">{note}</p> : null}
    </div>
  )
}

function RuleEvidence({
  rule,
  prompt,
  hit,
}: {
  rule: NonNullable<TryTarget['rule']>
  prompt: string
  hit?: TestQueryResult['matchedSignals'][number]
}) {
  const threshold = signalThreshold(rule.record)
  const confidence = typeof hit?.confidence === 'number' ? hit.confidence : undefined
  const occurrences = rule.type === 'keyword' ? keywordOccurrences(prompt, rule.record) : null
  if (confidence === undefined && !occurrences) return null
  return (
    <div className="flex flex-col gap-3 border-b border-border px-4 py-3.5">
      {confidence !== undefined ? (
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs text-subtle-foreground">
            <span>Confidence</span>
            <AnimatedNumber value={confidence} className="font-mono text-foreground" />
          </div>
          <ScoreMeter
            aria-label={`${rule.name} confidence`}
            value={confidence}
            threshold={threshold}
            matched={Boolean(hit)}
            tone="gold"
          />
          {threshold !== undefined ? (
            <span className="font-mono text-[10.5px] text-subtle-foreground">
              threshold {threshold}
            </span>
          ) : null}
        </div>
      ) : null}
      {occurrences ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">
            {occurrences.length
              ? `Keyword occurrences in the prompt (${occurrences.length})`
              : 'No keyword occurs in the prompt'}
          </span>
          {occurrences.length ? (
            <div className="rounded-lg border border-border bg-surface-subtle/60 px-3 py-2.5">
              <HighlightedText text={prompt} ranges={occurrences} />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function matchesTarget(name: string, target: string): boolean {
  return name === target || name.startsWith(`${target}:`)
}

export interface TryTarget {
  /** Signal references to highlight, such as one rule or the bands of a mapping. */
  signals?: Array<{ type: string; name: string }>
  /** A decision to report on. */
  decision?: string
  /** The one rule being tried, for its score against its threshold and keyword occurrences. */
  rule?: { type: string; name: string; record: Record<string, unknown> }
}

/**
 * Sends a sample prompt through the running Router with the saved
 * configuration (POST /api/topology/test-query, dry run) and reports what
 * matched. Drafts are not evaluated until they are saved.
 */
export function TryPromptPanel({
  entrypointModel,
  target,
  dirty,
  placeholder = 'Paste a prompt a user might send…',
}: {
  entrypointModel?: string
  target: TryTarget
  dirty: boolean
  placeholder?: string
}) {
  const [prompt, setPrompt] = React.useState('')
  const [running, setRunning] = React.useState(false)
  const [result, setResult] = React.useState<TestQueryResult | null>(null)
  const [ranPrompt, setRanPrompt] = React.useState('')
  const [error, setError] = React.useState<string | null>(null)
  const run = async () => {
    if (!prompt.trim() || running) return
    setRunning(true)
    setError(null)
    try {
      setResult(await testQueryDryRun(prompt.trim(), entrypointModel))
      setRanPrompt(prompt.trim())
    } catch (failure) {
      setResult(null)
      setError(failure instanceof Error ? failure.message : 'The Router preview failed.')
    } finally {
      setRunning(false)
    }
  }
  const unavailable = result && !result.isAccurate
  const targetSignals = target.signals ?? []
  const hits = result
    ? result.matchedSignals.filter((signal) =>
        targetSignals.some(
          (candidate) =>
            candidate.type === signal.type && matchesTarget(signal.name, candidate.name),
        ),
      )
    : []
  return (
    <div className="flex flex-col gap-4">
      {dirty ? (
        <Alert tone="warning" title="Unsaved edits are not evaluated">
          The Router evaluates the saved configuration. Save to try your changes.
        </Alert>
      ) : null}
      <div className="flex flex-col gap-2">
        <Label htmlFor="try-prompt">Prompt</Label>
        <Textarea
          id="try-prompt"
          rows={3}
          value={prompt}
          placeholder={placeholder}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
              event.preventDefault()
              void run()
            }
          }}
        />
        <div className="flex items-center gap-2">
          <span className="text-xs text-subtle-foreground">
            Runs a dry-run route preview
            {entrypointModel ? (
              <>
                {' '}
                for <span className="font-mono">{entrypointModel}</span>
              </>
            ) : null}
            ; no model is called.
          </span>
          <span className="ml-auto flex items-center gap-1 text-[11px] text-subtle-foreground">
            <Kbd>⌘</Kbd>
            <Kbd>
              <CornerDownLeft className="size-3" />
            </Kbd>
          </span>
          <Button size="sm" onClick={() => void run()} loading={running} disabled={!prompt.trim()}>
            {running ? null : <Play />}
            Run
          </Button>
        </div>
      </div>
      <AnimatePresence mode="wait" initial={false}>
        {running && !result ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col gap-3 rounded-xl border border-border p-4"
          >
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-2/3" />
          </motion.div>
        ) : error ? (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease }}
          >
            <Alert tone="danger" title="The preview could not run">
              {error}
            </Alert>
          </motion.div>
        ) : unavailable ? (
          <motion.div
            key="unavailable"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease }}
          >
            <Alert tone="warning" title="The Router is not answering">
              {result?.warning || 'Live preview could not complete.'} Start the Router with{' '}
              <span className="font-mono">vllm-sr serve</span>, then run the prompt again.
            </Alert>
          </motion.div>
        ) : result ? (
          <motion.div
            key={`${result.query}-${result.matchedDecision}`}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: running ? 0.55 : 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease }}
            className="overflow-hidden rounded-xl border border-border bg-card shadow-xs"
          >
            {targetSignals.length ? (
              <div className="flex items-center gap-2.5 border-b border-border px-4 py-3">
                <StatusDot tone={hits.length ? 'success' : 'neutral'} />
                <span className="text-[13px] font-semibold">
                  {hits.length ? 'Matched' : 'Did not match'}
                </span>
                <span className="flex min-w-0 flex-wrap gap-1">
                  {hits.map((hit) => (
                    <SignalChip key={`${hit.type}:${hit.name}`} type={hit.type} name={hit.name} />
                  ))}
                </span>
                {typeof result.routingLatency === 'number' ? (
                  <span className="ml-auto shrink-0 font-mono text-[11px] text-subtle-foreground">
                    <AnimatedNumber
                      value={result.routingLatency}
                      format={(value) => `${Math.round(value)} ms`}
                    />
                  </span>
                ) : null}
              </div>
            ) : null}
            {target.rule ? (
              <RuleEvidence rule={target.rule} prompt={ranPrompt} hit={hits[0]} />
            ) : null}
            <div className="flex flex-col gap-3 px-4 py-3.5">
              <div className="flex flex-wrap items-center gap-2 text-[13px]">
                <Split className="size-3.5 text-chart-3" />
                <span className="text-muted-foreground">Decision</span>
                {result.matchedDecision ? (
                  <Badge
                    tone={
                      target.decision && target.decision !== result.matchedDecision
                        ? 'outline'
                        : 'violet'
                    }
                    className="font-mono"
                  >
                    {result.matchedDecision}
                  </Badge>
                ) : (
                  <span className="text-subtle-foreground">none matched</span>
                )}
                {target.decision &&
                result.matchedDecision &&
                target.decision !== result.matchedDecision ? (
                  <span className="text-xs text-muted-foreground">
                    won instead of <span className="font-mono">{target.decision}</span>
                  </span>
                ) : null}
                {result.selectedModel ? (
                  <span className="text-xs text-muted-foreground">
                    → <span className="font-mono text-foreground/85">{result.selectedModel}</span>
                  </span>
                ) : null}
              </div>
              <div className="flex flex-col gap-1.5">
                <span className="flex items-center gap-1.5 text-xs text-subtle-foreground">
                  <Radar className="size-3" />
                  {result.matchedSignals.length} signals matched
                </span>
                {result.matchedSignals.length ? (
                  <div className="flex flex-wrap gap-1">
                    {result.matchedSignals.map((signal) => (
                      <span
                        key={`${signal.type}:${signal.name}`}
                        className="inline-flex items-center gap-1"
                      >
                        <SignalChip type={signal.type} name={signal.name} />
                        {typeof signal.confidence === 'number' ? (
                          <span className="font-mono text-[11px] tabular text-subtle-foreground">
                            {signal.confidence.toFixed(2)}
                          </span>
                        ) : null}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
              {result.signalErrors && Object.keys(result.signalErrors).length ? (
                <Alert tone="warning" title="Some signals could not be evaluated">
                  {Object.entries(result.signalErrors)
                    .map(([key, message]) => `${key}: ${message}`)
                    .join(' · ')}
                </Alert>
              ) : null}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  )
}
