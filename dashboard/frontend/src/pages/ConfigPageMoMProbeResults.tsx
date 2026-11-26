import { Check, ShieldCheck, ShieldQuestion, X } from 'lucide-react'

import type {
  RecipeProbeDetail,
  RecipeProbeExpectedRoute,
  RecipeProbeValidationResult,
} from '../types/recipe'
import { Badge, KeyValueGrid, SectionLabel, cn } from '../ui'
import { validationOutcome } from './configPageMoMProbesSupport'

function formatContent(content: unknown): string {
  if (typeof content === 'string') return content
  return JSON.stringify(content, null, 2) ?? String(content ?? '')
}

function compactDigest(value?: string): string {
  if (!value) return '—'
  return value.length > 24 ? `${value.slice(0, 15)}…${value.slice(-8)}` : value
}

const preClass =
  'max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-surface-sunken/60 px-3 py-2.5 font-mono text-[12px] leading-relaxed text-foreground/90'

export function ValidationBadge({ result }: { result?: RecipeProbeValidationResult }) {
  if (!result) return <span className="text-xs text-subtle-foreground">Not validated</span>
  const outcome = validationOutcome(result)
  return (
    <span className="inline-flex items-center gap-2">
      <Badge tone={outcome === 'pass' ? 'success' : outcome === 'mismatch' ? 'danger' : 'warning'}>
        {outcome === 'pass' ? <Check /> : outcome === 'mismatch' ? <X /> : null}
        {outcome === 'pass' ? 'Pass' : outcome === 'mismatch' ? 'Mismatch' : 'Unavailable'}
      </Badge>
      <span className="font-mono text-[11px] tabular text-subtle-foreground">
        {result.latency_ms} ms
      </span>
    </span>
  )
}

export function ExpectedRoute({ expected }: { expected: RecipeProbeExpectedRoute }) {
  const signals = Object.entries(expected.signals ?? {}).flatMap(([type, names]) =>
    names.map((name) => `${type}:${name}`),
  )
  const forbiddenSignals = Object.entries(expected.forbidden_signals ?? {}).flatMap(
    ([type, names]) => names.map((name) => `${type}:${name}`),
  )
  return (
    <KeyValueGrid
      items={[
        { label: 'Decision', value: expected.decision || '—', mono: true },
        { label: 'Recipe', value: expected.recipe || 'default', mono: true },
        {
          label: 'Selection',
          value: expected.selection_status?.replace(/_/g, ' ') || 'any',
        },
        {
          label: 'Algorithm or alias',
          value: expected.algorithm || expected.alias || 'route only',
          mono: Boolean(expected.algorithm || expected.alias),
        },
        ...(signals.length
          ? [
              {
                label: 'Signals that must match',
                value: (
                  <span className="flex flex-wrap gap-1">
                    {signals.map((signal) => (
                      <Badge key={signal} size="sm" tone="gold" className="font-mono">
                        {signal}
                      </Badge>
                    ))}
                  </span>
                ),
                span: true,
              },
            ]
          : []),
        ...(forbiddenSignals.length
          ? [
              {
                label: 'Signals that must not match',
                value: (
                  <span className="flex flex-wrap gap-1">
                    {forbiddenSignals.map((signal) => (
                      <Badge
                        key={signal}
                        size="sm"
                        tone="outline"
                        className="font-mono line-through"
                      >
                        {signal}
                      </Badge>
                    ))}
                  </span>
                ),
                span: true,
              },
            ]
          : []),
        ...(expected.plugins.length
          ? [{ label: 'Plugins', value: expected.plugins.join(', '), mono: true }]
          : []),
        ...(expected.forbidden_plugins.length
          ? [
              {
                label: 'Forbidden plugins',
                value: expected.forbidden_plugins.join(', '),
                mono: true,
              },
            ]
          : []),
      ]}
    />
  )
}

export function RecipeProbeDetailPanel({ detail }: { detail: RecipeProbeDetail }) {
  const imageFixtures = Object.entries(detail.image_fixtures ?? {})
  const materialization =
    detail.repeat > 1 || detail.padding || detail.generated_text || imageFixtures.length > 0
  return (
    <div className="flex flex-col gap-4">
      {detail.notes ? (
        <p className="text-[13px] leading-relaxed text-muted-foreground">{detail.notes}</p>
      ) : null}
      {detail.display_prompt ? (
        <section className="flex flex-col gap-1.5">
          <SectionLabel>Demo prompt</SectionLabel>
          <pre className={preClass}>{detail.display_prompt}</pre>
        </section>
      ) : null}
      {detail.raw_payload_hidden ? (
        <section role="note" className="flex flex-col gap-1.5">
          <SectionLabel>Exact payload</SectionLabel>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Hidden because this Eval-only fixture contains synthetic stress data. Validate sends the
            exact bounded payload directly to Router without rendering it in the browser.
          </p>
        </section>
      ) : null}
      {detail.query ? (
        <section className="flex flex-col gap-1.5">
          <SectionLabel>Query</SectionLabel>
          <pre className={preClass}>{detail.query}</pre>
        </section>
      ) : null}
      {detail.messages?.length ? (
        <section className="flex flex-col gap-1.5">
          <SectionLabel>Messages · {detail.messages.length}</SectionLabel>
          <div className="flex flex-col gap-2">
            {detail.messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className="flex flex-col gap-1">
                <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                  {message.role}
                </span>
                <pre className={preClass}>{formatContent(message.content)}</pre>
              </div>
            ))}
          </div>
        </section>
      ) : null}
      {detail.tools?.length ? (
        <section className="flex flex-col gap-1.5">
          <SectionLabel>Tools · {detail.tools.length}</SectionLabel>
          <pre className={preClass}>{JSON.stringify(detail.tools, null, 2)}</pre>
        </section>
      ) : null}
      {materialization ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Materialization</SectionLabel>
          <KeyValueGrid
            items={[
              ...(detail.repeat > 1 || detail.padding
                ? [{ label: 'Query repeat', value: `${detail.repeat || 1}×` }]
                : []),
              ...(detail.padding
                ? [
                    { label: 'Padding repeat', value: `${detail.padding.repeat}×` },
                    { label: 'Placement', value: detail.padding.placement },
                  ]
                : []),
              ...(detail.generated_text
                ? [
                    {
                      label: 'Generated text target',
                      value: `${detail.generated_text.target_text_bytes.toLocaleString()} bytes`,
                    },
                    {
                      label:
                        detail.generated_text.text !== undefined
                          ? 'Repeating text'
                          : 'Filler character',
                      value: JSON.stringify(
                        detail.generated_text.text ?? detail.generated_text.character,
                      ),
                      mono: true,
                    },
                    {
                      label: 'Insertion point (1-based)',
                      value: `message ${detail.generated_text.message_index + 1}, content ${detail.generated_text.content_index + 1}`,
                    },
                  ]
                : []),
              ...imageFixtures.map(([name, fixture]) => ({
                label: `Image fixture · ${name}`,
                value: (
                  <span className="flex flex-col">
                    <span>{fixture.description}</span>
                    <span className="font-mono text-[11.5px] text-subtle-foreground">
                      {fixture.media_type} · {fixture.bytes.toLocaleString()} bytes ·{' '}
                      {compactDigest(fixture.sha256)}
                    </span>
                  </span>
                ),
                span: true,
              })),
            ]}
          />
        </section>
      ) : null}
    </div>
  )
}

export function RecipeProbeValidation({ result }: { result: RecipeProbeValidationResult }) {
  const outcome = validationOutcome(result)
  const finalModel = result.actual.model?.trim()
  return (
    <div
      className={cn(
        'flex flex-col gap-3 rounded-lg border p-3.5',
        outcome === 'pass' && 'border-success/30 bg-success-soft/40',
        outcome === 'mismatch' && 'border-destructive/30 bg-destructive-soft/40',
        outcome === 'unavailable' && 'border-warning/30 bg-warning-soft/40',
      )}
      role={result.passed ? 'status' : 'alert'}
    >
      <div className="flex items-center justify-between gap-2">
        <strong className="text-[13px] font-medium">
          {outcome === 'unavailable'
            ? 'Validation unavailable'
            : outcome === 'pass'
              ? 'Route validated'
              : 'Route mismatch'}
        </strong>
        <span className="font-mono text-xs tabular text-muted-foreground">
          {result.latency_ms} ms
        </span>
      </div>
      <KeyValueGrid
        className="gap-y-3"
        items={[
          { label: 'Final model', value: finalModel || 'Not evaluated', mono: Boolean(finalModel) },
          {
            label: 'Selection status',
            value: result.actual.selection_status?.replace(/_/g, ' ') || 'not evaluated',
          },
          { label: 'Expected decision', value: result.expected.decision || '—', mono: true },
          { label: 'Actual decision', value: result.actual.decision || '—', mono: true },
          { label: 'Request model', value: result.actual.requested_model || '—', mono: true },
        ]}
      />
      {!finalModel && result.actual.recommended_models.length ? (
        <p className="text-xs text-muted-foreground">
          Candidates: {result.actual.recommended_models.join(', ')}
        </p>
      ) : null}
      {result.actual.selection_reason ? (
        <p className="text-xs text-muted-foreground">{result.actual.selection_reason}</p>
      ) : null}
      {result.provenance ? (
        <div className="flex flex-col gap-2 rounded-md border border-border bg-surface px-3 py-2.5">
          <span className="flex items-center gap-1.5 text-xs font-medium">
            {result.provenance.status === 'verified' ? (
              <ShieldCheck className="size-3.5 text-success" />
            ) : (
              <ShieldQuestion className="size-3.5 text-warning" />
            )}
            {result.provenance.status === 'verified'
              ? 'Runtime provenance verified'
              : 'Runtime provenance unverified'}
            {result.provenance.reason ? (
              <span className="font-normal text-muted-foreground">
                · {result.provenance.reason.replace(/_/g, ' ')}
              </span>
            ) : null}
          </span>
          <KeyValueGrid
            columns={3}
            className="gap-y-2"
            items={[
              {
                label: 'Package',
                value: compactDigest(result.provenance.package_hash),
                mono: true,
              },
              {
                label: 'Runtime before',
                value: compactDigest(result.provenance.before.active_runtime_hash),
                mono: true,
              },
              {
                label: 'Runtime after',
                value: compactDigest(result.provenance.after.active_runtime_hash),
                mono: true,
              },
            ]}
          />
        </div>
      ) : null}
      {result.failures.length ? (
        <ul className="flex list-disc flex-col gap-1 pl-5 text-xs text-foreground/85">
          {result.failures.map((failure, index) => (
            <li key={`${index}-${failure}`}>{failure}</li>
          ))}
        </ul>
      ) : null}
      {result.error ? <p className="text-xs text-foreground/85">{result.error}</p> : null}
    </div>
  )
}
