import * as React from 'react'
import { FileCode2, RefreshCw } from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  Card,
  CopyButton,
  Segmented,
  Skeleton,
  Textarea,
  cn,
  toast,
} from '../ui'
import { contractOrdered } from './configPageRouterSettingsModel'
import type { JsonRecord } from './configPageRouterSettingsDraft'
import { toYaml, yamlDiff } from './configPageRouterSettingsYaml'

function YamlText({ text }: { text: string }) {
  const match = text.match(/^(\s*(?:- )?)([^:#]+?)(:)(.*)$/)
  if (!match) return <>{text || ' '}</>
  const [, lead, key, colon, rest] = match
  return (
    <>
      {lead}
      <span className="text-chart-1">{key}</span>
      {colon}
      <span className={cn(/^\s*"/.test(rest) && 'text-success-foreground')}>{rest}</span>
    </>
  )
}

function OverridesView({
  saved,
  draft,
  dirty,
  onEditForm,
}: {
  saved: JsonRecord
  draft: JsonRecord
  dirty: boolean
  onEditForm: () => void
}) {
  const yaml = React.useMemo(() => toYaml({ global: contractOrdered(draft) }), [draft])
  const lines = React.useMemo(() => {
    const [hunk] = yamlDiff(
      { global: contractOrdered(saved) },
      { global: contractOrdered(draft) },
      Number.MAX_SAFE_INTEGER,
    )
    if (!hunk) return yaml.split('\n').map((text) => ({ text, changed: false }))
    return hunk.lines
      .filter((line) => line.kind !== 'remove')
      .map((line) => ({ text: line.text, changed: line.kind === 'add' }))
  }, [draft, saved, yaml])
  const keys = lines.length - 1
  return (
    <Card className="overflow-hidden">
      <div className="flex h-10 items-center gap-2 border-b border-border bg-surface-subtle/70 pl-3.5 pr-1.5">
        <FileCode2 className="size-4 text-subtle-foreground" />
        <span className="font-mono text-xs font-medium">config.yaml</span>
        <span className="text-xs text-subtle-foreground">· global · {keys} lines</span>
        {dirty ? (
          <Badge tone="brand" size="sm" className="ml-1">
            unsaved draft
          </Badge>
        ) : null}
        <CopyButton value={yaml} className="ml-auto" label="Copy YAML" />
      </div>
      <pre className="max-h-[70vh] overflow-auto py-3 font-mono text-[12.5px] leading-[1.7] text-foreground/90">
        <code className="grid min-w-fit">
          {lines.map((line, index) => (
            <span key={index} className={cn('flex pr-4', line.changed && 'bg-primary-soft/60')}>
              <span className={cn('w-1 shrink-0', line.changed && 'bg-primary')} aria-hidden />
              <span className="mr-4 w-9 shrink-0 select-none text-right text-subtle-foreground/70 tabular">
                {index + 1}
              </span>
              <span className="whitespace-pre">
                <YamlText text={line.text} />
              </span>
            </span>
          ))}
        </code>
      </pre>
      <div className="border-t border-border bg-surface-subtle/60 px-4 py-2.5 text-xs text-muted-foreground">
        What config.yaml sets, including unsaved edits. Keys left out use their router defaults.{' '}
        <button
          type="button"
          className="font-medium text-primary hover:underline"
          onClick={onEditForm}
        >
          Edit in the form
        </button>
        .
      </div>
    </Card>
  )
}

function EffectiveEditor({
  yaml,
  loading,
  error,
  readOnly,
  blocked,
  onReload,
  onSave,
}: {
  yaml: string
  loading: boolean
  error: string | null
  readOnly: boolean
  blocked: boolean
  onReload: () => void
  onSave: (yaml: string) => Promise<void>
}) {
  const [text, setText] = React.useState(yaml)
  const [saving, setSaving] = React.useState(false)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  React.useEffect(() => setText(yaml), [yaml])
  const changed = text !== yaml
  return (
    <Card className="overflow-hidden">
      <div className="flex h-10 items-center gap-2 border-b border-border bg-surface-subtle/70 pl-3.5 pr-1.5">
        <FileCode2 className="size-4 text-subtle-foreground" />
        <span className="font-mono text-xs font-medium">global</span>
        <span className="text-xs text-subtle-foreground">
          · router defaults merged with config.yaml
        </span>
        {changed ? (
          <Badge tone="brand" size="sm" className="ml-1">
            edited
          </Badge>
        ) : null}
        <span className="ml-auto flex items-center gap-1">
          <Button variant="ghost" size="xs" onClick={onReload} disabled={loading || saving}>
            <RefreshCw className={cn(loading && 'animate-spin')} />
            Reload
          </Button>
          {!readOnly ? (
            <Button
              size="xs"
              disabled={!changed || saving || loading || blocked}
              loading={saving}
              onClick={async () => {
                setSaving(true)
                setSaveError(null)
                try {
                  await onSave(text)
                } catch (cause) {
                  const message =
                    cause instanceof Error ? cause.message : 'Failed to save the global YAML.'
                  setSaveError(message)
                  toast.error('The global YAML was not saved', { description: message })
                } finally {
                  setSaving(false)
                }
              }}
            >
              Save YAML
            </Button>
          ) : null}
        </span>
      </div>
      <div className="flex flex-col gap-3 p-3">
        {error || saveError ? (
          <Alert
            tone="danger"
            title={
              saveError
                ? 'The router did not accept this YAML'
                : 'The effective global YAML could not be loaded'
            }
          >
            <span className="whitespace-pre-wrap break-words">{saveError ?? error}</span>
          </Alert>
        ) : null}
        {blocked && !readOnly ? (
          <Alert tone="info" title="The form has unsaved changes">
            Save or discard them before you replace the whole global block here.
          </Alert>
        ) : null}
        {loading && !yaml ? (
          <div className="flex flex-col gap-2 p-2">
            {Array.from({ length: 8 }).map((_, index) => (
              <Skeleton
                key={index}
                className="h-3.5"
                style={{ width: `${40 + ((index * 23) % 50)}%` }}
              />
            ))}
          </div>
        ) : (
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            readOnly={readOnly || saving}
            spellCheck={false}
            aria-label="Effective global YAML"
            rows={24}
            className="min-h-[420px] font-mono text-[12.5px] leading-[1.7]"
          />
        )}
        <p className="text-xs text-muted-foreground">
          Saving replaces the global block of config.yaml with exactly this YAML, so every key shown
          here becomes explicit. Empty content removes the block; the router defaults stay in
          effect.
        </p>
      </div>
    </Card>
  )
}

export function RouterSettingsYamlPanel({
  saved,
  draft,
  dirty,
  readOnly,
  rawYaml,
  rawLoading,
  rawError,
  onReloadRaw,
  onSaveRaw,
  onEditForm,
}: {
  saved: JsonRecord
  draft: JsonRecord
  dirty: boolean
  readOnly: boolean
  rawYaml: string
  rawLoading: boolean
  rawError: string | null
  onReloadRaw: () => void
  onSaveRaw: (yaml: string) => Promise<void>
  onEditForm: () => void
}) {
  const [mode, setMode] = React.useState<'overrides' | 'effective'>('overrides')
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          size="sm"
          aria-label="YAML view"
          value={mode}
          onValueChange={setMode}
          options={[
            { value: 'overrides', label: 'What config.yaml sets' },
            { value: 'effective', label: 'Effective global' },
          ]}
        />
        <span className="text-xs text-muted-foreground">
          {mode === 'overrides'
            ? 'Read-only draft of the global section.'
            : 'Advanced: edit the merged block as raw YAML.'}
        </span>
      </div>
      {mode === 'overrides' ? (
        <OverridesView saved={saved} draft={draft} dirty={dirty} onEditForm={onEditForm} />
      ) : (
        <EffectiveEditor
          yaml={rawYaml}
          loading={rawLoading}
          error={rawError}
          readOnly={readOnly}
          blocked={dirty}
          onReload={onReloadRaw}
          onSave={onSaveRaw}
        />
      )}
    </div>
  )
}
