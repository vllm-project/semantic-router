import * as React from 'react'
import { FileCode2, PencilLine, RefreshCw } from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  Card,
  CopyButton,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Skeleton,
  Textarea,
  cn,
  toast,
} from '../ui'
import {
  ROUTER_CONFIG_DOCUMENT_SCHEMA,
  configYaml,
  configYamlDiff,
  configYamlLines,
} from '../lib/configYaml'
import type { JsonRecord } from './configPageRouterSettingsDraft'

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
  readOnly,
  onEditForm,
  onOpenRaw,
}: {
  saved: JsonRecord
  draft: JsonRecord
  dirty: boolean
  readOnly: boolean
  onEditForm: () => void
  onOpenRaw: () => void
}) {
  const yaml = React.useMemo(
    () => configYaml({ global: draft }, ROUTER_CONFIG_DOCUMENT_SCHEMA),
    [draft],
  )
  const lines = React.useMemo(() => {
    const [hunk] = configYamlDiff(
      { global: saved },
      { global: draft },
      { schema: ROUTER_CONFIG_DOCUMENT_SCHEMA, context: Infinity },
    )
    if (!hunk) {
      return configYamlLines({ global: draft }, ROUTER_CONFIG_DOCUMENT_SCHEMA).map((line) => ({
        text: line.text,
        changed: false,
      }))
    }
    return hunk.lines
      .filter((line) => line.kind !== 'remove')
      .map((line) => ({ text: line.text, changed: line.kind === 'add' }))
  }, [draft, saved])
  return (
    <Card className="overflow-hidden">
      <div className="flex h-10 items-center gap-2 border-b border-border bg-surface-subtle/70 pl-3.5 pr-1.5">
        <FileCode2 className="size-4 text-subtle-foreground" />
        <span className="font-mono text-xs font-medium">config.yaml</span>
        <span className="text-xs text-subtle-foreground">· global · {lines.length} lines</span>
        {dirty ? (
          <Badge tone="brand" size="sm" className="ml-1">
            unsaved draft
          </Badge>
        ) : null}
        <span className="ml-auto flex items-center gap-1">
          <Button variant="ghost" size="xs" onClick={onOpenRaw}>
            <PencilLine />
            {readOnly ? 'Effective YAML' : 'Edit effective YAML'}
          </Button>
          <CopyButton value={yaml} label="Copy YAML" />
        </span>
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

function EffectiveYamlDialog({
  open,
  onOpenChange,
  yaml,
  loading,
  error,
  readOnly,
  blocked,
  onReload,
  onSave,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
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
  React.useEffect(() => {
    if (!open) setSaveError(null)
  }, [open])
  const changed = text !== yaml

  const save = async () => {
    setSaving(true)
    setSaveError(null)
    try {
      await onSave(text)
      onOpenChange(false)
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Failed to save the global YAML.'
      setSaveError(message)
      toast.error('The global YAML was not saved', { description: message })
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(next) => !saving && onOpenChange(next)}>
      <DialogContent size="xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Effective global YAML
            {changed ? (
              <Badge tone="brand" size="sm">
                edited
              </Badge>
            ) : null}
          </DialogTitle>
          <DialogDescription>
            Router defaults merged with config.yaml. Saving replaces the global block with exactly
            this YAML, so every key shown becomes explicit; empty content removes the block.
          </DialogDescription>
        </DialogHeader>
        <DialogBody className="flex flex-col gap-3 pb-4">
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
              rows={22}
              className="min-h-[min(52vh,460px)] font-mono text-[12.5px] leading-[1.7]"
            />
          )}
        </DialogBody>
        <DialogFooter>
          <Button
            variant="ghost"
            className="mr-auto"
            onClick={onReload}
            disabled={loading || saving}
          >
            <RefreshCw className={cn(loading && 'animate-spin')} />
            Reload
          </Button>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={saving}>
            {readOnly ? 'Close' : 'Cancel'}
          </Button>
          {!readOnly ? (
            <Button
              disabled={!changed || saving || loading || blocked}
              loading={saving}
              onClick={() => void save()}
            >
              Save YAML
            </Button>
          ) : null}
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
  const [rawOpen, setRawOpen] = React.useState(false)
  return (
    <>
      <OverridesView
        saved={saved}
        draft={draft}
        dirty={dirty}
        readOnly={readOnly}
        onEditForm={onEditForm}
        onOpenRaw={() => {
          setRawOpen(true)
          onReloadRaw()
        }}
      />
      <EffectiveYamlDialog
        open={rawOpen}
        onOpenChange={setRawOpen}
        yaml={rawYaml}
        loading={rawLoading}
        error={rawError}
        readOnly={readOnly}
        blocked={dirty}
        onReload={onReloadRaw}
        onSave={onSaveRaw}
      />
    </>
  )
}
