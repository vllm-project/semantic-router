import * as React from 'react'
import { FileUp, Link2, Server, Upload } from 'lucide-react'
import {
  Alert,
  Button,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  IconTile,
  Input,
  Label,
  Separator,
  Textarea,
} from '../../ui'

interface EditorImportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Replaces the working copy with a pasted, uploaded or fetched router config. */
  onImport: (yaml: string) => Promise<void>
  /** Replaces the working copy with the config the Router runs now. */
  onLoadFromRouter: () => Promise<void>
}

function errorText(prefix: string, error: unknown): string {
  return `${prefix}: ${error instanceof Error ? error.message : String(error)}`
}

/** Import a router config into the editor from text, a file, a URL or the running Router. */
export function EditorImportDialog({
  open,
  onOpenChange,
  onImport,
  onLoadFromRouter,
}: EditorImportDialogProps) {
  const [text, setText] = React.useState('')
  const [url, setUrl] = React.useState('')
  const [error, setError] = React.useState<string | null>(null)
  const [busy, setBusy] = React.useState<'import' | 'fetch' | 'router' | null>(null)
  const fileInput = React.useRef<HTMLInputElement>(null)
  const textArea = React.useRef<HTMLTextAreaElement>(null)

  const [wasOpen, setWasOpen] = React.useState(open)
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) {
      setText('')
      setUrl('')
      setError(null)
      setBusy(null)
    }
  }

  const importText = async () => {
    const yaml = text.trim()
    if (!yaml) {
      setError('Paste a router config, choose a file or fetch one from a URL first.')
      return
    }
    setBusy('import')
    setError(null)
    try {
      await onImport(yaml)
      onOpenChange(false)
    } catch (failure) {
      setError(errorText('Failed to import YAML', failure))
    } finally {
      setBusy(null)
    }
  }

  const fetchUrl = async () => {
    const target = url.trim()
    if (!target) {
      setError('Enter the URL of a raw YAML file.')
      return
    }
    try {
      new URL(target)
    } catch {
      setError('Enter a complete URL, such as https://example.com/config.yaml.')
      return
    }
    setBusy('fetch')
    setError(null)
    try {
      const response = await fetch('/api/tools/fetch-raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: target }),
      })
      const data = (await response.json()) as { content?: string; error?: string }
      if (data.error) throw new Error(data.error)
      if (!data.content?.trim()) throw new Error('The URL returned an empty file')
      setText(data.content)
    } catch (failure) {
      setError(errorText('Failed to fetch', failure))
    } finally {
      setBusy(null)
    }
  }

  const loadRouter = async () => {
    setBusy('router')
    setError(null)
    try {
      await onLoadFromRouter()
      onOpenChange(false)
    } catch (failure) {
      setError(errorText('Failed to load from the Router', failure))
    } finally {
      setBusy(null)
    }
  }

  const readFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setText(reader.result)
        setError(null)
      }
    }
    reader.onerror = () => setError(`Could not read ${file.name}.`)
    reader.readAsText(file)
  }

  return (
    <Dialog open={open} onOpenChange={(next) => (busy ? undefined : onOpenChange(next))}>
      <DialogContent
        size="lg"
        onOpenAutoFocus={(event) => {
          event.preventDefault()
          textArea.current?.focus()
        }}
      >
        <DialogHeader className="flex-row items-start gap-3.5 pb-3">
          <IconTile icon={FileUp} tone="violet" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <DialogTitle>Import config</DialogTitle>
            <DialogDescription>
              Replaces the working copy with the routing in a router config. Nothing is deployed
              until you review and deploy it.
            </DialogDescription>
          </div>
        </DialogHeader>
        <DialogBody className="flex flex-col gap-4 pb-5">
          <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface-subtle/60 px-3.5 py-3">
            <div className="flex min-w-0 items-start gap-2.5">
              <Server className="mt-0.5 size-4 shrink-0 text-subtle-foreground" />
              <div className="flex min-w-0 flex-col">
                <span className="text-[13px] font-medium">The running Router</span>
                <span className="text-xs text-muted-foreground">
                  Start again from the config the Router serves now.
                </span>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={loadRouter}
              loading={busy === 'router'}
              disabled={busy !== null && busy !== 'router'}
            >
              Load from Router
            </Button>
          </div>

          <div className="flex items-center gap-3 text-xs text-subtle-foreground">
            <Separator className="flex-1" />
            or bring a file
            <Separator className="flex-1" />
          </div>

          <div className="grid gap-1.5">
            <Label htmlFor="editor-import-url">From a URL</Label>
            <div className="flex gap-2">
              <div className="relative min-w-0 flex-1">
                <Link2 className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-subtle-foreground" />
                <Input
                  id="editor-import-url"
                  aria-label="Router config URL"
                  value={url}
                  onChange={(event) => {
                    setUrl(event.target.value)
                    setError(null)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      void fetchUrl()
                    }
                  }}
                  placeholder="https://raw.githubusercontent.com/…/config.yaml"
                  className="pl-8 font-mono text-[12.5px]"
                />
              </div>
              <Button
                variant="outline"
                onClick={fetchUrl}
                loading={busy === 'fetch'}
                disabled={busy !== null && busy !== 'fetch'}
              >
                Fetch
              </Button>
            </div>
          </div>

          <div className="grid gap-1.5">
            <div className="flex items-center justify-between">
              <Label htmlFor="editor-import-yaml">Router config YAML</Label>
              <Button variant="ghost" size="xs" onClick={() => fileInput.current?.click()}>
                <Upload />
                Choose file
              </Button>
            </div>
            <Textarea
              ref={textArea}
              id="editor-import-yaml"
              aria-label="Router config YAML"
              value={text}
              onChange={(event) => {
                setText(event.target.value)
                setError(null)
              }}
              spellCheck={false}
              placeholder={
                'version: v0.5\nproviders:\n  models:\n    - name: …\nrecipes:\n  - name: default\n    decisions: …'
              }
              className="h-56 resize-y font-mono text-[12px] leading-5"
            />
            <input
              ref={fileInput}
              type="file"
              accept=".yaml,.yml,.json"
              className="hidden"
              aria-hidden
              tabIndex={-1}
              onChange={readFile}
            />
          </div>

          {error ? (
            <div role="alert">
              <Alert tone="danger" title="Import failed">
                <span className="break-words font-mono text-[12px]">{error}</span>
              </Alert>
            </div>
          ) : null}
        </DialogBody>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy !== null}>
            Cancel
          </Button>
          <Button
            onClick={importText}
            loading={busy === 'import'}
            disabled={!text.trim() || (busy !== null && busy !== 'import')}
          >
            Import
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
