import * as React from 'react'
import { Download, FileCode2, Play, RefreshCw } from 'lucide-react'
import { Button, CopyButton, EmptyState, Segmented, Skeleton, Tooltip, cn } from '../../ui'

export type OutputKind = 'yaml' | 'crd'

const YAML_LINE = /^(\s*)(- )?(?:("(?:[^"\\]|\\.)*"|'[^']*'|[^\s:#][^:#]*?)(:)(?=\s|$))?(.*)$/

function highlightValue(text: string): React.ReactNode {
  if (!text) return null
  const comment = text.match(/(^|\s)(#.*)$/)
  const body = comment ? text.slice(0, comment.index! + comment[1].length) : text
  const trimmed = body.trim()
  let value: React.ReactNode = body
  if (/^(["']).*\1$/.test(trimmed)) value = <span className="text-chart-6">{body}</span>
  else if (/^-?\d+(\.\d+)?$/.test(trimmed)) value = <span className="text-chart-7">{body}</span>
  else if (/^(true|false|null|~)$/.test(trimmed))
    value = <span className="text-chart-3">{body}</span>
  return (
    <>
      {value}
      {comment ? <span className="italic text-subtle-foreground">{comment[2]}</span> : null}
    </>
  )
}

function YamlLine({ line }: { line: string }) {
  if (/^\s*#/.test(line)) return <span className="italic text-subtle-foreground">{line}</span>
  const match = YAML_LINE.exec(line)
  if (!match) return <>{line}</>
  const [, indent, dash, key, colon, rest] = match
  return (
    <>
      {indent}
      {dash ? <span className="text-subtle-foreground">{dash}</span> : null}
      {key ? <span className="text-chart-1">{key}</span> : null}
      {colon ? <span className="text-subtle-foreground">{colon}</span> : null}
      {highlightValue(rest)}
    </>
  )
}

/** Highlighted, line-numbered text; the numbers stay out of the copied and searched text. */
export function CodeLines({ code, className }: { code: string; className?: string }) {
  const lines = React.useMemo(() => code.replace(/\n$/, '').split('\n'), [code])
  return (
    <div className={cn('flex min-w-max font-mono text-[12.5px] leading-5', className)}>
      <div
        aria-hidden
        className="sticky left-0 shrink-0 select-none border-r border-border bg-surface py-3 pl-3 pr-3 text-right tabular text-subtle-foreground/70"
      >
        {lines.map((_, index) => (
          <div key={index}>{index + 1}</div>
        ))}
      </div>
      <pre className="py-3 pl-4 pr-6 text-foreground/90">
        <code>
          {lines.map((line, index) => (
            <React.Fragment key={index}>
              <YamlLine line={line} />
              {index < lines.length - 1 ? '\n' : null}
            </React.Fragment>
          ))}
        </code>
      </pre>
    </div>
  )
}

function useRelativeTime(timestamp: number | null): string {
  const [now, setNow] = React.useState(() => Date.now())
  React.useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 5000)
    return () => window.clearInterval(id)
  }, [])
  if (!timestamp) return ''
  const seconds = Math.max(0, Math.round((now - timestamp) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.round(seconds / 60)}m ago`
}

interface EditorOutputPaneProps {
  yaml: string
  crd: string
  kind: OutputKind
  onKindChange: (kind: OutputKind) => void
  compiling: boolean
  stale: boolean
  compiledAt: number | null
  canCompile: boolean
  onCompile: () => void
}

/** The compiled router YAML (or Kubernetes CRD) of the working copy, read-only. */
export function EditorOutputPane({
  yaml,
  crd,
  kind,
  onKindChange,
  compiling,
  stale,
  compiledAt,
  canCompile,
  onCompile,
}: EditorOutputPaneProps) {
  const text = kind === 'yaml' ? yaml : crd
  const compiledLabel = useRelativeTime(compiledAt)
  const download = () => {
    const blob = new Blob([text], { type: 'text/yaml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = Object.assign(document.createElement('a'), {
      href: url,
      download: kind === 'yaml' ? 'config.yaml' : 'semantic-router-crd.yaml',
    })
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-border bg-surface-subtle/50 pl-3 pr-2">
        <Segmented
          size="sm"
          aria-label="Output format"
          value={kind}
          onValueChange={onKindChange}
          className="shrink-0 whitespace-nowrap"
          options={[
            { value: 'yaml', label: 'Router YAML' },
            { value: 'crd', label: 'CRD', disabled: !crd && !compiling },
          ]}
        />
        <span className="ml-1 min-w-0 flex-1 truncate text-xs text-muted-foreground">
          {compiling
            ? 'Compiling the working copy…'
            : stale
              ? 'The working copy changed since this output was compiled.'
              : compiledLabel
                ? `Compiled ${compiledLabel} · read-only`
                : 'Read-only output of the compiler'}
        </span>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {text ? (
            <CopyButton value={text} label={`Copy ${kind === 'yaml' ? 'YAML' : 'CRD'}`} />
          ) : null}
          {text ? (
            <Tooltip content="Download">
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Download output"
                onClick={download}
              >
                <Download />
              </Button>
            </Tooltip>
          ) : null}
          <Button
            variant={stale ? 'soft' : 'ghost'}
            size="xs"
            onClick={onCompile}
            disabled={!canCompile}
            loading={compiling}
          >
            {compiling ? null : stale ? <Play /> : <RefreshCw />}
            Compile
          </Button>
        </div>
      </div>
      <div className="relative min-h-0 flex-1 overflow-auto bg-surface">
        {text ? (
          <CodeLines code={text} className={cn(stale && 'opacity-70 transition-opacity')} />
        ) : compiling ? (
          <div className="flex flex-col gap-2.5 px-12 py-4" aria-hidden>
            {[30, 46, 38, 60, 52, 34, 48, 40].map((width, index) => (
              <Skeleton key={index} className="h-3" style={{ width: `${width}%` }} />
            ))}
          </div>
        ) : (
          <EmptyState
            compact
            icon={FileCode2}
            tone="violet"
            title={kind === 'yaml' ? 'No compiled YAML yet' : 'No CRD output'}
            description={
              kind === 'yaml'
                ? 'Compile the working copy to see the router config it produces.'
                : 'The compiler emits a CRD only for configurations that define one.'
            }
            actions={
              kind === 'yaml' ? (
                <Button size="sm" onClick={onCompile} disabled={!canCompile}>
                  <Play />
                  Compile now
                </Button>
              ) : null
            }
          />
        )}
      </div>
    </div>
  )
}
