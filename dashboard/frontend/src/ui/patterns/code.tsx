import * as React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, Copy } from 'lucide-react'
import { cn } from '../lib/cn'
import { Tooltip } from '../components/overlays'

export function useCopy(timeout = 1600) {
  const [copied, setCopied] = React.useState(false)
  const copy = React.useCallback(
    (text: string) => {
      void navigator.clipboard?.writeText(text).catch(() => undefined)
      setCopied(true)
      window.setTimeout(() => setCopied(false), timeout)
    },
    [timeout],
  )
  return { copied, copy }
}

export function CopyButton({ value, className, label = 'Copy' }: { value: string; className?: string; label?: string }) {
  const { copied, copy } = useCopy()
  return (
    <Tooltip content={copied ? 'Copied' : label}>
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation()
          copy(value)
        }}
        className={cn(
          'grid size-7 shrink-0 place-items-center rounded-md text-subtle-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none',
          className,
        )}
        aria-label={label}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={copied ? 'check' : 'copy'}
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.6, opacity: 0 }}
            transition={{ duration: 0.14 }}
            className="grid place-items-center"
          >
            {copied ? <Check className="size-3.5 text-success" strokeWidth={2.5} /> : <Copy className="size-3.5" />}
          </motion.span>
        </AnimatePresence>
      </button>
    </Tooltip>
  )
}

export function CopyField({ value, mono = true, className }: { value: string; mono?: boolean; className?: string }) {
  return (
    <div className={cn('flex h-9 items-center gap-2 rounded-md border border-border bg-surface-subtle pl-3 pr-1', className)}>
      <span className={cn('min-w-0 flex-1 truncate text-[13px]', mono && 'font-mono text-[12.5px]')}>{value}</span>
      <CopyButton value={value} />
    </div>
  )
}

type Lang = 'bash' | 'python' | 'ts' | 'json' | 'yaml' | 'text' | 'dsl'

const KEYWORDS: Record<Lang, string[]> = {
  bash: ['curl', 'export', 'pip', 'npm', 'vllm-sr'],
  python: ['from', 'import', 'as', 'def', 'return', 'for', 'in', 'print', 'with', 'None', 'True', 'False'],
  ts: ['import', 'from', 'const', 'await', 'async', 'new', 'return', 'export', 'function'],
  json: ['true', 'false', 'null'],
  yaml: ['true', 'false', 'null'],
  dsl: ['ROUTE', 'SIGNAL', 'MODEL', 'PLUGIN', 'WHEN', 'AND', 'OR', 'NOT', 'PRIORITY', 'USE', 'RECIPE'],
  text: [],
}

function highlightLine(line: string, lang: Lang): React.ReactNode[] {
  if (lang === 'text') return [line]
  const commentMatch = lang === 'yaml' || lang === 'bash' || lang === 'python' || lang === 'dsl' ? line.match(/(^|\s)(#.*)$/) : line.match(/(^|\s)(\/\/.*)$/)
  let body = line
  let comment = ''
  if (commentMatch && commentMatch.index !== undefined) {
    body = line.slice(0, commentMatch.index + commentMatch[1].length)
    comment = commentMatch[2]
  }
  const tokens: React.ReactNode[] = []
  const regex = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\s-{1,2}[a-zA-Z][\w-]*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][\w./-]*)(?=\s*:(?:\s|$))|([A-Za-z_][\w-]*)/g
  let last = 0
  let match: RegExpExecArray | null
  const keywords = new Set(KEYWORDS[lang])
  while ((match = regex.exec(body))) {
    if (match.index > last) tokens.push(body.slice(last, match.index))
    const [text, str, flag, num, key, word] = match
    if (str) tokens.push(<span key={tokens.length} className="text-[oklch(0.55_0.13_150)] dark:text-[oklch(0.8_0.13_150)]">{text}</span>)
    else if (flag) tokens.push(<span key={tokens.length} className="text-[oklch(0.6_0.14_60)] dark:text-[oklch(0.82_0.13_75)]">{text}</span>)
    else if (num) tokens.push(<span key={tokens.length} className="text-[oklch(0.58_0.16_40)] dark:text-[oklch(0.78_0.13_45)]">{text}</span>)
    else if (key && (lang === 'yaml' || lang === 'json' || lang === 'dsl')) tokens.push(<span key={tokens.length} className="text-chart-1">{text}</span>)
    else if (word && keywords.has(word)) tokens.push(<span key={tokens.length} className="text-chart-3">{text}</span>)
    else tokens.push(text)
    last = match.index + text.length
  }
  if (last < body.length) tokens.push(body.slice(last))
  if (comment) tokens.push(<span key="c" className="italic text-subtle-foreground">{comment}</span>)
  return tokens
}

export function CodeView({ code, language = 'text', lineNumbers = false, className }: { code: string; language?: Lang; lineNumbers?: boolean; className?: string }) {
  const lines = code.replace(/\n$/, '').split('\n')
  return (
    <pre className={cn('overflow-x-auto py-3 font-mono text-[12.5px] leading-[1.7] text-foreground/90', className)}>
      <code className="grid">
        {lines.map((line, index) => (
          <span key={index} className="flex px-4">
            {lineNumbers ? <span className="mr-4 w-6 shrink-0 select-none text-right text-subtle-foreground/70">{index + 1}</span> : null}
            <span className="whitespace-pre">{highlightLine(line, language)}{line === '' ? ' ' : null}</span>
          </span>
        ))}
      </code>
    </pre>
  )
}

export function CodeBlock({
  tabs,
  title,
  className,
  lineNumbers,
}: {
  tabs: { label: string; language: Lang; code: string }[]
  title?: React.ReactNode
  className?: string
  lineNumbers?: boolean
}) {
  const [active, setActive] = React.useState(0)
  const current = tabs[active] ?? tabs[0]
  return (
    <div className={cn('overflow-hidden rounded-xl border border-border bg-surface-sunken/60 dark:bg-surface-sunken', className)}>
      <div className="flex h-10 items-center justify-between gap-2 border-b border-border bg-surface-subtle/70 pl-2 pr-1.5">
        <div className="flex items-center gap-0.5">
          {title ? <span className="px-2 text-xs font-medium text-muted-foreground">{title}</span> : null}
          {tabs.length > 1
            ? tabs.map((tab, index) => (
                <button
                  key={tab.label}
                  type="button"
                  onClick={() => setActive(index)}
                  className={cn(
                    'relative h-7 rounded-md px-2.5 text-xs font-medium transition-colors',
                    index === active ? 'text-foreground' : 'text-subtle-foreground hover:text-foreground',
                  )}
                >
                  {index === active ? (
                    <motion.span layoutId={`code-tab-${tabs.map((t) => t.label).join('')}`} className="absolute inset-0 rounded-md bg-surface shadow-sm ring-1 ring-border dark:bg-accent" transition={{ type: 'spring', bounce: 0.18, duration: 0.4 }} />
                  ) : null}
                  <span className="relative">{tab.label}</span>
                </button>
              ))
            : !title ? <span className="px-2 text-xs font-medium text-muted-foreground">{current.label}</span> : null}
        </div>
        <CopyButton value={current.code} />
      </div>
      <CodeView code={current.code} language={current.language} lineNumbers={lineNumbers} />
    </div>
  )
}

export function InlineCode({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <code className={cn('rounded-[5px] border border-border bg-surface-subtle px-1.5 py-0.5 font-mono text-[12px] text-foreground/90', className)}>
      {children}
    </code>
  )
}
