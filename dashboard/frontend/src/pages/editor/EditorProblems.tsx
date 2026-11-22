import * as React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Info, OctagonX, ShieldCheck, TriangleAlert, WandSparkles } from 'lucide-react'
import { Button, EmptyState, Segmented, Tooltip, cn } from '../../ui'
import type { DiagLevel, Diagnostic, QuickFix } from '@/types/dsl'
import { countDiagnostics, type EditorEntity } from './editorModel'

const severityIcon = { error: OctagonX, warning: TriangleAlert, constraint: Info }
const severityText: Record<DiagLevel, string> = {
  error: 'text-destructive',
  warning: 'text-warning',
  constraint: 'text-info',
}

type Filter = 'all' | DiagLevel

interface EditorProblemsProps {
  diagnostics: Diagnostic[]
  /** A whole-source error from the compiler or the config load, shown first. */
  sourceError?: string | null
  ownerOf: (diagnostic: Diagnostic) => EditorEntity | null
  onSelect: (diagnostic: Diagnostic) => void
  onFix: (diagnostic: Diagnostic, fix: QuickFix) => void
  footer?: React.ReactNode
}

/** The compiler's diagnostics for the working copy, with jump-to-source and quick fixes. */
export function EditorProblems({
  diagnostics,
  sourceError,
  ownerOf,
  onSelect,
  onFix,
  footer,
}: EditorProblemsProps) {
  const [filter, setFilter] = React.useState<Filter>('all')
  const errors = countDiagnostics(diagnostics, 'error') + (sourceError ? 1 : 0)
  const warnings = countDiagnostics(diagnostics, 'warning')
  const hints = countDiagnostics(diagnostics, 'constraint')
  const visible = diagnostics.filter(
    (diagnostic) => filter === 'all' || diagnostic.level === filter,
  )
  const showSourceError = Boolean(sourceError) && (filter === 'all' || filter === 'error')

  return (
    <section aria-label="Problems" className="flex min-h-0 flex-1 flex-col">
      <div className="flex h-10 shrink-0 items-center justify-between gap-2 pl-4 pr-2.5">
        <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
          Problems
        </span>
        <Segmented
          size="sm"
          aria-label="Filter problems"
          value={filter}
          onValueChange={setFilter}
          options={[
            {
              value: 'all',
              label: (
                <span className="whitespace-nowrap tabular">All {errors + warnings + hints}</span>
              ),
            },
            {
              value: 'error',
              label: <span className="tabular">{errors}</span>,
              icon: <OctagonX className="text-destructive" aria-label="Errors" />,
            },
            {
              value: 'warning',
              label: <span className="tabular">{warnings}</span>,
              icon: <TriangleAlert className="text-warning" aria-label="Warnings" />,
            },
            {
              value: 'constraint',
              label: <span className="tabular">{hints}</span>,
              icon: <Info className="text-info" aria-label="Hints" />,
            },
          ]}
        />
      </div>
      <div className="flex shrink-0 items-center gap-3 border-t border-border px-4 py-1.5 text-[11.5px] text-muted-foreground">
        <span className={cn('tabular', errors > 0 && 'font-medium text-destructive-foreground')}>
          {errors} {errors === 1 ? 'error' : 'errors'}
        </span>
        <span className="tabular">
          {warnings} {warnings === 1 ? 'warning' : 'warnings'}
        </span>
        <span className="tabular">
          {hints} {hints === 1 ? 'hint' : 'hints'}
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto border-t border-border">
        {errors + warnings + hints === 0 ? (
          <EmptyState
            compact
            icon={ShieldCheck}
            tone="green"
            title="No problems"
            description="The compiler accepts the working copy: references resolve and no decision shadows another."
            className="px-6"
          />
        ) : visible.length === 0 && !showSourceError ? (
          <p className="px-4 py-8 text-center text-xs text-subtle-foreground">
            Nothing at this severity.
          </p>
        ) : (
          <ul className="divide-y divide-border">
            {showSourceError ? (
              <li className="flex gap-2.5 px-4 py-3">
                <OctagonX className="mt-0.5 size-4 shrink-0 text-destructive" />
                <p className="min-w-0 break-words font-mono text-[11.5px] leading-relaxed text-foreground">
                  {sourceError}
                </p>
              </li>
            ) : null}
            <AnimatePresence initial={false}>
              {visible.map((diagnostic) => {
                const Icon = severityIcon[diagnostic.level]
                const owner = ownerOf(diagnostic)
                return (
                  <motion.li
                    key={`${diagnostic.level}:${diagnostic.line}:${diagnostic.column}:${diagnostic.message}`}
                    layout="position"
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: 12 }}
                    transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                    className="group relative"
                  >
                    <button
                      type="button"
                      onClick={() => onSelect(diagnostic)}
                      className="flex w-full gap-2.5 px-4 pb-1.5 pt-3 text-left outline-none transition-colors hover:bg-surface-subtle focus-visible:bg-surface-subtle"
                    >
                      <Icon
                        className={cn('mt-0.5 size-4 shrink-0', severityText[diagnostic.level])}
                      />
                      <span className="flex min-w-0 flex-1 flex-col gap-1">
                        <span className="text-[12.5px] leading-snug text-foreground">
                          {diagnostic.message}
                        </span>
                        <span className="flex items-center gap-2 font-mono text-[11px] tabular text-subtle-foreground">
                          Ln {diagnostic.line}, Col {diagnostic.column}
                          {owner ? (
                            <span className="truncate font-sans">· {owner.name}</span>
                          ) : null}
                        </span>
                      </span>
                    </button>
                    {diagnostic.fixes?.length ? (
                      <div className="flex flex-wrap gap-1.5 pb-3 pl-[42px] pr-4">
                        {diagnostic.fixes.map((fix) => (
                          <Tooltip key={fix.description} content={`Replace with ${fix.newText}`}>
                            <Button
                              size="xs"
                              variant="soft"
                              className="h-6 px-2 text-[11.5px]"
                              onClick={() => onFix(diagnostic, fix)}
                            >
                              <WandSparkles />
                              {fix.description || 'Apply fix'}
                            </Button>
                          </Tooltip>
                        ))}
                      </div>
                    ) : (
                      <div className="pb-1.5" />
                    )}
                  </motion.li>
                )
              })}
            </AnimatePresence>
          </ul>
        )}
      </div>
      {footer ? (
        <div className="shrink-0 border-t border-border px-4 py-2.5 text-[11.5px] text-subtle-foreground">
          {footer}
        </div>
      ) : null}
    </section>
  )
}
