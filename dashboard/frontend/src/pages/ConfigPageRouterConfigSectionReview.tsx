import * as React from 'react'
import { motion } from 'framer-motion'
import { RotateCw, ShieldCheck } from 'lucide-react'
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
  cn,
} from '../ui'
import {
  ROUTER_CONFIG_DOCUMENT_SCHEMA,
  configYamlDiff,
  type ConfigYamlHunk,
} from '../lib/configYaml'
import { ROUTER_CONTRACT_VERSION } from './configPageRouterSettingsModel'
import type { JsonRecord } from './configPageRouterSettingsDraft'

export interface RestartNotice {
  path: string
  reason: string
}

export function YamlDiffView({ hunks }: { hunks: ConfigYamlHunk[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-surface-sunken/50 font-mono text-[12px] leading-[1.75] dark:bg-surface-sunken">
      {hunks.map((hunk, hunkIndex) => (
        <div key={`${hunk.header}-${hunkIndex}`} className="border-b border-border last:border-0">
          <div className="flex items-center gap-2 truncate bg-surface-subtle px-3 py-1 text-[11px] text-subtle-foreground">
            <span>@@ {hunk.header || 'global'} @@</span>
            {hunk.skippedBefore > 0 ? (
              <span className="ml-auto font-sans">{hunk.skippedBefore} unchanged lines above</span>
            ) : null}
          </div>
          <div className="overflow-x-auto">
            {hunk.lines.map((line, index) => (
              <div
                key={index}
                className={cn(
                  'flex min-w-fit',
                  line.kind === 'add' && 'bg-success-soft',
                  line.kind === 'remove' && 'bg-destructive-soft',
                  line.kind === 'same' && 'text-subtle-foreground',
                )}
              >
                <span className="w-10 shrink-0 select-none pr-2 text-right text-[11px] text-subtle-foreground/70 tabular">
                  {line.kind === 'remove' ? line.oldNumber : line.newNumber}
                </span>
                <span
                  className={cn(
                    'w-5 shrink-0 select-none text-center',
                    line.kind === 'add' && 'text-success-foreground',
                    line.kind === 'remove' && 'text-destructive-foreground',
                  )}
                >
                  {line.kind === 'add' ? '+' : line.kind === 'remove' ? '−' : ''}
                </span>
                <span className="whitespace-pre pr-4">{line.text}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export function RouterSettingsReviewDialog({
  open,
  onOpenChange,
  saved,
  draft,
  changeCount,
  restart,
  saving,
  error,
  onSave,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  saved: JsonRecord
  draft: JsonRecord
  changeCount: number
  restart: RestartNotice[]
  saving: boolean
  error: string | null
  onSave: () => void
}) {
  const hunks = React.useMemo(
    () =>
      open
        ? configYamlDiff(
            { global: saved },
            { global: draft },
            { schema: ROUTER_CONFIG_DOCUMENT_SCHEMA },
          )
        : [],
    [draft, open, saved],
  )
  return (
    <Dialog open={open} onOpenChange={(next) => !saving && onOpenChange(next)}>
      <DialogContent size="xl">
        <DialogHeader>
          <DialogTitle>Review changes</DialogTitle>
          <DialogDescription>
            {changeCount} {changeCount === 1 ? 'change' : 'changes'} to the global section of
            config.yaml. Only these keys are written; everything else keeps its current value.
          </DialogDescription>
        </DialogHeader>
        <DialogBody className="flex max-h-[min(64vh,600px)] flex-col gap-3 pb-5">
          {error ? (
            <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}>
              <Alert tone="danger" title="The router did not accept these changes">
                <span className="whitespace-pre-wrap break-words">{error}</span>
              </Alert>
            </motion.div>
          ) : null}
          {restart.length ? (
            <Alert tone="warning" title="Restart required" icon={<RotateCw />}>
              <span className="font-mono text-[12px] text-foreground/80">
                {restart.map((item) => item.path).join(', ')}
              </span>{' '}
              {restart.length === 1 ? 'takes' : 'take'} effect after{' '}
              <span className="font-mono text-[12px]">vllm-sr serve</span> restarts the stack.{' '}
              {restart[0].reason} Everything else reloads in place.
            </Alert>
          ) : (
            <Alert tone="success" title="Applies with hot reload">
              The router reloads these settings in place without dropping requests. If it cannot,
              saving tells you what needs a restart.
            </Alert>
          )}
          {hunks.length ? (
            <YamlDiffView hunks={hunks} />
          ) : (
            <p className="text-[13px] text-muted-foreground">
              These edits cancel out; config.yaml stays as it is.
            </p>
          )}
        </DialogBody>
        <DialogFooter>
          <span className="mr-auto flex items-center gap-1.5 text-xs text-muted-foreground">
            <ShieldCheck className="size-3.5 text-success" />
            Checked against the {ROUTER_CONTRACT_VERSION} contract; the router validates again on
            save.
          </span>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={onSave} loading={saving} disabled={hunks.length === 0}>
            {restart.length ? 'Save for restart' : 'Save and reload'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
