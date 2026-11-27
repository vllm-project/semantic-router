import * as React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { DiffEditor, type Monaco, type MonacoDiffEditor } from '@monaco-editor/react'
import {
  ArrowDown,
  ArrowUp,
  CircleCheck,
  CircleDashed,
  CircleX,
  FileDiff,
  Info,
  Loader2,
  OctagonX,
  Rocket,
  TriangleAlert,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import {
  Alert,
  Badge,
  Button,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  IconTile,
  Progress,
  SectionLabel,
  Segmented,
  Skeleton,
  Tooltip,
  cn,
} from '../../ui'
import ConfigSaveWarnings from '../../components/ConfigSaveWarnings'
import type { DeployResult, DeployStep } from '@/types/dsl'
import { changedRegion } from './editorDiff'
import {
  MonacoBoundary,
  defineEditorTheme,
  editorBaseOptions,
  useEditorMonacoTheme,
  useMonacoLoader,
} from './editorMonaco'

const STEPS: Array<{ step: DeployStep; label: string; detail: string }> = [
  { step: 'validating', label: 'Validate', detail: 'the merged config, as the Router loads it' },
  { step: 'backing_up', label: 'Back up', detail: 'the running config' },
  { step: 'writing', label: 'Write the new version', detail: 'config store' },
  { step: 'reloading', label: 'Reload the Router', detail: 'waits until it reports healthy' },
]

function stepIndex(step: DeployStep | null): number {
  if (step === 'done') return STEPS.length
  const index = STEPS.findIndex((item) => item.step === step)
  return index < 0 ? 0 : index
}

interface DiffStats {
  hunks: number
  added: number
  removed: number
}

interface EditorDeployDialogProps {
  open: boolean
  previewLoading: boolean
  previewError: string | null
  currentYaml: string
  mergedYaml: string
  errors: number
  warnings: number
  /** The working copy came from the running Router rather than an imported file. */
  baseFromRouter: boolean
  deploying: boolean
  deployStep: DeployStep | null
  deployResult: DeployResult | null
  onDeploy: () => void
  onRetry: () => void
  onClose: () => void
  onShowProblems: () => void
}

function CheckRow({
  state,
  title,
  children,
  action,
}: {
  state: 'ok' | 'warn' | 'fail' | 'info'
  title: string
  children: React.ReactNode
  action?: React.ReactNode
}) {
  const Icon =
    state === 'ok'
      ? CircleCheck
      : state === 'warn'
        ? TriangleAlert
        : state === 'fail'
          ? OctagonX
          : Info
  return (
    <div className="flex gap-2.5 py-2.5">
      <Icon
        className={cn(
          'mt-0.5 size-4 shrink-0',
          state === 'ok' && 'text-success',
          state === 'warn' && 'text-warning',
          state === 'fail' && 'text-destructive',
          state === 'info' && 'text-info',
        )}
      />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-medium">{title}</span>
          {action}
        </div>
        <div className="text-xs leading-relaxed text-muted-foreground">{children}</div>
      </div>
    </div>
  )
}

function Outcome({ result }: { result: DeployResult }) {
  if (result.status === 'error') {
    return (
      <Alert tone="danger" title="Deploy failed">
        <span className="break-words">{result.message}</span> The running config was not changed.
      </Alert>
    )
  }
  const version = result.version ? `v${result.version}` : 'The new version'
  switch (result.outcome) {
    case 'restart-required':
      return (
        <Alert tone="warning" title={`${version} saved · restart required`}>
          {result.message}
        </Alert>
      )
    case 'persisted':
      return (
        <Alert tone="info" title={`${version} saved to the config store`}>
          {result.message}
        </Alert>
      )
    case 'reload-unknown':
      return (
        <Alert
          tone="warning"
          title={`${version} deployed · reload not confirmed`}
          actions={
            <Button asChild variant="outline" size="xs">
              <Link to="/observe/logs">Open logs</Link>
            </Button>
          }
        >
          The config was written, but the Router did not confirm it. Check its logs before you route
          traffic to the change.
        </Alert>
      )
    default:
      return (
        <Alert tone="success" title={`${version} is live · hot reload`}>
          {result.message} In-flight requests finished on the previous version.
        </Alert>
      )
  }
}

// The editor package disposes the diff's models before the diff editor, which
// Monaco reports as an error; a layout cleanup runs first and detaches them.
function DetachDiffModels({ editor }: { editor: React.MutableRefObject<MonacoDiffEditor | null> }) {
  React.useLayoutEffect(
    () => () => {
      editor.current?.setModel(null)
      editor.current = null
    },
    [editor],
  )
  return null
}

/** Review the merged config as a diff, deploy it, and follow the reload or restart outcome. */
export function EditorDeployDialog({
  open,
  previewLoading,
  previewError,
  currentYaml,
  mergedYaml,
  errors,
  warnings,
  baseFromRouter,
  deploying,
  deployStep,
  deployResult,
  onDeploy,
  onRetry,
  onClose,
  onShowProblems,
}: EditorDeployDialogProps) {
  const [monaco, setMonaco] = React.useState<Monaco | null>(null)
  const [split, setSplit] = React.useState<'unified' | 'split'>('unified')
  const [stats, setStats] = React.useState<DiffStats | null>(null)
  const diffRef = React.useRef<MonacoDiffEditor | null>(null)
  const theme = useEditorMonacoTheme(monaco)
  const loadState = useMonacoLoader(open)
  const plain = React.useMemo(
    () =>
      loadState === 'failed' && !previewLoading && !previewError
        ? changedRegion(currentYaml, mergedYaml)
        : null,
    [currentYaml, loadState, mergedYaml, previewError, previewLoading],
  )
  const phase: 'review' | 'deploying' | 'done' = deploying
    ? 'deploying'
    : deployResult
      ? 'done'
      : 'review'
  const identical = !previewLoading && !previewError && currentYaml === mergedYaml
  const current = stepIndex(deployStep)

  React.useEffect(() => {
    if (!open) setStats(null)
  }, [open])

  const goToChange = (direction: 1 | -1) => {
    const editor = diffRef.current
    const changes = editor?.getLineChanges()
    if (!editor || !changes?.length) return
    const modified = editor.getModifiedEditor()
    const line = modified.getPosition()?.lineNumber ?? 1
    const starts = changes.map(
      (change) => change.modifiedStartLineNumber || change.originalStartLineNumber,
    )
    const target =
      direction === 1
        ? (starts.find((start) => start > line) ?? starts[0])
        : ([...starts].reverse().find((start) => start < line) ?? starts[starts.length - 1])
    modified.revealLineInCenter(target)
    modified.setPosition({ lineNumber: target, column: 1 })
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next && phase !== 'deploying') onClose()
      }}
    >
      <DialogContent
        size="xl"
        hideClose={phase === 'deploying'}
        className="max-w-[min(1080px,calc(100%-2rem))]"
        onOpenAutoFocus={(event) => event.preventDefault()}
      >
        <DialogHeader className="flex-row items-start gap-3.5 pb-4">
          <IconTile icon={Rocket} tone="violet" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <DialogTitle>
              {phase === 'review'
                ? 'Review & deploy'
                : phase === 'deploying'
                  ? 'Deploying'
                  : deployResult?.status === 'error'
                    ? 'Deploy failed'
                    : 'Deployed'}
            </DialogTitle>
            <DialogDescription>
              {baseFromRouter
                ? 'Applies the compiled routing to the config the Router runs now, as a new version.'
                : 'Replaces the running config with the imported file and the compiled routing, as a new version.'}
            </DialogDescription>
          </div>
        </DialogHeader>

        <AnimatePresence mode="wait" initial={false}>
          {phase === 'review' ? (
            <motion.div
              key="review"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
              className="min-h-0"
            >
              <DialogBody className="flex max-h-[calc(100vh-14rem)] flex-col gap-4 pb-5">
                <div className="overflow-hidden rounded-lg border border-border">
                  <div className="flex h-10 items-center gap-2 border-b border-border bg-surface-subtle/70 pl-3 pr-2">
                    <FileDiff className="size-3.5 text-subtle-foreground" />
                    <span className="font-mono text-[12px]">config.yaml</span>
                    {plain && !identical ? (
                      <span className="text-xs text-subtle-foreground">
                        changed region · {plain.lines.length} lines
                      </span>
                    ) : null}
                    {stats && !identical && !plain ? (
                      <span className="font-mono text-[11.5px] tabular">
                        <span className="text-success-foreground">+{stats.added}</span>{' '}
                        <span className="text-destructive-foreground">−{stats.removed}</span>
                        <span className="ml-2 font-sans text-subtle-foreground">
                          {stats.hunks} {stats.hunks === 1 ? 'change' : 'changes'}
                        </span>
                      </span>
                    ) : null}
                    <div className={cn('ml-auto flex items-center gap-1', plain && 'hidden')}>
                      <Tooltip content="Previous change">
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-label="Go to previous config change"
                          disabled={!stats?.hunks}
                          onClick={() => goToChange(-1)}
                        >
                          <ArrowUp />
                        </Button>
                      </Tooltip>
                      <Tooltip content="Next change">
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-label="Go to next config change"
                          disabled={!stats?.hunks}
                          onClick={() => goToChange(1)}
                        >
                          <ArrowDown />
                        </Button>
                      </Tooltip>
                      <Segmented
                        size="sm"
                        aria-label="Diff layout"
                        value={split}
                        onValueChange={setSplit}
                        options={[
                          { value: 'unified', label: 'Unified' },
                          { value: 'split', label: 'Split' },
                        ]}
                      />
                    </div>
                  </div>
                  <div className="relative h-[clamp(220px,38vh,420px)] bg-surface">
                    {previewLoading ? (
                      <div
                        className="flex flex-col gap-2.5 px-6 py-4"
                        aria-label="Loading the diff"
                      >
                        {[42, 64, 36, 58, 70, 44, 52].map((width, index) => (
                          <Skeleton key={index} className="h-3" style={{ width: `${width}%` }} />
                        ))}
                      </div>
                    ) : previewError ? (
                      <div className="p-4" role="alert">
                        <Alert tone="danger" title="The diff could not be prepared">
                          {previewError} You can still deploy; the Router validates the merged
                          config first.
                        </Alert>
                      </div>
                    ) : identical ? (
                      <div className="grid h-full place-items-center px-6 text-center">
                        <div className="flex max-w-sm flex-col items-center gap-2">
                          <CircleCheck className="size-5 text-success" />
                          <p className="text-[13px] font-medium">
                            No changes to the running config
                          </p>
                          <p className="text-xs text-muted-foreground">
                            The compiled routing already matches what the Router runs. Deploying
                            writes the same config as a new version.
                          </p>
                        </div>
                      </div>
                    ) : plain ? (
                      <div className="h-full overflow-auto py-2 font-mono text-[12px] leading-5">
                        <p className="px-4 pb-2 font-sans text-xs text-muted-foreground">
                          The diff editor could not load; this is the changed region as plain text.
                        </p>
                        {plain.lines.slice(0, 600).map((line, index) => (
                          <div
                            key={index}
                            className={cn(
                              'flex whitespace-pre px-4',
                              line.kind === 'add' && 'bg-success-soft text-foreground',
                              line.kind === 'remove' && 'bg-destructive-soft text-foreground',
                            )}
                          >
                            <span className="w-10 shrink-0 select-none pr-3 text-right text-subtle-foreground">
                              {line.number}
                            </span>
                            <span className="w-4 shrink-0 select-none text-subtle-foreground">
                              {line.kind === 'add' ? '+' : line.kind === 'remove' ? '−' : ' '}
                            </span>
                            {line.text}
                          </div>
                        ))}
                        {plain.lines.length > 600 ? (
                          <p className="px-4 pt-2 font-sans text-xs text-subtle-foreground">
                            {plain.lines.length - 600} more lines not shown.
                          </p>
                        ) : null}
                      </div>
                    ) : loadState === 'loading' ? (
                      <div
                        className="flex flex-col gap-2.5 px-6 py-4"
                        aria-label="Loading the diff"
                      >
                        {[42, 64, 36, 58].map((width, index) => (
                          <Skeleton key={index} className="h-3" style={{ width: `${width}%` }} />
                        ))}
                      </div>
                    ) : (
                      <MonacoBoundary>
                        <DetachDiffModels editor={diffRef} />
                        <DiffEditor
                          original={currentYaml}
                          modified={mergedYaml}
                          language="yaml"
                          theme={theme}
                          loading={null}
                          beforeMount={(api) =>
                            defineEditorTheme(
                              api,
                              document.documentElement.classList.contains('dark')
                                ? 'dark'
                                : 'light',
                            )
                          }
                          onMount={(editor, api) => {
                            diffRef.current = editor
                            setMonaco(api)
                            const update = () => {
                              const changes = editor.getLineChanges() ?? []
                              let added = 0
                              let removed = 0
                              for (const change of changes) {
                                if (change.modifiedEndLineNumber >= change.modifiedStartLineNumber)
                                  added +=
                                    change.modifiedEndLineNumber -
                                    change.modifiedStartLineNumber +
                                    1
                                if (change.originalEndLineNumber >= change.originalStartLineNumber)
                                  removed +=
                                    change.originalEndLineNumber -
                                    change.originalStartLineNumber +
                                    1
                              }
                              setStats({ hunks: changes.length, added, removed })
                            }
                            editor.onDidUpdateDiff(update)
                          }}
                          options={{
                            ...editorBaseOptions(),
                            readOnly: true,
                            originalEditable: false,
                            renderSideBySide: split === 'split',
                            useInlineViewWhenSpaceIsLimited: false,
                            hideUnchangedRegions: { enabled: true, contextLineCount: 3 },
                            renderOverviewRuler: false,
                            renderIndicators: true,
                            stickyScroll: { enabled: false },
                            contextmenu: false,
                          }}
                        />
                      </MonacoBoundary>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
                  <section className="flex flex-col">
                    <SectionLabel>Checks</SectionLabel>
                    <div className="mt-1 divide-y divide-border">
                      {errors > 0 ? (
                        <CheckRow
                          state="fail"
                          title="Compiler"
                          action={
                            <Badge tone="danger" size="sm">
                              {errors}
                            </Badge>
                          }
                        >
                          Fix the errors before deploying.{' '}
                          <button
                            type="button"
                            className="font-medium text-primary hover:underline"
                            onClick={onShowProblems}
                          >
                            Show problems
                          </button>
                        </CheckRow>
                      ) : (
                        <CheckRow
                          state={warnings ? 'warn' : 'ok'}
                          title="Compiler"
                          action={
                            warnings ? (
                              <Badge tone="warning" size="sm">
                                {warnings}
                              </Badge>
                            ) : null
                          }
                        >
                          {warnings ? (
                            <>
                              Compiles without errors, with {warnings}{' '}
                              {warnings === 1 ? 'warning' : 'warnings'} worth a look.{' '}
                              <button
                                type="button"
                                className="font-medium text-primary hover:underline"
                                onClick={onShowProblems}
                              >
                                Show problems
                              </button>
                            </>
                          ) : (
                            'Compiles without errors or warnings; every reference resolves.'
                          )}
                        </CheckRow>
                      )}
                      <CheckRow state="info" title="Base">
                        {baseFromRouter
                          ? 'Settings outside the editor (providers, listeners, router settings) stay as the Router has them.'
                          : 'The imported file replaces the running config, including settings outside the editor.'}
                      </CheckRow>
                      <CheckRow state="ok" title="Backup">
                        The running config is backed up before the new version is written, so you
                        can roll back.
                      </CheckRow>
                    </div>
                  </section>
                  <section className="flex flex-col gap-3">
                    <SectionLabel>Rollout</SectionLabel>
                    <Alert tone="info" title="Hot reload">
                      The Router validates and reloads the new version without dropping in-flight
                      requests. Changes that need a restart, such as listeners, are reported after
                      the write.
                    </Alert>
                  </section>
                </div>
              </DialogBody>
            </motion.div>
          ) : (
            <motion.div
              key="progress"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            >
              <DialogBody className="flex flex-col gap-4 pb-6">
                <Progress
                  value={(Math.min(current, STEPS.length) / STEPS.length) * 100}
                  tone={
                    deployResult?.status === 'error'
                      ? 'danger'
                      : phase === 'done'
                        ? 'success'
                        : 'brand'
                  }
                />
                <ol className="flex flex-col" aria-label="Deploy progress">
                  {STEPS.map((item, index) => {
                    const failed = deployResult?.status === 'error' && index === current
                    const state =
                      phase === 'done' && deployResult?.status !== 'error'
                        ? 'done'
                        : failed
                          ? 'failed'
                          : index < current
                            ? 'done'
                            : index === current && phase === 'deploying'
                              ? 'running'
                              : 'pending'
                    return (
                      <li
                        key={item.step}
                        className="flex items-center gap-3 border-b border-border py-2.5 last:border-0"
                      >
                        <span className="grid size-5 shrink-0 place-items-center">
                          {state === 'done' ? (
                            <CircleCheck className="size-[18px] text-success" />
                          ) : state === 'running' ? (
                            <Loader2 className="size-4 animate-spin text-primary" />
                          ) : state === 'failed' ? (
                            <CircleX className="size-[18px] text-destructive" />
                          ) : (
                            <CircleDashed className="size-4 text-subtle-foreground" />
                          )}
                        </span>
                        <span
                          className={cn(
                            'text-[13px]',
                            state === 'pending' ? 'text-muted-foreground' : 'font-medium',
                          )}
                        >
                          {item.label}
                        </span>
                        <span className="truncate text-xs text-subtle-foreground">
                          {item.detail}
                        </span>
                      </li>
                    )
                  })}
                </ol>
                {deployResult ? (
                  <div className="flex flex-col gap-0">
                    <Outcome result={deployResult} />
                    {deployResult.warnings?.length ? (
                      <ConfigSaveWarnings warnings={deployResult.warnings} />
                    ) : null}
                  </div>
                ) : null}
              </DialogBody>
            </motion.div>
          )}
        </AnimatePresence>

        <DialogFooter className="justify-between">
          <span className="min-w-0 truncate text-xs text-muted-foreground">
            {phase === 'deploying'
              ? 'Keep this window open until the Router confirms.'
              : phase === 'done'
                ? deployResult?.status === 'error'
                  ? 'Nothing changed. Review the error and try again.'
                  : 'The editor now tracks the deployed version.'
                : errors > 0
                  ? 'Deploy is blocked until the compiler errors are fixed.'
                  : 'A backup of the running config is created first.'}
          </span>
          <div className="flex shrink-0 items-center gap-2">
            {phase === 'done' ? (
              <>
                {deployResult?.status === 'error' ? (
                  <Button variant="outline" onClick={onRetry}>
                    Review again
                  </Button>
                ) : null}
                <Button onClick={onClose}>Done</Button>
              </>
            ) : (
              <>
                <Button variant="ghost" onClick={onClose} disabled={phase === 'deploying'}>
                  Cancel
                </Button>
                <Button
                  onClick={onDeploy}
                  disabled={phase === 'review' && (errors > 0 || previewLoading)}
                  loading={phase === 'deploying'}
                >
                  {phase === 'deploying' ? null : <Rocket />}
                  {phase === 'deploying' ? 'Deploying…' : 'Deploy'}
                </Button>
              </>
            )}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
