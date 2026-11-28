import { useState, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import {
  ArrowRight,
  ChevronDown,
  FileCode2,
  Info,
  Loader2,
  RefreshCw,
  Send,
  TriangleAlert,
} from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  CodeView,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  CopyButton,
  CopyField,
  MiniStat,
  ProviderStack,
  StatusDot,
  cn,
} from '../ui'
import type { SetupActivationState, SetupStep, SetupValidationState } from './setupWizardSupport'
import type { SetupRouterReadiness } from './setupRouterReadiness'

interface ReviewActivatePanelProps {
  routeLabel: string
  decisionModelLabel: string
  modelNames: string[]
  providers: string[]
  listenerPort?: number
  validationState: SetupValidationState
  validationError: string | null
  activationState: SetupActivationState
  activationError: string | null
  validatedCounts: {
    models: number
    decisions: number
    signals: number
    canActivate: boolean
  }
  modelsCount: number
  generatedDecisions: number
  generatedSignals: number
  previewYaml: string
  readonlyLoading: boolean
  isReadonly: boolean
  /** Why the setup capability does not allow activation, or null. */
  activationBlockedReason: string | null
  /** A degraded setup capability's caveat, such as running `vllm-sr serve`. */
  activationNotice: string | null
  onValidateAgain: () => void
  onEdit: (step: SetupStep) => void
}

const VALIDATION_META: Record<
  SetupValidationState,
  { label: string; tone: 'success' | 'danger' | 'info' | 'neutral' }
> = {
  validating: { label: 'Validating', tone: 'info' },
  valid: { label: 'Ready', tone: 'success' },
  error: { label: 'Needs fixes', tone: 'danger' },
  idle: { label: 'Pending', tone: 'neutral' },
}

export function ReviewActivatePanel({
  routeLabel,
  decisionModelLabel,
  modelNames,
  providers,
  listenerPort,
  validationState,
  validationError,
  activationState,
  activationError,
  validatedCounts,
  modelsCount,
  generatedDecisions,
  generatedSignals,
  previewYaml,
  readonlyLoading,
  isReadonly,
  activationBlockedReason,
  activationNotice,
  onValidateAgain,
  onEdit,
}: ReviewActivatePanelProps) {
  const [open, setOpen] = useState(false)
  const status = VALIDATION_META[validationState]
  const rows: Array<{ label: string; value: ReactNode; step?: SetupStep }> = [
    {
      label: 'Backend models',
      step: 0,
      value: (
        <span className="flex min-w-0 items-center gap-2.5">
          {providers.length ? <ProviderStack providers={providers} size={16} /> : null}
          <span className="truncate font-mono text-[12.5px]">
            {modelNames.join(', ') || 'From the imported config'}
          </span>
        </span>
      ),
    },
    { label: 'Routing', step: 1, value: <span className="font-medium">{routeLabel}</span> },
    {
      label: 'Decision model',
      step: 2,
      value: <span className="font-mono text-[12.5px]">{decisionModelLabel}</span>,
    },
    {
      label: 'Listener',
      value: (
        <span className="font-mono text-[12.5px]">
          {listenerPort ? `:${listenerPort}` : 'Bootstrap'}
        </span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-hidden rounded-xl border border-border bg-card">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex min-h-12 items-center gap-4 border-b border-border px-4 py-2.5 last:border-0"
          >
            <span className="w-28 shrink-0 text-xs text-subtle-foreground">{row.label}</span>
            <div className="min-w-0 flex-1 text-[13px]">{row.value}</div>
            {row.step !== undefined ? (
              <Button variant="ghost" size="xs" onClick={() => onEdit(row.step as SetupStep)}>
                Edit
              </Button>
            ) : null}
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-3 rounded-xl border border-border bg-surface-subtle/50 p-4">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-semibold">Validation</span>
          <Badge
            size="sm"
            tone={status.tone === 'neutral' ? 'neutral' : status.tone}
            dot
            role="status"
            aria-live="polite"
          >
            {status.label}
          </Badge>
          <Button
            variant="ghost"
            size="xs"
            className="ml-auto"
            onClick={onValidateAgain}
            disabled={validationState === 'validating'}
            aria-busy={validationState === 'validating'}
          >
            <RefreshCw className={cn(validationState === 'validating' && 'animate-spin')} />
            {validationState === 'validating' ? 'Validating…' : 'Revalidate'}
          </Button>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <MiniStat label="Models" value={validatedCounts.models || modelsCount} />
          <MiniStat label="Decisions" value={validatedCounts.decisions || generatedDecisions} />
          <MiniStat label="Signals" value={validatedCounts.signals || generatedSignals} />
        </div>
      </div>

      {validationError ? (
        <div role="alert">
          <Alert tone="danger" title="Validation failed">
            {validationError}
          </Alert>
        </div>
      ) : null}
      {activationError ? (
        <div role="alert">
          <Alert tone="danger" title="Activation failed">
            {activationError}
          </Alert>
        </div>
      ) : null}
      {!readonlyLoading && isReadonly ? (
        <Alert tone="neutral" title="Read-only dashboard">
          This dashboard is running in read-only mode. Review is available, but activation is
          disabled.
        </Alert>
      ) : null}
      {activationBlockedReason && !(!readonlyLoading && isReadonly) ? (
        <div role="status">
          <Alert tone="warning" title="Activation is not available">
            {activationBlockedReason}
          </Alert>
        </div>
      ) : null}
      {activationNotice && !activationBlockedReason ? (
        <div role="status">
          <Alert tone="info">{activationNotice}</Alert>
        </div>
      ) : null}
      {activationState === 'persisted' ? (
        <div role="status">
          <Alert tone="info" title="Saved; rollout required">
            Setup was saved to the Kubernetes ConfigMap. Roll out Router and Envoy to activate it.
          </Alert>
        </div>
      ) : null}

      <Collapsible
        open={open}
        onOpenChange={setOpen}
        className="overflow-hidden rounded-xl border border-border bg-surface-sunken/60 dark:bg-surface-sunken"
      >
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="flex h-10 w-full items-center gap-2 px-3.5 text-left text-[13px] font-medium outline-none hover:bg-accent/40 focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            <FileCode2 className="size-4 text-subtle-foreground" />
            <span className="font-mono text-[12.5px]">config.yaml</span>
            <span className="text-xs font-normal text-subtle-foreground">
              {previewYaml ? `${previewYaml.split('\n').length} lines · ` : ''}generated preview ·
              secrets masked
            </span>
            <ChevronDown
              className={cn(
                'ml-auto size-4 text-subtle-foreground transition-transform duration-200',
                open && 'rotate-180',
              )}
            />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="relative border-t border-border">
            <CopyButton
              value={previewYaml}
              label="Copy YAML"
              className="absolute right-2 top-2 z-10 bg-surface-sunken/80"
            />
            <CodeView
              code={previewYaml || '# Nothing to preview yet'}
              language="yaml"
              lineNumbers
              className="max-h-[300px] overflow-y-auto"
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
      <p className="flex items-start gap-2 text-xs leading-relaxed text-muted-foreground">
        <Info className="mt-px size-3.5 shrink-0 text-subtle-foreground" />
        Activating writes config.yaml, applies it to the router and leaves setup mode. Everything
        stays editable in Settings.
      </p>
    </div>
  )
}

const ACTIVATION_TASKS = [
  'Validated configuration',
  'Writing config.yaml',
  'Applying to the router',
  'Leaving setup mode',
]

export function SetupActivationView() {
  return (
    <div
      className="flex h-full flex-col justify-center px-6 py-10 sm:px-10"
      role="status"
      aria-live="polite"
    >
      <div className="mx-auto w-full max-w-md">
        <h2 className="text-lg font-semibold tracking-[-0.02em]">Activating your router</h2>
        <p className="mt-1 text-[13px] text-muted-foreground">
          Activating the validated router configuration. Keep this tab open.
        </p>
        <ol className="mt-6 flex flex-col rounded-xl border border-border bg-surface-subtle/50 px-4 py-2">
          {ACTIVATION_TASKS.map((task, index) => (
            <li key={task} className="flex items-center gap-3 py-2">
              <span className="grid size-5 shrink-0 place-items-center">
                {index === 0 ? (
                  <StatusDot tone="success" />
                ) : index === 1 ? (
                  <Loader2 className="size-4 animate-spin text-primary" />
                ) : (
                  <span className="size-1.5 rounded-full bg-border-strong" />
                )}
              </span>
              <span
                className={cn('text-[13px] font-medium', index > 1 && 'text-subtle-foreground')}
              >
                {task}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

export function SetupSuccessView({
  baseUrl,
  defaultModel,
  summary,
  leaving,
  readiness,
  activationMessage,
  onTest,
  onDashboard,
}: {
  baseUrl: string
  defaultModel: string
  summary: string
  leaving: boolean
  readiness: SetupRouterReadiness
  activationMessage?: string | null
  onTest: () => void
  onDashboard: () => void
}) {
  if (readiness.state !== 'ready') {
    const stopped = readiness.state === 'stopped'
    return (
      <div
        className="flex h-full flex-col items-center justify-center px-6 py-10 text-center sm:px-10"
        role="status"
        aria-live="polite"
      >
        <span
          className={cn(
            'grid size-[60px] place-items-center rounded-full',
            stopped ? 'bg-destructive-soft text-destructive' : 'bg-primary-soft text-primary',
          )}
        >
          {stopped ? (
            <TriangleAlert className="size-7" aria-hidden />
          ) : (
            <Loader2 className="size-7 animate-spin" aria-hidden />
          )}
        </span>
        <div className="mt-8 flex w-full max-w-xl flex-col items-center">
          <h2 className="text-xl font-semibold tracking-[-0.02em]">
            {stopped ? 'Your router is not running' : 'Starting your router'}
          </h2>
          <p className="mt-1.5 max-w-md text-[13px] leading-relaxed text-muted-foreground">
            {stopped
              ? 'Setup was saved, but the Router did not start. Check the `vllm-sr serve` output for the start-up error, fix it, and run `vllm-sr serve` again. This page updates when the Router is up.'
              : 'Setup was saved. Apps can call the router once it is ready; this page updates on its own.'}
          </p>
          {readiness.detail || activationMessage ? (
            <div className="mt-5 w-full text-left">
              <Alert
                tone={stopped ? 'danger' : 'neutral'}
                title={stopped ? 'Router status' : 'Start-up progress'}
              >
                {readiness.detail || activationMessage}
              </Alert>
            </div>
          ) : null}
          <p className="mt-4 text-xs text-muted-foreground">{summary}</p>
          <div className="mt-8 flex flex-wrap justify-center gap-2">
            <Button size="lg" variant="outline" onClick={onDashboard} loading={leaving}>
              Go to dashboard
              {leaving ? null : <ArrowRight />}
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col items-center justify-center px-6 py-10 text-center sm:px-10">
      <div className="relative grid place-items-center">
        {[0, 1, 2].map((ring) => (
          <motion.span
            key={ring}
            className="absolute rounded-full border border-success/25"
            style={{ width: 76 + ring * 34, height: 76 + ring * 34 }}
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1 - ring * 0.3, scale: 1 }}
            transition={{ delay: 0.15 + ring * 0.07, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          />
        ))}
        <motion.span
          initial={{ scale: 0.6, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', bounce: 0.45, duration: 0.6 }}
          className="grid size-[60px] place-items-center rounded-full bg-success text-white shadow-lg"
        >
          <svg
            viewBox="0 0 24 24"
            className="size-7"
            fill="none"
            stroke="currentColor"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <motion.path
              d="M5 12.5l4.5 4.5L19 7.5"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ delay: 0.25, duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
            />
          </svg>
        </motion.span>
      </div>
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25, duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        className="mt-10 flex w-full flex-col items-center"
      >
        <h2 className="text-xl font-semibold tracking-[-0.02em]">Your router is live</h2>
        <p className="mt-1.5 max-w-md text-[13px] leading-relaxed text-muted-foreground">
          Apps can call it now. Requests show up on Home as soon as they arrive.
        </p>
        <div className="mt-6 grid w-full max-w-xl grid-cols-1 gap-3 text-left sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-subtle-foreground">Base URL</span>
            <CopyField value={baseUrl} />
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-subtle-foreground">Default model</span>
            <CopyField value={defaultModel} />
          </div>
        </div>
        <p className="mt-4 text-xs text-muted-foreground">{summary}</p>
        <div className="mt-8 flex flex-wrap justify-center gap-2">
          <Button variant="outline" size="lg" onClick={onTest} disabled={leaving}>
            <Send />
            Send a test request
          </Button>
          <Button size="lg" onClick={onDashboard} loading={leaving}>
            Go to dashboard
            {leaving ? null : <ArrowRight />}
          </Button>
        </div>
      </motion.div>
    </div>
  )
}
