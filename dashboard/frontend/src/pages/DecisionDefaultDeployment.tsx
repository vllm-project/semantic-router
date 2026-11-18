import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  ChevronDown,
  Info,
  Link2,
  MessagesSquare,
  MoreHorizontal,
  PanelRightOpen,
  ScrollText,
} from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  InlineCode,
  MiniStat,
  ProviderLogo,
  SectionLabel,
  Skeleton,
  Tooltip,
  cn,
} from '../ui'
import type { Observation } from '../utils/dashboardObservation'
import type { DecisionDeploymentRow } from './decisionDeploymentRows'
import {
  decisionActivationLabel,
  type DecisionModelActivation,
  decisionStageStatus,
} from './decisionModelManagement'
import { questionTypeLabel } from './decisionQuestionTypes'
import DecisionObservationStatus from './DecisionObservationStatus'
import DecisionStatusTrack from './DecisionStatusTrack'

type ObservationLike = Pick<
  Observation<unknown>,
  'loading' | 'refreshing' | 'stale' | 'error' | 'updatedAt'
>

export interface DecisionDefaultDiagnostics {
  savedModel: string
  defaultDeployment: string
  ownership?: string
  messages: string[]
  activation: DecisionModelActivation | null
}

function Chip({ children, mono = true }: { children: ReactNode; mono?: boolean }) {
  return (
    <span className="inline-flex h-6 max-w-full items-center gap-1.5 rounded-md border border-border bg-card px-2 text-xs">
      <span className={cn('truncate', mono && 'font-mono text-[11.5px]')}>{children}</span>
    </span>
  )
}

export default function DecisionDefaultDeployment({
  row,
  loading,
  mode,
  phase,
  servingNote,
  busy,
  operationError,
  observations,
  diagnostics,
  writable,
  onBind,
  onDetails,
}: {
  row?: DecisionDeploymentRow
  loading: boolean
  mode: string
  phase: string
  servingNote?: string
  busy: boolean
  operationError?: string
  observations: Array<{ label: string; observation: ObservationLike }>
  diagnostics: DecisionDefaultDiagnostics
  writable: boolean
  onBind: () => void
  onDetails: () => void
}) {
  const navigate = useNavigate()
  if (!row)
    return (
      <Card className="gap-4 p-5" aria-busy="true">
        <div className="flex items-center gap-3">
          <Skeleton className="size-9 rounded-lg" />
          <div className="flex flex-col gap-2">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-3 w-64" />
          </div>
        </div>
        <Skeleton className="h-16 w-full" />
      </Card>
    )
  const status = decisionStageStatus(row.progress)
  const questions = row.consumers.filter((consumer) => consumer.kind === 'question')
  const selectors = row.consumers.filter((consumer) => consumer.kind === 'selector')
  const playground = `/models/decision?tab=playground&target=${encodeURIComponent(`deployment:${row.name}`)}`
  return (
    <section aria-label="Default decision model">
      <Card className="overflow-hidden">
        <CardHeader className="pb-4">
          <div className="flex min-w-0 items-center gap-3">
            <ProviderLogo provider="vllm" size={22} framed />
            <div className="flex min-w-0 flex-col gap-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="font-mono text-[15px] font-semibold leading-none tracking-[-0.01em]">
                  {row.modelName}
                </h2>
                <Badge tone="brand" size="sm">
                  Default
                </Badge>
                <Badge tone={status.tone} size="sm" dot>
                  {status.label}
                </Badge>
              </div>
              <CardDescription className="text-xs">
                Deployment <span className="font-mono">{row.name}</span> ·{' '}
                {row.kind === 'systemone' ? 'System One API' : 'model runtime'}
                {row.device ? ` · ${row.device}` : ''}
              </CardDescription>
            </div>
          </div>
          <CardAction>
            <Button variant="outline" size="sm" onClick={() => navigate(playground)}>
              <MessagesSquare />
              Try it
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon-sm" aria-label="Default deployment actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                <DropdownMenuItem onSelect={onDetails}>
                  <PanelRightOpen />
                  Deployment details
                </DropdownMenuItem>
                <DropdownMenuItem disabled={!writable} onSelect={onBind}>
                  <Link2 />
                  Bind consumers
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onSelect={() =>
                    navigate(
                      `/models/decision?tab=monitoring&deployment=${encodeURIComponent(row.name)}`,
                    )
                  }
                >
                  <Activity />
                  Monitoring
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => navigate('/observe/logs')}>
                  <ScrollText />
                  View logs
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </CardAction>
        </CardHeader>
        <CardContent className="flex flex-col gap-5 border-t border-border bg-surface-subtle/40 pt-5">
          <DecisionStatusTrack progress={row.progress} />
          {busy ? (
            <Alert tone="info" title="Deployment in progress">
              This page remains available while the runtime changes.
            </Alert>
          ) : null}
          {servingNote ? (
            <Alert tone="neutral" icon={<Info />}>
              {servingNote}
            </Alert>
          ) : null}
          {operationError ? (
            <div role="alert">
              <Alert tone="danger" title="The last deployment failed">
                {operationError}
              </Alert>
            </div>
          ) : null}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <MiniStat
              label="Readiness"
              value={
                row.replicas.ready === null ? (
                  <span className="text-muted-foreground">{loading ? '…' : 'unknown'}</span>
                ) : (
                  `${row.replicas.ready}/${row.replicas.desired}`
                )
              }
              hint="replicas ready"
            />
            <MiniStat
              label="Device"
              value={
                <span className="font-mono text-base">
                  {row.observed?.device || row.device || '—'}
                </span>
              }
              hint={row.observed ? 'observed' : 'as configured'}
            />
            <MiniStat
              label="Engine"
              value={<span className="font-mono text-base">{row.observed?.engine || '—'}</span>}
              hint={row.observed?.profile ? `profile ${row.observed.profile}` : 'not reported'}
            />
            <MiniStat
              label="Restarts"
              value={row.observed?.restarts ?? '—'}
              hint={row.observed?.managed === false ? 'external worker' : 'managed worker'}
            />
            <MiniStat
              label="Serving mode"
              value={<span className="text-base">{mode}</span>}
              hint="instance startup"
            />
            <MiniStat
              label="Deployment phase"
              value={<span className="text-base">{phase}</span>}
              hint="instance operation"
            />
          </div>
          <div className="flex flex-col gap-2">
            <SectionLabel>Serves</SectionLabel>
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-subtle-foreground">Default</span>
                <Chip mono={false}>Every question without its own binding</Chip>
              </div>
              {questions.length ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs text-subtle-foreground">Signals</span>
                  {questions.map((consumer) => (
                    <Chip key={consumer.id}>
                      {consumer.recipe ? `${consumer.recipe} · ` : ''}
                      {consumer.name}
                      <span className="ml-1.5 font-sans text-[11px] text-subtle-foreground">
                        {questionTypeLabel(consumer.questionType)}
                      </span>
                    </Chip>
                  ))}
                </div>
              ) : null}
              {selectors.length ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs text-subtle-foreground">Decisions</span>
                  {selectors.map((consumer) => (
                    <Chip key={consumer.id}>{consumer.name}</Chip>
                  ))}
                </div>
              ) : null}
              {row.taskBindings.length ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs text-subtle-foreground">Tasks</span>
                  {row.taskBindings.slice(0, 8).map((binding) => (
                    <Chip key={`${binding.recipe}:${binding.consumer}:${binding.task_id}`}>
                      {binding.consumer}
                    </Chip>
                  ))}
                  {row.taskBindings.length > 8 ? (
                    <Tooltip
                      content={row.taskBindings
                        .slice(8)
                        .map((binding) => binding.consumer)
                        .join(', ')}
                    >
                      <button
                        type="button"
                        onClick={onDetails}
                        className="h-6 rounded-md px-1.5 text-xs text-muted-foreground outline-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
                      >
                        +{row.taskBindings.length - 8} more
                      </button>
                    </Tooltip>
                  ) : null}
                </div>
              ) : null}
            </div>
            {row.inherited.length ? (
              <p className="text-xs text-muted-foreground">
                {row.inherited.length}{' '}
                {row.inherited.length === 1 ? 'question asks' : 'questions ask'} the default because{' '}
                {row.inherited.length === 1 ? 'it names' : 'they name'} no deployment.
              </p>
            ) : null}
          </div>
          {observations.map((item) => (
            <DecisionObservationStatus
              key={item.label}
              label={item.label}
              observation={item.observation}
            />
          ))}
          <Collapsible>
            <CollapsibleTrigger className="group inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40">
              <ChevronDown className="size-3.5 transition-transform group-data-[state=open]:rotate-180" />
              Deployment diagnostics
            </CollapsibleTrigger>
            <CollapsibleContent>
              <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 rounded-lg border border-border bg-card p-3 text-xs sm:grid-cols-2">
                <div className="flex flex-col gap-0.5">
                  <dt className="text-subtle-foreground">Saved model</dt>
                  <dd>Saved model: {diagnostics.savedModel}</dd>
                </div>
                <div className="flex flex-col gap-0.5">
                  <dt className="text-subtle-foreground">Default deployment</dt>
                  <dd className="font-mono">{diagnostics.defaultDeployment}</dd>
                </div>
                <div className="flex flex-col gap-0.5">
                  <dt className="text-subtle-foreground">Managed by</dt>
                  <dd>{diagnostics.ownership || 'Not reported'}</dd>
                </div>
                <div className="flex flex-col gap-0.5">
                  <dt className="text-subtle-foreground">Activation</dt>
                  <dd>{decisionActivationLabel(diagnostics.activation)}</dd>
                </div>
                <div className="flex min-w-0 flex-col gap-0.5">
                  <dt className="text-subtle-foreground">Generated configuration</dt>
                  <dd className="truncate">
                    <InlineCode>
                      {diagnostics.activation?.generated_runtime_hash || 'Not reported'}
                    </InlineCode>
                  </dd>
                </div>
                <div className="flex min-w-0 flex-col gap-0.5">
                  <dt className="text-subtle-foreground">Active configuration</dt>
                  <dd className="truncate">
                    <InlineCode>
                      {diagnostics.activation?.active_runtime_hash || 'Not reported'}
                    </InlineCode>
                  </dd>
                </div>
                {diagnostics.messages.length ? (
                  <div className="flex flex-col gap-1 sm:col-span-2">
                    <dt className="text-subtle-foreground">Messages</dt>
                    {diagnostics.messages.map((message, index) => (
                      <dd key={index} className="text-muted-foreground">
                        {message}
                      </dd>
                    ))}
                  </div>
                ) : null}
              </dl>
            </CollapsibleContent>
          </Collapsible>
        </CardContent>
      </Card>
    </section>
  )
}
