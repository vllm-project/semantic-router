import { Link } from 'react-router-dom'
import { ArrowRight, Check } from 'lucide-react'

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Progress,
  Skeleton,
  cn,
} from '../ui'
import type { StartStep } from './startSupport'

function StepMark({ step }: { step: StartStep }) {
  if (step.state === 'pending') return <Skeleton className="size-5 shrink-0 rounded-full" />
  return (
    <span
      className={cn(
        'grid size-5 shrink-0 place-items-center rounded-full border',
        step.state === 'done'
          ? 'border-transparent bg-success text-white'
          : 'border-dashed border-border-strong text-subtle-foreground',
      )}
    >
      {step.state === 'done' ? <Check className="size-3" strokeWidth={3} /> : null}
    </span>
  )
}

/** The first-run steps with their live state; each links to where the step is done. */
export function StartChecklistCard({
  steps,
  title = 'Getting started',
  className,
}: {
  steps: StartStep[]
  title?: string
  className?: string
}) {
  const done = steps.filter((step) => step.state === 'done').length
  const percent = steps.length ? Math.round((done / steps.length) * 100) : 0
  const next = steps.find((step) => step.state === 'todo')
  return (
    <Card className={className} data-testid="start-checklist">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>
          {done === steps.length ? 'Everything is in place' : `${done} of ${steps.length} done`}
        </CardDescription>
        <CardAction>
          <span className="text-[13px] font-semibold tabular text-primary">{percent}%</span>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-0.5 pt-3">
        <Progress
          value={percent}
          className="mb-2.5"
          aria-label={`${done} of ${steps.length} done`}
        />
        {steps.map((step) => (
          <Link
            key={step.id}
            to={step.to}
            data-step={step.id}
            data-state={step.state}
            className="group -mx-2 flex items-center gap-3 rounded-lg px-2 py-2 outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            <StepMark step={step} />
            <span className="flex min-w-0 flex-1 flex-col">
              <span
                className={cn(
                  'text-[13px] font-medium',
                  step.state === 'done' && 'font-normal text-muted-foreground',
                )}
              >
                {step.label}
              </span>
              <span className="truncate text-xs text-subtle-foreground">{step.detail}</span>
            </span>
            {next?.id === step.id ? (
              <span className="rounded-full bg-primary-soft px-1.5 text-[11px] font-medium leading-5 text-primary-soft-foreground">
                Next
              </span>
            ) : null}
            <ArrowRight className="size-3.5 shrink-0 text-subtle-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100" />
          </Link>
        ))}
      </CardContent>
    </Card>
  )
}
