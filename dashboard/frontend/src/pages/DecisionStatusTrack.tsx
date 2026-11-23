import { Fragment } from 'react'
import { motion } from 'framer-motion'
import { Check } from 'lucide-react'
import { Tooltip, cn } from '../ui'
import type { DecisionDeploymentProgress, DecisionDeploymentStage } from './decisionModelManagement'

const steps: { key: DecisionDeploymentStage; label: string }[] = [
  { key: 'saved', label: 'Saved configuration' },
  { key: 'activated', label: 'Activated' },
  { key: 'ready', label: 'Observed ready' },
]
const order: Record<DecisionDeploymentStage, number> = { saved: 0, activated: 1, ready: 2 }

export default function DecisionStatusTrack({
  progress,
  size = 'lg',
  className,
}: {
  progress: DecisionDeploymentProgress
  size?: 'lg' | 'sm'
  className?: string
}) {
  const reached = order[progress.stage]
  const pending = progress.pending
  if (size === 'sm') {
    return (
      <span className={cn('inline-flex items-center', className)} aria-hidden="true">
        {steps.map((step, index) => {
          const done = index <= reached
          const next = index === reached + 1
          return (
            <Fragment key={step.key}>
              {index > 0 ? (
                <span
                  className={cn('h-px w-3.5', index <= reached ? 'bg-success' : 'bg-border-strong')}
                />
              ) : null}
              <Tooltip
                content={`${step.label} · ${progress.details[step.key] ?? (next && pending ? pending : 'Not yet')}`}
              >
                <span
                  className={cn(
                    'grid size-4 place-items-center rounded-full border',
                    done
                      ? 'border-transparent bg-success text-white'
                      : next && pending
                        ? 'border-primary bg-primary-soft'
                        : 'border-dashed border-border-strong',
                  )}
                >
                  {done ? (
                    <Check className="size-2.5" strokeWidth={3.5} />
                  ) : next && pending ? (
                    <span className="size-1.5 animate-pulse rounded-full bg-primary" />
                  ) : null}
                </span>
              </Tooltip>
            </Fragment>
          )
        })}
      </span>
    )
  }
  return (
    <ol className={cn('grid grid-cols-1 gap-4 sm:grid-cols-3 sm:gap-0', className)}>
      {steps.map((step, index) => {
        const done = index <= reached
        const next = index === reached + 1
        const detail = progress.details[step.key] ?? (next && pending ? pending : 'Not started')
        return (
          <li key={step.key} className="relative flex flex-col gap-2 sm:pr-4">
            <div className="flex items-center">
              <motion.span
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.12 * index, type: 'spring', bounce: 0.4, duration: 0.45 }}
                className={cn(
                  'z-10 grid size-6 shrink-0 place-items-center rounded-full border',
                  done
                    ? 'border-transparent bg-success text-white shadow-xs'
                    : next && pending
                      ? 'border-primary bg-primary-soft'
                      : 'border-dashed border-border-strong bg-card',
                )}
              >
                {done ? (
                  <Check className="size-3.5" strokeWidth={3} />
                ) : next && pending ? (
                  <span className="size-2 animate-pulse rounded-full bg-primary" />
                ) : (
                  <span className="text-[10px] font-semibold text-subtle-foreground">
                    {index + 1}
                  </span>
                )}
              </motion.span>
              {index < steps.length - 1 ? (
                <span className="relative mx-2 hidden h-0.5 flex-1 overflow-hidden rounded-full bg-border sm:block">
                  <motion.span
                    className={cn(
                      'absolute inset-y-0 left-0 rounded-full',
                      index < reached ? 'bg-success' : 'bg-primary/40',
                    )}
                    initial={{ width: 0 }}
                    animate={{
                      width: index < reached ? '100%' : index === reached && pending ? '45%' : '0%',
                    }}
                    transition={{
                      delay: 0.15 + 0.12 * index,
                      duration: 0.6,
                      ease: [0.16, 1, 0.3, 1],
                    }}
                  />
                </span>
              ) : null}
            </div>
            <div className="flex flex-col gap-0.5">
              <span className={cn('text-[13px] font-medium', !done && 'text-muted-foreground')}>
                {step.label}
              </span>
              <span
                className={cn(
                  'text-xs leading-snug',
                  done
                    ? 'text-muted-foreground'
                    : next && pending
                      ? 'text-primary'
                      : 'text-subtle-foreground',
                )}
              >
                {detail}
              </span>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
