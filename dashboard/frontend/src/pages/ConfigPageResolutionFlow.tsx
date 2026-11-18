import { Fragment, type ReactNode } from 'react'
import { motion } from 'framer-motion'

import { cn } from '../ui'

export interface ResolutionStep {
  key: string
  icon: ReactNode
  label: string
  value: ReactNode
  hint?: ReactNode
  /** Relative width; long identifiers such as model names get more room. */
  grow?: number
  onClick?: () => void
  /** Accessible name when the step navigates somewhere. */
  actionLabel?: string
}

function Connector({ index }: { index: number }) {
  return (
    <span className="flex w-5 shrink-0 items-center justify-center" aria-hidden>
      <svg viewBox="0 0 24 10" className="h-2.5 w-5 overflow-visible">
        <line
          x1="1"
          y1="5"
          x2="19"
          y2="5"
          stroke="var(--border-strong)"
          strokeWidth="1.5"
          strokeDasharray="3 3"
          className="animate-flow-dash"
          style={{ animationDelay: `${index * -0.3}s` }}
        />
        <path
          d="M18 1.5 L22 5 L18 8.5"
          fill="none"
          stroke="var(--border-strong)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  )
}

/** A left-to-right chain of routing stages: model name → recipe → signals → decisions → backends. */
export default function ConfigPageResolutionFlow({
  steps,
  className,
}: {
  steps: ResolutionStep[]
  className?: string
}) {
  return (
    <div className={cn('flex items-stretch', className)}>
      {steps.map((step, index) => (
        <Fragment key={step.key}>
          {index > 0 ? <Connector index={index} /> : null}
          <motion.button
            type="button"
            onClick={step.onClick}
            disabled={!step.onClick}
            aria-label={step.actionLabel}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.06, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            style={{ flexGrow: step.grow ?? 1, flexBasis: 0 }}
            className="group flex min-w-0 flex-col gap-2 rounded-lg border border-border bg-surface px-2.5 py-2.5 text-left shadow-xs outline-none transition-colors enabled:hover:border-border-strong enabled:hover:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-default"
          >
            {step.icon}
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="text-[11px] leading-4 text-subtle-foreground">{step.label}</span>
              <span className="truncate text-[12.5px] font-medium leading-4">{step.value}</span>
              {step.hint ? (
                <span className="truncate text-[11px] leading-4 text-muted-foreground">
                  {step.hint}
                </span>
              ) : null}
            </span>
          </motion.button>
        </Fragment>
      ))}
    </div>
  )
}
