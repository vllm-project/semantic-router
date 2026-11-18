import React from 'react'
import { LoaderCircle, RotateCw, TriangleAlert } from 'lucide-react'
import AuthBackdrop, { AuthCard } from '../components/AuthBackdrop'
import { Button, cn } from '../ui'

export interface SetupStatusPageProps {
  title: string
  description: string
  actionLabel: string
  onAction: () => void
  variant?: 'loading' | 'error'
}

/** Full-screen status card shown while the session or setup state loads, or when it cannot be read. */
const SetupStatusPage: React.FC<SetupStatusPageProps> = ({
  title,
  description,
  actionLabel,
  onAction,
  variant = 'error',
}) => (
  <AuthBackdrop>
    <AuthCard>
      <div role="status" aria-live="polite" className="flex flex-col items-center text-center">
        <span
          className={cn(
            'mb-5 grid size-12 place-items-center rounded-2xl border shadow-sm',
            variant === 'loading'
              ? 'border-border bg-surface text-primary'
              : 'border-warning/30 bg-warning-soft text-warning-foreground',
          )}
          aria-hidden="true"
        >
          {variant === 'loading' ? (
            <LoaderCircle className="size-5 animate-spin" />
          ) : (
            <TriangleAlert className="size-5" />
          )}
        </span>
        <h1 className="text-lg font-semibold tracking-[-0.015em]">{title}</h1>
        <p className="mt-2 text-[13px] leading-relaxed text-muted-foreground [overflow-wrap:anywhere]">
          {description}
        </p>
        {variant === 'error' ? (
          <Button className="mt-6 w-full" onClick={onAction}>
            <RotateCw />
            {actionLabel}
          </Button>
        ) : null}
      </div>
    </AuthCard>
  </AuthBackdrop>
)

export default SetupStatusPage
