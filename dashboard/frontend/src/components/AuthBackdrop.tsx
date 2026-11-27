import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { BookOpen, OctagonAlert } from 'lucide-react'

import { ThemeToggle } from '../shell/ThemeToggle'
import { BrandLockup, Button, LogoMark, MosaicBackground, cn } from '../ui'

const DOCS_URL = 'https://vllm-sr.ai/docs/intro/'
const GITHUB_URL = 'https://github.com/vllm-project/semantic-router'
const ease = [0.16, 1, 0.3, 1] as const

/** Centered glass card that holds an auth form, with an optional note below it. */
export function AuthCard({
  children,
  note,
  busy,
  className,
}: {
  children: ReactNode
  note?: ReactNode
  busy?: boolean
  className?: string
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, ease }}
      className={cn('w-full max-w-[400px]', className)}
    >
      <div
        aria-busy={busy || undefined}
        className="rounded-2xl border border-border bg-card/90 p-7 shadow-xl backdrop-blur-xl sm:p-8"
      >
        {children}
      </div>
      {note ? (
        <p className="mx-auto mt-4 max-w-[340px] text-center text-xs leading-relaxed text-subtle-foreground">
          {note}
        </p>
      ) : null}
    </motion.div>
  )
}

export function AuthCardHeader({
  title,
  description,
  children,
}: {
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="mb-6 flex flex-col items-center text-center">
      <span className="mb-5 grid size-12 place-items-center rounded-2xl border border-border bg-surface shadow-sm">
        <LogoMark size={26} animated />
      </span>
      {children}
      <h1 className="text-xl font-semibold tracking-[-0.02em] text-foreground">{title}</h1>
      {description ? (
        <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">{description}</p>
      ) : null}
    </div>
  )
}

export function AuthError({ children }: { children: ReactNode }) {
  return (
    <motion.div
      role="alert"
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease }}
      className="flex items-start gap-2 rounded-md bg-destructive-soft px-3 py-2 text-[13px] leading-5 text-destructive-foreground"
    >
      <OctagonAlert className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <span className="min-w-0 [overflow-wrap:anywhere]">{children}</span>
    </motion.div>
  )
}

/** Quiet full-screen frame shared by sign-in and invitation pages. */
export default function AuthBackdrop({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex min-h-dvh flex-col overflow-hidden bg-background">
      <div
        aria-hidden="true"
        data-testid="login-motion-background"
        className="pointer-events-none absolute inset-0"
      >
        <div className="absolute -left-[12%] -top-[22%] size-[60vmax] rounded-full bg-[radial-gradient(closest-side,var(--glow-brand),transparent)] opacity-70 blur-2xl animate-float-slow" />
        <div className="absolute -bottom-[30%] -right-[14%] size-[55vmax] rounded-full bg-[radial-gradient(closest-side,var(--glow-gold),transparent)] opacity-70 blur-2xl animate-float-slow [animation-delay:-9s]" />
        <MosaicBackground />
      </div>
      <header className="relative z-10 flex items-center justify-between gap-4 px-5 py-4 sm:px-6 sm:py-5">
        <BrandLockup height={30} />
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" asChild>
            <a href={DOCS_URL} target="_blank" rel="noreferrer">
              <BookOpen />
              Docs
            </a>
          </Button>
          <ThemeToggle />
        </div>
      </header>
      <main className="relative z-10 flex flex-1 items-center justify-center px-4 py-8">
        {children}
      </main>
      <footer className="relative z-10 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 px-6 pb-6 text-xs text-subtle-foreground">
        <span>vLLM Semantic Router</span>
        <span aria-hidden="true">·</span>
        <a
          className="transition-colors hover:text-foreground"
          href={DOCS_URL}
          target="_blank"
          rel="noreferrer"
        >
          Documentation
        </a>
        <span aria-hidden="true">·</span>
        <a
          className="transition-colors hover:text-foreground"
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
        >
          GitHub
        </a>
      </footer>
    </div>
  )
}
