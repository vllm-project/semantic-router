import * as React from 'react'
import { cn } from '../lib/cn'

export function PageContainer({
  width = 'default',
  className,
  children,
}: {
  width?: 'default' | 'narrow' | 'wide' | 'full'
  className?: string
  children: React.ReactNode
}) {
  return (
    <div
      className={cn(
        // A full-height page only fades: rising, it would overflow its frame for a moment and flash a scrollbar.
        width === 'full' ? 'animate-page-fade' : 'animate-fade-up',
        width === 'full'
          ? 'flex min-h-0 flex-1 flex-col'
          : 'mx-auto w-full px-5 pb-16 pt-6 md:px-8',
        width === 'default' && 'max-w-[1360px]',
        width === 'wide' && 'max-w-[1680px]',
        width === 'narrow' && 'max-w-[880px]',
        className,
      )}
    >
      {children}
    </div>
  )
}

export function PageHeader({
  title,
  description,
  icon,
  eyebrow,
  actions,
  meta,
  children,
  className,
}: {
  title: React.ReactNode
  description?: React.ReactNode
  icon?: React.ReactNode
  eyebrow?: React.ReactNode
  actions?: React.ReactNode
  meta?: React.ReactNode
  children?: React.ReactNode
  className?: string
}) {
  return (
    <header className={cn('mb-6 flex flex-col gap-4', className)}>
      <div className="flex flex-col gap-x-8 gap-y-3 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 flex-1 items-start gap-3.5">
          {icon}
          <div className="flex min-w-0 flex-col gap-1">
            {eyebrow ? (
              <div className="text-xs font-medium text-subtle-foreground">{eyebrow}</div>
            ) : null}
            <h1 className="text-[22px] font-semibold leading-7 tracking-[-0.022em] text-foreground">
              {title}
            </h1>
            {description ? (
              <p className="max-w-2xl text-[13.5px] leading-relaxed text-muted-foreground">
                {description}
              </p>
            ) : null}
            {meta ? (
              <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
                {meta}
              </div>
            ) : null}
          </div>
        </div>
        {actions ? (
          <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>
        ) : null}
      </div>
      {children}
    </header>
  )
}

export function MetaItem({
  icon,
  label,
  children,
}: {
  icon?: React.ReactNode
  label?: string
  children: React.ReactNode
}) {
  return (
    <span className="inline-flex items-center gap-1.5 [&_svg]:size-3.5 [&_svg]:text-subtle-foreground">
      {icon}
      {label ? <span className="text-subtle-foreground">{label}</span> : null}
      <span className="font-medium text-foreground/80">{children}</span>
    </span>
  )
}

export function Section({
  title,
  description,
  actions,
  children,
  className,
}: {
  title?: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section className={cn('flex flex-col gap-3', className)}>
      {title || actions ? (
        <div className="flex items-end justify-between gap-4">
          <div className="flex flex-col gap-0.5">
            {title ? (
              <h2 className="text-[15px] font-semibold tracking-[-0.012em]">{title}</h2>
            ) : null}
            {description ? (
              <p className="text-[13px] text-muted-foreground">{description}</p>
            ) : null}
          </div>
          {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  )
}
