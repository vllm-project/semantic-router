import React from 'react'
import clsx from 'clsx'
import Link from '@docusaurus/Link'
import styles from './Chrome.module.css'

interface SectionLabelProps {
  children: React.ReactNode
  className?: string
}

interface PillLinkProps {
  children: React.ReactNode
  className?: string
  href?: string
  muted?: boolean
  rel?: string
  target?: string
  to?: string
}

interface PillButtonProps {
  children: React.ReactNode
  className?: string
  muted?: boolean
  onClick?: React.MouseEventHandler<HTMLButtonElement>
  type?: 'button' | 'reset' | 'submit'
}

interface PageIntroProps {
  actions?: React.ReactNode
  className?: string
  description?: React.ReactNode
  label?: React.ReactNode
  title: React.ReactNode
}

interface StatItem {
  description: React.ReactNode
  label: React.ReactNode
  value: React.ReactNode
}

interface StatStripProps {
  className?: string
  items: StatItem[]
}

export function SectionLabel({ children, className }: SectionLabelProps): JSX.Element {
  return <span className={clsx(styles.label, className)}>{children}</span>
}

export function PillLink({
  children,
  className,
  href,
  muted = false,
  rel,
  target,
  to,
}: PillLinkProps): JSX.Element {
  return (
    <Link
      className={clsx(styles.pill, muted && styles.pillMuted, className)}
      href={href}
      rel={rel}
      target={target}
      to={to}
    >
      {children}
    </Link>
  )
}

export function PillButton({
  children,
  className,
  muted = false,
  onClick,
  type = 'button',
}: PillButtonProps): JSX.Element {
  return (
    <button
      className={clsx(styles.pill, styles.pillButton, muted && styles.pillMuted, className)}
      onClick={onClick}
      type={type}
    >
      {children}
    </button>
  )
}

export function PageIntro({
  actions,
  className,
  description,
  label,
  title,
}: PageIntroProps): JSX.Element {
  return (
    <div className={clsx(styles.pageIntro, className)}>
      {label && <SectionLabel>{label}</SectionLabel>}
      <h1 className={styles.pageTitle}>{title}</h1>
      {description && <p className={styles.pageDescription}>{description}</p>}
      {actions && <div className={styles.pageActions}>{actions}</div>}
    </div>
  )
}

export function StatStrip({ className, items }: StatStripProps): JSX.Element {
  return (
    <div className={clsx(styles.statStrip, className)}>
      {items.map((item, index) => (
        <div key={index} className={styles.statItem}>
          <SectionLabel>{item.label}</SectionLabel>
          <span className={styles.statValue}>{item.value}</span>
          <p className={styles.statDescription}>{item.description}</p>
        </div>
      ))}
    </div>
  )
}
