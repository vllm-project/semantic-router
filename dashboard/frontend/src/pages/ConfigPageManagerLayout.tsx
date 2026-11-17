import React from 'react'
import { PageContainer, PageHeader } from '../ui'
import { SettingsSection, useInSettingsLayout } from './SettingsLayout'

interface ConfigPageManagerLayoutProps {
  /** Kept for existing callers; the shell breadcrumb now shows the section. */
  eyebrow?: string
  title: string
  description: string
  icon?: React.ReactNode
  actions?: React.ReactNode
  meta?: React.ReactNode
  tabs?: React.ReactNode
  width?: 'default' | 'narrow' | 'wide' | 'full'
  children: React.ReactNode
}

/**
 * Frame for configuration sections: one page header, then the content. Inside
 * the settings frame it renders a section header instead.
 */
export default function ConfigPageManagerLayout({ title, description, icon, actions, meta, tabs, width, children }: ConfigPageManagerLayoutProps) {
  const inSettings = useInSettingsLayout()
  if (inSettings) {
    return (
      <SettingsSection title={title} description={description} actions={actions}>
        {meta ? <div className="-mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">{meta}</div> : null}
        {tabs}
        {children}
      </SettingsSection>
    )
  }
  return (
    <PageContainer width={width}>
      <PageHeader title={title} description={description} icon={icon} actions={actions} meta={meta}>
        {tabs}
      </PageHeader>
      {children}
    </PageContainer>
  )
}
