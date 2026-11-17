import React, { createContext, useContext } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Settings } from 'lucide-react'
import { IconTile, PageContainer, PageHeader, cn } from '../ui'
import { navigation } from '../shell/navigation'
import { useAuth } from '../contexts/AuthContext'
import { useCapability } from '../contexts/CapabilitiesContext'
import { isCapabilityUsable } from '../contexts/dashboardCapabilities'
import { canAccessDashboardPath } from '../utils/accessControl'

const InSettingsContext = createContext(false)

/** True inside the settings frame, where pages render a section header instead of a page header. */
export function useInSettingsLayout(): boolean {
  return useContext(InSettingsContext)
}

const settingsGroup = navigation.find((group) => group.id === 'settings')

/** One "Settings" page with a sub-navigation; each settings route renders its content on the right. */
export default function SettingsLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const { user } = useAuth()
  const mcp = useCapability('mcp')
  const items = (settingsGroup?.items ?? []).filter(
    (item) => canAccessDashboardPath(user, item.to) && !(item.id === 'integrations' && mcp && !isCapabilityUsable(mcp)),
  )
  const current = items.find((item) => location.pathname.startsWith(item.to))

  return (
    <PageContainer>
      <PageHeader
        icon={<IconTile icon={Settings} tone="slate" size="lg" />}
        title="Settings"
        description="Router-wide defaults, integrations, people and the configuration reference."
      />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[208px_minmax(0,1fr)] xl:gap-10">
        <nav aria-label="Settings" className="flex gap-1 overflow-x-auto border-b border-border pb-px xl:flex-col xl:border-b-0 xl:pb-0">
          {items.map((item) => {
            const active = item.id === current?.id
            return (
              <NavLink
                key={item.id}
                to={item.to}
                className={cn(
                  'relative flex h-9 shrink-0 items-center gap-2.5 rounded-md px-2.5 text-[13px] font-medium outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                  active ? 'text-foreground' : 'text-muted-foreground hover:bg-accent/70 hover:text-foreground',
                )}
              >
                {active ? (
                  <motion.span
                    layoutId="settings-nav-active"
                    className="absolute inset-0 rounded-md bg-accent shadow-[inset_0_0_0_1px_var(--border)]"
                    transition={{ type: 'spring', bounce: 0.15, duration: 0.4 }}
                  />
                ) : null}
                <item.icon className={cn('relative size-4', active ? 'text-primary' : 'text-subtle-foreground')} strokeWidth={1.9} />
                <span className="relative whitespace-nowrap">{item.label}</span>
              </NavLink>
            )
          })}
        </nav>
        <div className="min-w-0">
          <InSettingsContext.Provider value>{children}</InSettingsContext.Provider>
        </div>
      </div>
    </PageContainer>
  )
}

/** Header of one settings page inside the settings frame. */
export function SettingsSection({
  title,
  description,
  actions,
  children,
}: {
  title: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
  children?: React.ReactNode
}) {
  return (
    <section className="flex min-w-0 flex-col gap-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-[17px] font-semibold tracking-[-0.015em]">{title}</h2>
          {description ? <p className="max-w-2xl text-[13px] leading-relaxed text-muted-foreground">{description}</p> : null}
        </div>
        {actions ? <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div> : null}
      </div>
      {children}
    </section>
  )
}
