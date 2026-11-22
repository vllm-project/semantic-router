import * as React from 'react'
import { Sheet, SheetContent, SheetTitle, TooltipProvider } from '../ui'
import {
  AppSidebar,
  SidebarContents,
  type NavBadges,
  type ShellAccount,
  type ShellInstance,
} from './AppSidebar'
import { CommandMenu, type CommandSection } from './CommandMenu'
import type { NavItem } from './navigation'
import { NAVIGATION_TRIGGER_ID, Topbar, type ShellNotification, type TopbarStatus } from './Topbar'
import { useShell } from './shell-context'

export interface AppShellProps {
  account: ShellAccount
  instance: ShellInstance
  status: TopbarStatus
  notifications?: ShellNotification[]
  badges?: NavBadges
  commandSections?: CommandSection[]
  topbarTrailing?: React.ReactNode
  onSignOut: () => void
  isItemVisible?: (item: NavItem) => boolean
  children: React.ReactNode
}

/**
 * Authenticated application frame: a collapsible sidebar on a tinted canvas and
 * an inset content panel with a sticky top bar. Pages render inside `<main>`.
 */
export function AppShell({
  account,
  instance,
  status,
  notifications,
  badges,
  commandSections,
  topbarTrailing,
  onSignOut,
  isItemVisible,
  children,
}: AppShellProps) {
  const { mobileOpen, setMobileOpen } = useShell()
  return (
    <TooltipProvider delayDuration={250} skipDelayDuration={150}>
      <div className="flex h-dvh overflow-hidden bg-sidebar">
        <AppSidebar
          account={account}
          instance={instance}
          badges={badges}
          onSignOut={onSignOut}
          isItemVisible={isItemVisible}
        />
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetContent
            side="left"
            size="sm"
            className="inset-y-0 left-0 w-[280px] rounded-none rounded-r-xl border-y-0 border-l-0 bg-sidebar p-0"
            onCloseAutoFocus={(event) => {
              event.preventDefault()
              document.getElementById(NAVIGATION_TRIGGER_ID)?.focus()
            }}
          >
            <SheetTitle className="sr-only">Navigation</SheetTitle>
            <SidebarContents
              collapsed={false}
              account={account}
              instance={instance}
              badges={badges}
              onSignOut={onSignOut}
              showCollapse={false}
              isItemVisible={isItemVisible}
            />
          </SheetContent>
        </Sheet>
        <div className="flex min-w-0 flex-1 flex-col lg:py-2 lg:pr-2">
          <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-background lg:rounded-xl lg:border lg:border-sidebar-border lg:shadow-sm">
            <Topbar status={status} notifications={notifications} trailing={topbarTrailing} />
            <main id="main" className="relative flex min-h-0 flex-1 flex-col overflow-y-auto">
              {children}
            </main>
          </div>
        </div>
      </div>
      <CommandMenu sections={commandSections} isItemVisible={isItemVisible} />
    </TooltipProvider>
  )
}
