import * as React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Bell, BookOpen, ChevronRight, CircleHelp, ExternalLink, Keyboard, Menu, Search, Sparkles } from 'lucide-react'
import {
  Badge,
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
  Kbd,
  Popover,
  PopoverContent,
  PopoverTrigger,
  StatusDot,
  Tooltip,
  cn,
  type StatusTone,
} from '../ui'
import { findNavItem } from './navigation'
import { useShell } from './shell-context'
import { ThemeToggle } from './ThemeToggle'

export interface ShellNotification {
  id: string
  title: string
  body: string
  time: string
  tone: StatusTone
  unread?: boolean
}

export interface TopbarStatus {
  tone: StatusTone
  label: string
  detail?: string
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="size-4" aria-hidden="true">
      <path d="M12 .5C5.73.5.75 5.48.75 11.75c0 4.97 3.22 9.18 7.69 10.67.56.1.77-.24.77-.54v-1.9c-3.13.68-3.79-1.34-3.79-1.34-.51-1.3-1.25-1.65-1.25-1.65-1.02-.7.08-.68.08-.68 1.13.08 1.72 1.16 1.72 1.16 1 1.72 2.63 1.22 3.27.93.1-.73.39-1.22.71-1.5-2.5-.28-5.13-1.25-5.13-5.57 0-1.23.44-2.24 1.16-3.03-.12-.28-.5-1.43.11-2.98 0 0 .95-.3 3.1 1.16a10.8 10.8 0 0 1 5.64 0c2.15-1.46 3.1-1.16 3.1-1.16.61 1.55.23 2.7.11 2.98.72.79 1.16 1.8 1.16 3.03 0 4.33-2.64 5.28-5.15 5.56.4.35.76 1.03.76 2.08v3.08c0 .3.2.65.78.54 4.46-1.5 7.68-5.7 7.68-10.67C23.25 5.48 18.27.5 12 .5Z" />
    </svg>
  )
}

export function Topbar({
  status,
  notifications = [],
  trailing,
}: {
  status: TopbarStatus
  notifications?: ShellNotification[]
  trailing?: React.ReactNode
}) {
  const { setCommandOpen, setMobileOpen } = useShell()
  const location = useLocation()
  const current = findNavItem(location.pathname)
  const unread = notifications.filter((notification) => notification.unread).length

  return (
    <header className="sticky top-0 z-30 flex h-[52px] shrink-0 items-center gap-2 border-b border-border bg-background/80 px-3 backdrop-blur-md md:px-4">
      <Button variant="ghost" size="icon-sm" className="lg:hidden" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
        <Menu />
      </Button>
      <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5 text-[13px]">
        {current?.group.label ? (
          <>
            <span className="hidden text-subtle-foreground sm:inline">{current.group.label}</span>
            <ChevronRight className="hidden size-3.5 text-subtle-foreground/60 sm:inline" />
          </>
        ) : null}
        {current ? (
          <Link to={current.item.to} className="flex min-w-0 items-center gap-1.5 font-medium text-foreground">
            <current.item.icon className="size-4 text-subtle-foreground" strokeWidth={1.9} />
            <span className="truncate">{current.item.label}</span>
          </Link>
        ) : null}
      </nav>

      <div className="ml-auto flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => setCommandOpen(true)}
          className="group hidden h-8 w-[280px] items-center gap-2 rounded-lg border border-border bg-surface px-2.5 text-[13px] text-subtle-foreground shadow-xs transition-[border-color,box-shadow,color] hover:border-border-strong hover:text-muted-foreground md:flex xl:w-[340px]"
        >
          <Search className="size-4" />
          <span className="flex-1 truncate text-left">Search or jump to…</span>
          <span className="flex items-center gap-0.5">
            <Kbd>⌘</Kbd>
            <Kbd>K</Kbd>
          </span>
        </button>
        <Button variant="ghost" size="icon-sm" className="md:hidden" onClick={() => setCommandOpen(true)} aria-label="Search">
          <Search />
        </Button>

        <Link
          to="/observe/status"
          className="hidden h-8 items-center gap-2 rounded-lg border border-border bg-surface px-2.5 text-xs font-medium text-foreground/85 shadow-xs transition-colors hover:border-border-strong sm:flex"
        >
          <StatusDot tone={status.tone} pulse={status.tone !== 'neutral'} />
          {status.label}
          {status.detail ? <span className="font-mono text-[11px] text-subtle-foreground">{status.detail}</span> : null}
        </Link>

        {trailing}

        <Popover>
          <Tooltip content="Notifications">
            <PopoverTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label="Notifications" className="relative">
                <Bell />
                {unread > 0 ? (
                  <span className="absolute right-1.5 top-1.5 size-2 rounded-full bg-primary ring-2 ring-background" />
                ) : null}
              </Button>
            </PopoverTrigger>
          </Tooltip>
          <PopoverContent align="end" className="w-[360px] p-0">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <span className="text-[13px] font-semibold">Activity</span>
              {unread > 0 ? <Badge tone="brand" size="sm">{unread} new</Badge> : null}
            </div>
            <div className="max-h-[360px] overflow-y-auto p-1.5">
              {notifications.map((notification) => (
                <div key={notification.id} className={cn('flex gap-3 rounded-lg px-2.5 py-2.5 transition-colors hover:bg-accent', notification.unread && 'bg-primary-soft/40')}>
                  <StatusDot tone={notification.tone} className="mt-1.5" />
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="text-[13px] font-medium">{notification.title}</span>
                    <span className="text-xs leading-relaxed text-muted-foreground">{notification.body}</span>
                  </div>
                  <span className="shrink-0 text-[11px] text-subtle-foreground">{notification.time}</span>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>

        <ThemeToggle />

        <DropdownMenu>
          <Tooltip content="Help">
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label="Help">
                <CircleHelp />
              </Button>
            </DropdownMenuTrigger>
          </Tooltip>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>Help & resources</DropdownMenuLabel>
            <DropdownMenuItem asChild>
              <a href="https://vllm-sr.ai/docs/intro/" target="_blank" rel="noreferrer">
                <BookOpen />
                Documentation
                <ExternalLink className="ml-auto size-3.5" />
              </a>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <a href="https://github.com/vllm-project/semantic-router" target="_blank" rel="noreferrer">
                <GitHubIcon />
                GitHub
                <ExternalLink className="ml-auto size-3.5" />
              </a>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <a href="https://github.com/vllm-project/semantic-router/releases" target="_blank" rel="noreferrer">
                <Sparkles />
                What&apos;s new
                <ExternalLink className="ml-auto size-3.5" />
              </a>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <a href="/api/router/docs" target="_blank" rel="noreferrer">
                <BookOpen />
                Router API reference
                <ExternalLink className="ml-auto size-3.5" />
              </a>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => setCommandOpen(true)}>
              <Keyboard />
              Command menu
              <DropdownMenuShortcut>⌘K</DropdownMenuShortcut>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
