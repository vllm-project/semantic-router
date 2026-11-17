import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  BookOpen,
  Check,
  ChevronsUpDown,
  Keyboard,
  LogOut,
  Monitor,
  Moon,
  PanelLeft,
  Palette,
  Sun,
} from 'lucide-react'
import {
  Avatar,
  Badge,
  BrandLockup,
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
  Popover,
  PopoverContent,
  PopoverTrigger,
  StatusDot,
  Tooltip,
  cn,
  type StatusTone,
} from '../ui'
import { navigation, navItemMatch, type NavItem } from './navigation'
import { useShell } from './shell-context'
import { useTheme, type AccentPreset, type ThemePreference } from './theme'

export interface ShellAccount {
  name: string
  email: string
  role: string
}

export interface ShellInstance {
  name: string
  version: string
  mode: string
  status: StatusTone
  statusLabel: string
  endpoint: string
  details: { label: string; value: string }[]
}

export interface NavBadges {
  [navItemId: string]: { label?: string; tone?: StatusTone } | undefined
}

function SidebarLink({
  item,
  collapsed,
  badge,
  activePrefix,
}: {
  item: NavItem
  collapsed: boolean
  badge?: NavBadges[string]
  activePrefix?: string
}) {
  const { setMobileOpen } = useShell()
  const location = useLocation()
  const active = activePrefix
    ? location.pathname === activePrefix || location.pathname.startsWith(`${activePrefix}/`)
    : navItemMatch(item, location.pathname) > 0
  const link = (
    <NavLink
      to={item.to}
      end={item.end}
      onClick={() => setMobileOpen(false)}
      className={cn(
        'group/link relative flex h-8 items-center gap-2.5 rounded-md px-2 text-[13px] font-medium outline-none transition-colors',
        'text-sidebar-foreground hover:bg-sidebar-accent/70 hover:text-sidebar-accent-foreground focus-visible:ring-[3px] focus-visible:ring-sidebar-ring',
        active && 'text-sidebar-accent-foreground',
        collapsed && 'justify-center px-0',
      )}
    >
      {active ? (
        <motion.span
          layoutId="sidebar-active"
          className="absolute inset-0 rounded-md bg-sidebar-accent shadow-[inset_0_0_0_1px_var(--sidebar-border)]"
          transition={{ type: 'spring', bounce: 0.15, duration: 0.45 }}
        />
      ) : null}
      {active ? (
        <motion.span
          layoutId="sidebar-active-bar"
          className="absolute -left-2 top-1.5 bottom-1.5 w-[3px] rounded-r-full bg-primary"
          transition={{ type: 'spring', bounce: 0.15, duration: 0.45 }}
        />
      ) : null}
      <item.icon
        className={cn(
          'relative size-4 shrink-0 transition-colors',
          active
            ? 'text-primary'
            : 'text-subtle-foreground group-hover/link:text-sidebar-accent-foreground',
        )}
        strokeWidth={1.9}
      />
      {collapsed ? null : <span className="relative truncate">{item.label}</span>}
      {!collapsed && badge ? (
        <span className="relative ml-auto flex items-center">
          {badge.label ? (
            <Badge
              size="sm"
              tone={
                badge.tone === 'warning'
                  ? 'warning'
                  : badge.tone === 'danger'
                    ? 'danger'
                    : badge.tone === 'success'
                      ? 'success'
                      : 'brand'
              }
            >
              {badge.label}
            </Badge>
          ) : (
            <StatusDot tone={badge.tone} pulse />
          )}
        </span>
      ) : null}
      {collapsed && badge ? (
        <StatusDot tone={badge.tone} className="absolute right-1 top-1" />
      ) : null}
    </NavLink>
  )
  if (!collapsed) return link
  return (
    <Tooltip content={item.label} side="right">
      {link}
    </Tooltip>
  )
}

/** Version pill beside the wordmark (LiteLLM-style); opens the connected instance's details. */
function VersionBadge({ instance }: { instance: ShellInstance }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={`Version ${instance.version}, ${instance.statusLabel}`}
          className="inline-flex h-5 max-w-[96px] shrink-0 items-center gap-1 rounded-full border border-sidebar-border bg-surface/80 px-1.5 font-mono text-[10.5px] font-medium text-muted-foreground outline-none transition-colors hover:border-border-strong hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-sidebar-ring data-[state=open]:text-foreground dark:bg-white/[0.04]"
        >
          <StatusDot tone={instance.status} className="size-1.5" />
          <span className="truncate">{instance.version}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" side="bottom" className="w-72 p-0">
        <div className="flex items-center gap-2.5 border-b border-border px-3.5 py-3">
          <StatusDot tone={instance.status} pulse />
          <div className="flex min-w-0 flex-col">
            <span className="text-[13px] font-semibold">{instance.name}</span>
            <span className="text-xs text-muted-foreground">{instance.statusLabel}</span>
          </div>
        </div>
        <dl className="grid gap-2 px-3.5 py-3 text-xs">
          {instance.details.map((detail) => (
            <div key={detail.label} className="flex items-center justify-between gap-3">
              <dt className="text-subtle-foreground">{detail.label}</dt>
              <dd className="truncate font-mono text-[11.5px] text-foreground/85">
                {detail.value}
              </dd>
            </div>
          ))}
        </dl>
        <div className="border-t border-border p-1.5">
          <NavLink
            to="/observe/status"
            className="flex h-8 items-center rounded-md px-2 text-[13px] text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            Open status page
          </NavLink>
        </div>
      </PopoverContent>
    </Popover>
  )
}

export function AccountMenu({
  account,
  collapsed,
  onSignOut,
}: {
  account: ShellAccount
  collapsed: boolean
  onSignOut: () => void
}) {
  const { preference, setPreference, accent, setAccent } = useTheme()
  const { setCommandOpen } = useShell()
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className={cn(
            'flex w-full items-center gap-2.5 rounded-lg p-1.5 text-left outline-none transition-colors hover:bg-sidebar-accent focus-visible:ring-[3px] focus-visible:ring-sidebar-ring data-[state=open]:bg-sidebar-accent',
            collapsed && 'justify-center',
          )}
        >
          <Avatar name={account.name} size={28} />
          {collapsed ? null : (
            <>
              <span className="flex min-w-0 flex-1 flex-col leading-tight">
                <span className="truncate text-[13px] font-medium text-foreground">
                  {account.name}
                </span>
                <span className="truncate text-[11px] text-subtle-foreground">{account.email}</span>
              </span>
              <ChevronsUpDown className="size-3.5 text-subtle-foreground" />
            </>
          )}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side={collapsed ? 'right' : 'top'} align="start" className="w-60">
        <DropdownMenuLabel className="flex items-center gap-2.5 py-2 normal-case tracking-normal">
          <Avatar name={account.name} size={32} />
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-[13px] font-semibold text-foreground">
              {account.name}
            </span>
            <span className="truncate text-xs font-normal text-muted-foreground">
              {account.email}
            </span>
          </span>
          <Badge size="sm" tone="brand" className="ml-auto capitalize">
            {account.role}
          </Badge>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            {preference === 'dark' ? <Moon /> : preference === 'light' ? <Sun /> : <Monitor />}
            Theme
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-40">
            <DropdownMenuRadioGroup
              value={preference}
              onValueChange={(value) => setPreference(value as ThemePreference)}
            >
              <DropdownMenuRadioItem value="light">Light</DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="dark">Dark</DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="system">System</DropdownMenuRadioItem>
            </DropdownMenuRadioGroup>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            <Palette />
            Accent
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-44">
            {(['blue', 'gold'] as AccentPreset[]).map((preset) => (
              <DropdownMenuItem key={preset} onSelect={() => setAccent(preset)}>
                <span
                  className={cn(
                    'size-3.5 rounded-full ring-1 ring-black/10',
                    preset === 'blue' ? 'bg-[#30A2FF]' : 'bg-[#FDB515]',
                  )}
                />
                {preset === 'blue' ? 'vLLM Blue' : 'vLLM Gold'}
                {accent === preset ? <Check className="ml-auto !text-primary" /> : null}
              </DropdownMenuItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuItem onSelect={() => setCommandOpen(true)}>
          <Keyboard />
          Command menu
          <DropdownMenuShortcut>⌘K</DropdownMenuShortcut>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <a href="https://vllm-sr.ai/docs/intro/" target="_blank" rel="noreferrer">
            <BookOpen />
            Documentation
          </a>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onSelect={onSignOut}>
          <LogOut />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function SidebarContents({
  collapsed,
  account,
  instance,
  badges,
  onSignOut,
  showCollapse = true,
  isItemVisible = () => true,
}: {
  collapsed: boolean
  account: ShellAccount
  instance: ShellInstance
  badges?: NavBadges
  onSignOut: () => void
  showCollapse?: boolean
  isItemVisible?: (item: NavItem) => boolean
}) {
  const { setCollapsed } = useShell()
  const groups = navigation
    .map((group) => ({ ...group, items: group.items.filter(isItemVisible) }))
    .filter((group) => group.items.length > 0)
  return (
    <div className="flex h-full flex-col">
      <div
        className={cn(
          'flex h-14 shrink-0 items-center gap-2 px-3',
          collapsed && 'justify-center px-0',
        )}
      >
        <NavLink
          to="/"
          className="min-w-0 rounded-lg outline-none focus-visible:ring-[3px] focus-visible:ring-sidebar-ring"
          aria-label="Home"
        >
          <BrandLockup collapsed={collapsed} />
        </NavLink>
        {collapsed ? null : <VersionBadge instance={instance} />}
        {showCollapse && !collapsed ? (
          <Tooltip content="Collapse sidebar" shortcut="⌘B" side="right">
            <Button
              variant="ghost"
              size="icon-xs"
              className="ml-auto text-subtle-foreground"
              onClick={() => setCollapsed(true)}
              aria-label="Collapse sidebar"
            >
              <PanelLeft />
            </Button>
          </Tooltip>
        ) : null}
      </div>
      <nav
        aria-label="Primary"
        className={cn('flex-1 overflow-y-auto overflow-x-hidden px-3 pb-3', collapsed && 'px-2')}
      >
        {groups
          .filter((group) => group.placement !== 'footer')
          .map((group) => (
            <div key={group.id} className="mt-3 first:mt-1">
              {group.label ? (
                collapsed ? (
                  <div className="mx-auto mb-2 mt-1 h-px w-5 bg-sidebar-border" />
                ) : (
                  <div className="mb-1 px-2 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground/90">
                    {group.label}
                  </div>
                )
              ) : null}
              <div className="flex flex-col gap-px">
                {group.items.map((item) => (
                  <SidebarLink
                    key={item.id}
                    item={item}
                    collapsed={collapsed}
                    badge={badges?.[item.id]}
                  />
                ))}
              </div>
            </div>
          ))}
      </nav>
      <div
        className={cn(
          'border-t border-sidebar-border p-2',
          collapsed && 'flex flex-col items-center gap-1',
        )}
      >
        {groups
          .filter((group) => group.placement === 'footer')
          .map((group) => (
            <div key={group.id} className={cn('mb-1', !collapsed && 'px-1')}>
              <SidebarLink
                item={{
                  id: group.id,
                  label: group.label ?? group.id,
                  to: group.items[0].to,
                  icon: group.icon ?? group.items[0].icon,
                  description: group.items[0].description,
                }}
                activePrefix={`/${group.id}`}
                collapsed={collapsed}
              />
            </div>
          ))}
        {collapsed && showCollapse ? (
          <Tooltip content="Expand sidebar" shortcut="⌘B" side="right">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setCollapsed(false)}
              aria-label="Expand sidebar"
              className="text-subtle-foreground"
            >
              <PanelLeft className="rotate-180" />
            </Button>
          </Tooltip>
        ) : null}
        <AccountMenu account={account} collapsed={collapsed} onSignOut={onSignOut} />
      </div>
    </div>
  )
}

export function AppSidebar(props: {
  account: ShellAccount
  instance: ShellInstance
  badges?: NavBadges
  onSignOut: () => void
  isItemVisible?: (item: NavItem) => boolean
}) {
  const { collapsed } = useShell()
  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 60 : 248 }}
      transition={{ type: 'spring', bounce: 0, duration: 0.38 }}
      className="relative hidden h-full shrink-0 overflow-hidden bg-sidebar lg:block"
    >
      <SidebarContents collapsed={collapsed} {...props} />
    </motion.aside>
  )
}
