import React, { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppShell } from '../shell/AppShell'
import type { NavBadges, ShellInstance } from '../shell/AppSidebar'
import type { NavItem } from '../shell/navigation'
import type { ShellNotification } from '../shell/Topbar'
import { useAuth } from '../contexts/AuthContext'
import { useCapability } from '../contexts/CapabilitiesContext'
import { isCapabilityUsable } from '../contexts/dashboardCapabilities'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canAccessDashboardPath } from '../utils/accessControl'
import { useShellStatus } from './useShellStatus'

export interface AppShellLayoutProps {
  children: React.ReactNode
}

const modeLabel: Record<string, string> = { router: 'Router', engine: 'Engine' }

/** Wraps authenticated pages in the sidebar shell, fed by the session, capabilities and `/api/status`. */
const AppShellLayout: React.FC<AppShellLayoutProps> = ({ children }) => {
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const { platform, isReadonly } = useReadonly()
  const evaluation = useCapability('evaluation')
  const mcp = useCapability('mcp')
  const shellStatus = useShellStatus()
  const status = shellStatus.status

  const isItemVisible = (item: NavItem) => {
    if (!canAccessDashboardPath(user, item.to)) return false
    if (item.id === 'evaluations' && evaluation && !isCapabilityUsable(evaluation)) return false
    if (item.id === 'integrations' && mcp && !isCapabilityUsable(mcp)) return false
    return true
  }

  const instance: ShellInstance = useMemo(() => {
    const unhealthy = status?.services.filter((service) => !service.healthy) ?? []
    return {
      name:
        status?.deployment_type && status.deployment_type !== 'none'
          ? `vllm-sr · ${status.deployment_type}`
          : 'vllm-sr',
      version: status?.version ?? '—',
      mode: modeLabel[status?.serving_mode ?? ''] ?? 'Router',
      status: shellStatus.tone,
      statusLabel: !status
        ? shellStatus.label
        : unhealthy.length === 0
          ? `${shellStatus.label} · ${status.services.length} services reachable`
          : `${unhealthy.length} of ${status.services.length} services need attention`,
      endpoint: '',
      details: [
        { label: 'Version', value: status?.version ?? 'unknown' },
        { label: 'Mode', value: modeLabel[status?.serving_mode ?? ''] ?? 'Router' },
        { label: 'Deployment', value: status?.deployment_type ?? 'unknown' },
        ...(platform
          ? [{ label: 'Platform', value: platform === 'rocm' ? 'AMD ROCm' : platform }]
          : []),
        ...(isReadonly ? [{ label: 'Access', value: 'Read-only' }] : []),
      ],
    }
  }, [status, shellStatus.tone, shellStatus.label, platform, isReadonly])

  const notifications: ShellNotification[] = useMemo(
    () =>
      (status?.services ?? [])
        .filter((service) => !service.healthy)
        .map((service) => ({
          id: service.name,
          title: `${service.name} needs attention`,
          body: service.message || `Reported ${service.status}.`,
          time: 'now',
          tone: 'warning' as const,
          unread: true,
        })),
    [status],
  )

  const badges: NavBadges = useMemo(
    () =>
      status && status.overall?.toLowerCase() !== 'healthy'
        ? { status: { tone: shellStatus.tone } }
        : {},
    [status, shellStatus.tone],
  )

  return (
    <AppShell
      account={{
        name: user?.name?.trim() || user?.email?.split('@')[0] || 'Account',
        email: user?.email?.trim() || '',
        role: user?.role || 'member',
      }}
      instance={instance}
      status={{ tone: shellStatus.tone, label: shellStatus.label }}
      notifications={notifications}
      badges={badges}
      isItemVisible={isItemVisible}
      onSignOut={() => {
        logout()
        navigate('/login', { replace: true })
      }}
    >
      {children}
    </AppShell>
  )
}

export default AppShellLayout
