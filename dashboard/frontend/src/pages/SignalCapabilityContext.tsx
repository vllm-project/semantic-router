import type { ReactNode } from 'react'
import { SignalCapabilityContext, useSignalCapabilities } from './signalCapabilityState'
import { useDecisionTasks } from './useDecisionTasks'
import type { SignalCapabilityScope } from './signalCapabilities'

export function SignalCapabilityProvider({
  scope,
  children,
  active = true,
}: {
  scope: SignalCapabilityScope
  children: ReactNode
  active?: boolean
}) {
  const tasks = useDecisionTasks(active)
  return (
    <SignalCapabilityContext.Provider value={{ ...tasks, scope }}>
      {children}
    </SignalCapabilityContext.Provider>
  )
}
export function SignalCapabilityNotice({ reason }: { reason?: string }) {
  const { error, refresh } = useSignalCapabilities()
  if (!reason && !error) return null
  return (
    <p role="status" style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--text-sm)' }}>
      {error || reason}{' '}
      {error && (
        <button type="button" onClick={refresh}>
          Retry
        </button>
      )}
    </p>
  )
}
