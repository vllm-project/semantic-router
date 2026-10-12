import { createContext, useContext } from 'react'
import type { DecisionTasks } from './useDecisionTasks'
import type { SignalCapabilityScope } from './signalCapabilities'

interface Context {
  data: DecisionTasks | null
  error: string | null
  refresh: () => void
  scope: SignalCapabilityScope
}
export const SignalCapabilityContext = createContext<Context>({
  data: null,
  error: null,
  refresh: () => {},
  scope: { recipe: 'default' },
})
export const useSignalCapabilities = () => useContext(SignalCapabilityContext)
