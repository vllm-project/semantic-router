// topology/hooks/useCollapseState.ts - Collapse state management

import { useState, useCallback } from 'react'
import { CollapseState, SignalType } from '../types'
import { SIGNAL_TYPES } from '../constants'

interface UseCollapseStateResult {
  collapseState: CollapseState
  toggleCollapse: (type: 'signalGroup' | 'decision' | 'pluginChain', key: string) => void
  expandAll: () => void
  collapseAll: () => void
  isCollapsed: (type: 'signalGroup' | 'decision' | 'pluginChain', key: string) => boolean
}

export function useCollapseState(): UseCollapseStateResult {
  const [collapseState, setCollapseState] = useState<CollapseState>({
    signalGroups: Object.fromEntries(SIGNAL_TYPES.map(t => [t, false])) as Record<SignalType, boolean>,
    decisions: {},
    pluginChains: {},
  })

  const toggleCollapse = useCallback((
    type: 'signalGroup' | 'decision' | 'pluginChain',
    key: string
  ) => {
    setCollapseState(prev => {
      const section = type === 'signalGroup' ? 'signalGroups' :
                      type === 'decision' ? 'decisions' : 'pluginChains'
      return {
        ...prev,
        [section]: {
          ...prev[section],
          [key]: !prev[section][key as keyof typeof prev[typeof section]],
        },
      }
    })
  }, [])

  const expandAll = useCallback(() => {
    setCollapseState({
      signalGroups: Object.fromEntries(SIGNAL_TYPES.map(t => [t, false])) as Record<SignalType, boolean>,
      decisions: {},
      pluginChains: {},
    })
  }, [])

  const collapseAll = useCallback(() => {
    setCollapseState(prev => ({
      signalGroups: Object.fromEntries(SIGNAL_TYPES.map(t => [t, true])) as Record<SignalType, boolean>,
      decisions: Object.fromEntries(Object.keys(prev.decisions).map(k => [k, true])),
      pluginChains: Object.fromEntries(Object.keys(prev.pluginChains).map(k => [k, true])),
    }))
  }, [])

  const isCollapsed = useCallback((
    type: 'signalGroup' | 'decision' | 'pluginChain',
    key: string
  ): boolean => {
    if (type === 'signalGroup') {
      return collapseState.signalGroups[key as SignalType] ?? false
    } else if (type === 'decision') {
      return collapseState.decisions[key] ?? false
    } else {
      return collapseState.pluginChains[key] ?? false
    }
  }, [collapseState])

  return {
    collapseState,
    toggleCollapse,
    expandAll,
    collapseAll,
    isCollapsed,
  }
}
