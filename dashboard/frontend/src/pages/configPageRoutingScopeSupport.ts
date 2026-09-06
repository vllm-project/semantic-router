import { useCallback, useEffect, useMemo, useState } from 'react'

import type { ConfigData } from './configPageSupport'
import {
  applyRoutingScopeProjection,
  listRoutingScopes,
  projectConfigForRoutingScope,
  type RoutingScopedConfigLike,
} from '../utils/routingScopes'

export function useRoutingScopeManager(config: ConfigData | null) {
  const routingScopes = useMemo(
    () => listRoutingScopes(config as ConfigData & RoutingScopedConfigLike),
    [config],
  )
  const [selectedScopeId, setSelectedScopeId] = useState('')

  useEffect(() => {
    if (routingScopes.some((scope) => scope.id === selectedScopeId)) return
    setSelectedScopeId(routingScopes[0]?.id ?? '')
  }, [routingScopes, selectedScopeId])

  const scopedConfig = useMemo(
    () =>
      config
        ? projectConfigForRoutingScope(
            config as ConfigData & RoutingScopedConfigLike,
            selectedScopeId,
          )
        : null,
    [config, selectedScopeId],
  )

  const applyScopedConfig = useCallback(
    (projectedConfig: ConfigData): ConfigData => {
      if (!config) {
        throw new Error('Configuration not loaded yet.')
      }
      return applyRoutingScopeProjection(
        config as ConfigData & RoutingScopedConfigLike,
        projectedConfig as ConfigData & RoutingScopedConfigLike,
        selectedScopeId,
      )
    },
    [config, selectedScopeId],
  )

  return {
    applyScopedConfig,
    routingScopes,
    scopedConfig,
    selectedScope: routingScopes.find((scope) => scope.id === selectedScopeId),
    selectedScopeId,
    setSelectedScopeId,
  }
}
