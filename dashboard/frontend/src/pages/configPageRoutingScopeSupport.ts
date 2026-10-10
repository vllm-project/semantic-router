import { useCallback, useMemo, useState } from 'react'

import type { ConfigData } from './configPageSupport'
import {
  applyRoutingScopeProjection,
  listRoutingScopes,
  projectConfigForRoutingScope,
  resolveRoutingScope,
  type RoutingScopedConfigLike,
} from '../utils/routingScopes'

export function useRoutingScopeManager(config: ConfigData | null) {
  const routingScopes = useMemo(
    () => listRoutingScopes(config as ConfigData & RoutingScopedConfigLike),
    [config],
  )
  const [choice, setSelectedScopeId] = useState('')
  const selectedScopeId = resolveRoutingScope(config as ConfigData & RoutingScopedConfigLike, choice)?.id ?? ''

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
