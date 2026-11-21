import { useCallback, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import type { ConfigData } from './configPageSupport'
import {
  applyRoutingScopeProjection,
  listRoutingScopes,
  projectConfigForRoutingScope,
  resolveRoutingScope,
  type RoutingScopedConfigLike,
} from '../utils/routingScopes'

/**
 * The routing profile (recipe) a manager page edits. Pages that keep the
 * choice in the URL pass it in; others keep it locally.
 */
export function useRoutingScopeManager(
  config: ConfigData | null,
  controlledChoice?: string | null,
) {
  const routingScopes = useMemo(
    () => listRoutingScopes(config as ConfigData & RoutingScopedConfigLike),
    [config],
  )
  const [localChoice, setSelectedScopeId] = useState('')
  const choice = controlledChoice ?? localChoice
  const selectedScopeId =
    resolveRoutingScope(config as ConfigData & RoutingScopedConfigLike, choice)?.id ?? ''

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

/** The recipe name a scope is saved under; the default scope edits the default recipe. */
export function routingScopeRecipeName(
  config: ConfigData | null,
  scopeId: string,
  isDefault: boolean,
): string {
  return isDefault ? config?.default_recipe?.name || 'default' : scopeId
}

/** Reads and patches the page's URL state, so drawers and tabs survive reloads and links. */
export function useRoutingPageParams() {
  const [params, setParams] = useSearchParams()
  const update = useCallback(
    (patch: Record<string, string | null | undefined>, options: { replace?: boolean } = {}) =>
      setParams(
        (current) => {
          const next = new URLSearchParams(current)
          for (const [key, value] of Object.entries(patch)) {
            if (value === null || value === undefined || value === '') next.delete(key)
            else next.set(key, value)
          }
          return next
        },
        { replace: options.replace ?? false },
      ),
    [setParams],
  )
  return { params, update }
}
