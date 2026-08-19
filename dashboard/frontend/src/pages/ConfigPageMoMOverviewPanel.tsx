import { useCallback, useEffect, useState } from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import type { RecipeDescriptor } from '../types/recipe'
import { getActiveRecipe } from '../utils/recipeApi'
import ConfigPageMoMActiveRecipePanel from './ConfigPageMoMActiveRecipePanel'
import ConfigPageMoMCatalogPanel from './ConfigPageMoMCatalogPanel'
import ConfigPageMoMPackagesPanel from './ConfigPageMoMPackagesPanel'
import styles from './ConfigPageMoMWorkspace.module.css'
import type { RecipePackageCapabilities } from './configPageMoMPackagesSupport'

interface ConfigPageMoMOverviewPanelProps {
  catalog: BuiltInModelCatalog | null
  catalogLoading: boolean
  catalogError: string | null
  onCatalogRetry: () => void
  packageCapabilities: RecipePackageCapabilities
  recipeRevision: number
  onRecipeLifecycleChanged: () => Promise<boolean>
}

export default function ConfigPageMoMOverviewPanel({
  catalog,
  catalogLoading,
  catalogError,
  onCatalogRetry,
  packageCapabilities,
  recipeRevision,
  onRecipeLifecycleChanged,
}: ConfigPageMoMOverviewPanelProps) {
  const [recipe, setRecipe] = useState<RecipeDescriptor | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [localRevision, setLocalRevision] = useState(0)
  const refresh = useCallback(() => setLocalRevision((current) => current + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void getActiveRecipe(controller.signal)
      .then(setRecipe)
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setRecipe(null)
        setError(reason instanceof Error ? reason.message : 'Failed to load the active Recipe.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [localRevision, recipeRevision])

  return (
    <div className={styles.overviewStack}>
      <ConfigPageMoMCatalogPanel
        catalog={catalog}
        loading={catalogLoading}
        error={catalogError}
        onRetry={onCatalogRetry}
      />
      <ConfigPageMoMActiveRecipePanel
        recipe={recipe}
        loading={loading}
        error={error}
        onRetry={refresh}
      />
      <ConfigPageMoMPackagesPanel
        packageCapabilities={packageCapabilities}
        recipeRevision={recipeRevision}
        onRecipeLifecycleChanged={onRecipeLifecycleChanged}
      />
    </div>
  )
}
