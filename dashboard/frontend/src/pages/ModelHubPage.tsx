import React, { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import { ModelHubArena } from './ModelHubArena'
import { HubHero } from './ModelHubComponents'
import { ModelHubExplorer } from './ModelHubExplorer'
import {
  parseModelHubArenaRoute,
  serializeModelHubArenaRoute,
  type ModelHubArenaRoute,
} from './modelHubArenaSupport'
import { useModelHubPageController } from './modelHubPageController'
import styles from './ModelHubPage.module.css'

const ModelHubPage: React.FC = () => {
  const { catalog, error } = useBuiltInModelCatalog()
  const hub = useModelHubPageController(catalog)
  const [searchParameters, setSearchParameters] = useSearchParams()
  const arenaRoute = useMemo(() => parseModelHubArenaRoute(searchParameters), [searchParameters])
  const setArenaRoute = useCallback(
    (patch: Partial<ModelHubArenaRoute>): void => {
      setSearchParameters(
        serializeModelHubArenaRoute({ ...arenaRoute, ...patch }, searchParameters),
        { replace: true },
      )
    },
    [arenaRoute, searchParameters, setSearchParameters],
  )

  return (
    <main className={styles.container} data-testid="model-hub-page">
      <HubHero stats={hub.stats} />
      {error ? (
        <div className={styles.notice} role="status">
          Live catalog unavailable. Showing the identical bundled release snapshot. {error}
        </div>
      ) : null}

      <ModelHubArena
        catalog={catalog}
        route={arenaRoute}
        setRoute={setArenaRoute}
        openModel={hub.openModelFromBenchmark}
      />

      <ModelHubExplorer catalog={catalog} hub={hub} />
    </main>
  )
}

export default ModelHubPage
