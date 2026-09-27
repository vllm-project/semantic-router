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
  const { catalog, error, loading, source, retry } = useBuiltInModelCatalog()
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
        <div className={styles.notice}>
          <div role="status">
            <strong>
              {source === 'bundled'
                ? 'Showing the catalog bundled with this Dashboard.'
                : 'Showing the last catalog loaded from the server.'}
            </strong>
            <span>Server catalog could not be refreshed. {error}</span>
          </div>
          <button type="button" onClick={retry} disabled={loading}>
            {loading ? 'Retrying…' : 'Retry'}
          </button>
        </div>
      ) : null}

      <ModelHubArena
        catalog={catalog}
        route={arenaRoute}
        setRoute={setArenaRoute}
        openModel={hub.openModelFromArena}
      />

      <ModelHubExplorer catalog={catalog} hub={hub} />
    </main>
  )
}

export default ModelHubPage
