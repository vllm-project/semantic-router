import React, { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import ProductLoadingState from '../components/ProductLoadingState'
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
  const { catalog, error, loading, ready, source, retry } = useBuiltInModelCatalog()
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
      <HubHero stats={hub.stats} evidenceReady={ready} />
      {error ? (
        <div className={styles.notice}>
          <div role="status">
            <strong>
              {!ready
                ? 'Model catalog details are unavailable.'
                : source === 'bundled'
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

      {!ready && loading ? <ProductLoadingState label="Loading model evaluations" compact /> : null}

      {ready ? (
        <ModelHubArena
          catalog={catalog}
          route={arenaRoute}
          setRoute={setArenaRoute}
          openModel={hub.openModelFromArena}
        />
      ) : null}

      <ModelHubExplorer catalog={catalog} hub={hub} evidenceReady={ready} />
    </main>
  )
}

export default ModelHubPage
