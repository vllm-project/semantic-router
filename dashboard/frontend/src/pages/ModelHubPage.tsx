import React from 'react'

import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import { HubHero } from './ModelHubComponents'
import { ModelHubExplorer } from './ModelHubExplorer'
import { useModelHubPageController } from './modelHubPageController'
import styles from './ModelHubPage.module.css'

const ModelHubPage: React.FC = () => {
  const { catalog, error } = useBuiltInModelCatalog()
  const hub = useModelHubPageController(catalog)

  return (
    <main className={styles.container} data-testid="model-hub-page">
      <HubHero stats={hub.stats} />
      {error ? (
        <div className={styles.notice} role="status">
          Live catalog unavailable. Showing the identical bundled release snapshot. {error}
        </div>
      ) : null}

      <ModelHubExplorer catalog={catalog} hub={hub} />
    </main>
  )
}

export default ModelHubPage
