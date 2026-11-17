import { lazy, Suspense } from 'react'
import { useSearchParams } from 'react-router-dom'
import RouteLoadingFallback from '../app/RouteLoadingFallback'

const DecisionModelPage = lazy(() => import('./DecisionModelPage'))
const SystemOnePlaygroundPage = lazy(() => import('./SystemOnePlaygroundPage'))
const DecisionMonitoringPage = lazy(() => import('./DecisionMonitoringPage'))

/** Decision models with deployment, playground and monitoring as tabs (`?tab=`). */
export default function DecisionModelsWorkspace() {
  const [params] = useSearchParams()
  const tab = params.get('tab')
  return (
    <Suspense fallback={<RouteLoadingFallback />}>
      {tab === 'playground' ? <SystemOnePlaygroundPage /> : tab === 'monitoring' ? <DecisionMonitoringPage /> : <DecisionModelPage />}
    </Suspense>
  )
}
