import {
  Component,
  Suspense,
  lazy,
  useMemo,
  useState,
  type ComponentType,
  type ErrorInfo,
  type ReactNode,
} from 'react'

import RouteLoadingFallback from './RouteLoadingFallback'
import { resetDashboardRouteLoader, type RouteLoader } from './routeLoaders'
import { getRouteLoadFailureCopy, routeLoadErrorMessage } from './routeLoadFailureSupport'
import styles from './RecoverableLazyRoute.module.css'

interface RouteLoadErrorBoundaryProps {
  children: ReactNode
  routeLabel: string
  onRetry: () => void
}

interface RouteLoadErrorBoundaryState {
  error: Error | null
}

class RouteLoadErrorBoundary extends Component<
  RouteLoadErrorBoundaryProps,
  RouteLoadErrorBoundaryState
> {
  private retryButton: HTMLButtonElement | null = null

  state: RouteLoadErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): RouteLoadErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Lazy dashboard route failed:', error, errorInfo)
  }

  componentDidUpdate(
    _previousProps: RouteLoadErrorBoundaryProps,
    previousState: RouteLoadErrorBoundaryState,
  ) {
    if (!previousState.error && this.state.error) {
      window.requestAnimationFrame(() => this.retryButton?.focus())
    }
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    const copy = getRouteLoadFailureCopy(error, this.props.routeLabel)
    return (
      <section className={styles.surface} role="alert" data-testid="route-load-error">
        <div className={styles.panel}>
          <span className={styles.eyebrow}>{copy.eyebrow}</span>
          <h1 className={styles.title}>{copy.title}</h1>
          <p className={styles.description}>{copy.description}</p>
          <div className={styles.actions}>
            <button
              ref={(element) => {
                this.retryButton = element
              }}
              type="button"
              className={styles.primaryButton}
              onClick={this.props.onRetry}
            >
              Retry route
            </button>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => window.location.reload()}
            >
              Reload dashboard
            </button>
          </div>
          <details className={styles.details}>
            <summary>Technical details</summary>
            <code>{routeLoadErrorMessage(error)}</code>
          </details>
        </div>
      </section>
    )
  }
}

export interface RecoverableLazyRouteProps<Props extends object> {
  loader: () => Promise<{ default: ComponentType<Props> }>
  routeLabel: string
  componentProps?: Props
}

function createRetryableLazyPage<Props extends object>(
  loader: RecoverableLazyRouteProps<Props>['loader'],
  _attempt: number,
) {
  return lazy(loader)
}

export default function RecoverableLazyRoute<Props extends object = Record<string, never>>({
  loader,
  routeLabel,
  componentProps,
}: RecoverableLazyRouteProps<Props>) {
  const [attempt, setAttempt] = useState(0)
  const LazyPage = useMemo(() => createRetryableLazyPage(loader, attempt), [attempt, loader])
  const RenderablePage = LazyPage as unknown as ComponentType<Record<string, unknown>>
  const renderProps = (componentProps ?? {}) as Record<string, unknown>

  const handleRetry = () => {
    resetDashboardRouteLoader(loader as RouteLoader)
    setAttempt((current) => current + 1)
  }

  return (
    <RouteLoadErrorBoundary key={attempt} routeLabel={routeLabel} onRetry={handleRetry}>
      <Suspense fallback={<RouteLoadingFallback />}>
        <RenderablePage {...renderProps} />
      </Suspense>
    </RouteLoadErrorBoundary>
  )
}
