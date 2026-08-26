import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import styles from './ConfigPageRoutingScopeState.module.css'

interface Props {
  error: string | null
  loading: boolean
  onRetry: () => void
}

export default function ConfigPageRoutingScopeState({ error, loading, onRetry }: Props) {
  if (loading) return <ProductLoadingState compact label="Loading Recipes" />
  const empty = !loading && !error
  return (
    <div className={styles.state} role={error ? 'alert' : 'status'}>
      <div className={styles.copy}>
        <ProductIcon name={error ? 'alert' : 'mixture'} />
        <div>
          <strong>{error ? 'Recipes unavailable' : 'Create a Recipe first'}</strong>
          <span>{error ?? 'Then shape it here without touching deployment configuration.'}</span>
        </div>
      </div>
      {error ? (
        <button type="button" onClick={onRetry}>
          <ProductIcon name="refresh" />
          Try again
        </button>
      ) : empty ? (
        <a href="/config/entrypoints-recipes">
          <ProductIcon name="plus" />
          Open Recipes
        </a>
      ) : null}
    </div>
  )
}
