import ProductIcon from '../components/ProductIcon'
import styles from './ConfigPageRoutingScopeState.module.css'

interface Props {
  error: string | null
  loading: boolean
  onRetry: () => void
}

export default function ConfigPageRoutingScopeState({ error, loading, onRetry }: Props) {
  const empty = !loading && !error
  return (
    <div className={styles.state} role={error ? 'alert' : 'status'}>
      <div className={styles.copy}>
        <ProductIcon name={error ? 'alert' : loading ? 'activity' : 'mixture'} />
        <div>
          <strong>
            {error ? 'Recipes unavailable' : loading ? 'Loading Recipes' : 'Create a Recipe first'}
          </strong>
          <span>
            {error ??
              (loading
                ? 'Getting the latest Router-managed design.'
                : 'Then shape it here without touching deployment configuration.')}
          </span>
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
