import { ChevronDown, RefreshCw } from 'lucide-react'
import { Button } from '../../ui'
import { BenchLoading, Notice } from './benchUi'

export default function RunOptionsStatus({
  loading,
  error,
  next,
  loaded,
  scanLimited,
  label,
  onRetry,
  onMore,
}: {
  loading: boolean
  error: string
  next: string | null
  loaded: boolean
  scanLimited: boolean
  label: string
  onRetry: () => void
  onMore: () => void
}) {
  return (
    <>
      {scanLimited && (
        <Notice tone="warning" as="p">
          Some saved evidence exceeded the verification limit. Only verified combinations are shown.
        </Notice>
      )}
      {loading && <BenchLoading label={`Loading ${label}…`} />}
      {error && (
        <Notice tone="danger" role="alert">
          <p>{error}</p>
          <div>
            <Button variant="outline" size="xs" disabled={loading} onClick={onRetry}>
              <RefreshCw aria-hidden="true" className="size-[15px]" /> Retry {label}
            </Button>
          </div>
        </Notice>
      )}
      {loaded && next && !error && (
        <Button variant="outline" size="xs" className="my-1" disabled={loading} onClick={onMore}>
          Load more {label} <ChevronDown aria-hidden="true" />
        </Button>
      )}
    </>
  )
}
