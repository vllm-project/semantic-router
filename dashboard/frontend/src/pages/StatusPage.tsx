import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { HeartPulse, RefreshCw } from 'lucide-react'

import {
  Alert,
  Button,
  IconTile,
  PageContainer,
  PageHeader,
  Skeleton,
  StatusLabel,
  cn,
} from '../ui'
import StatusAvailabilityPanel from './StatusAvailabilityPanel'
import StatusRuntimePanel, { type StatusSnapshot } from './StatusRuntimePanel'
import { fetchDashboardJson } from './dashboardPageRequests'
import { formatElapsed } from './statusPageSupport'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'

const REFRESH_MS = 10_000

function useNow(intervalMs: number) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])
  return now
}

function StatusSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-hidden="true">
      <Skeleton className="h-[148px] rounded-xl" />
      <Skeleton className="h-72 rounded-xl" />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Skeleton className="h-64 rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
    </div>
  )
}

const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<StatusSnapshot | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const now = useNow(1000)

  const fetchStatus = useCallback(async () => {
    setChecking(true)
    try {
      setStatus(await fetchDashboardJson<StatusSnapshot>('/api/status', 'System status'))
      setLastUpdated(new Date())
      setError(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'System status is unavailable.')
    } finally {
      setIsLoading(false)
      setChecking(false)
    }
  }, [])

  const statusRequest = useMemo(() => createVisibilityAwareRequest(fetchStatus), [fetchStatus])

  useEffect(() => {
    void statusRequest.run({ allowHidden: true })
    const refreshWhenVisible = () => {
      if (!document.hidden) void statusRequest.run()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)
    const interval = window.setInterval(refreshWhenVisible, REFRESH_MS)
    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [statusRequest])

  const live = Boolean(lastUpdated && !error)

  return (
    <PageContainer>
      <div data-testid="status-page" className="min-w-0">
        <PageHeader
          icon={<IconTile icon={HeartPulse} tone="green" size="lg" />}
          title="Status"
          description="Health of every service in this deployment, the Router's live state and how the stack is deployed."
          actions={
            <>
              <span className="mr-1 flex flex-col items-end gap-0.5 text-xs" aria-live="polite">
                <StatusLabel tone={live ? 'success' : error ? 'danger' : 'info'} pulse={live}>
                  {live ? 'Live · checks every 10 s' : error ? 'Unavailable' : 'Checking'}
                </StatusLabel>
                {lastUpdated ? (
                  <span className="text-muted-foreground tabular-nums">
                    Last checked {formatElapsed(now - lastUpdated.getTime())}
                  </span>
                ) : null}
              </span>
              <Button
                variant="outline"
                onClick={() => void statusRequest.run({ allowHidden: true })}
                aria-label="Refresh system status"
                title={
                  lastUpdated ? `Last checked ${lastUpdated.toLocaleTimeString()}` : 'Check now'
                }
                disabled={checking && !status}
              >
                <RefreshCw aria-hidden="true" className={cn(checking && 'animate-spin')} />
                Refresh
              </Button>
            </>
          }
        />

        {error ? (
          <div role="alert" className="mb-4">
            <Alert
              tone="danger"
              title={status ? 'Showing the last known status' : 'Status check failed'}
              actions={
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void statusRequest.run({ allowHidden: true })}
                >
                  Try again
                </Button>
              }
            >
              {error} The page keeps retrying every 10 seconds.
            </Alert>
          </div>
        ) : null}

        {isLoading && !status ? (
          <>
            <span role="status" className="sr-only">
              Checking service availability
            </span>
            <StatusSkeleton />
          </>
        ) : (
          <div className="flex flex-col gap-6">
            <StatusAvailabilityPanel
              status={status}
              lastUpdated={lastUpdated}
              unavailable={Boolean(error)}
            />
            <StatusRuntimePanel status={status} />
          </div>
        )}
      </div>
    </PageContainer>
  )
}

export default StatusPage
