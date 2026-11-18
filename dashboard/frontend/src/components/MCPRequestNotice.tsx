import { Alert, Button } from '../ui'

interface MCPRequestNoticeProps {
  message: string
  retryLabel: string
  title: string
  onDismiss?: () => void
  onRetry: () => void
}

export function MCPRequestNotice({
  message,
  retryLabel,
  title,
  onDismiss,
  onRetry,
}: MCPRequestNoticeProps) {
  return (
    <div role="alert">
      <Alert
        tone="danger"
        title={title}
        actions={
          <>
            <Button size="sm" variant="outline" onClick={onRetry}>
              {retryLabel}
            </Button>
            {onDismiss ? (
              <Button size="sm" variant="ghost" onClick={onDismiss}>
                Dismiss
              </Button>
            ) : null}
          </>
        }
      >
        {message}
      </Alert>
    </div>
  )
}
