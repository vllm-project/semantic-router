import { useEffect, useRef } from 'react'

import type { RouterConfigWarning } from '../utils/configSaveWarnings'
import { Alert, Button } from '../ui'

export default function ConfigSaveWarnings({
  warnings,
  onDismiss,
}: {
  warnings: RouterConfigWarning[]
  onDismiss?: () => void
}) {
  const notice = useRef<HTMLDivElement>(null)
  // A save made far down a page reports here, at the top of its section.
  useEffect(() => {
    if (warnings.length) notice.current?.scrollIntoView?.({ block: 'nearest' })
  }, [warnings])
  if (!warnings.length) return null
  return (
    <div ref={notice}>
      <Alert
        tone="warning"
        className="my-3"
        actions={
          onDismiss ? (
            <Button size="sm" variant="ghost" onClick={onDismiss}>
              Dismiss
            </Button>
          ) : undefined
        }
        title={
          <>
            Saved. The Router reports{' '}
            {warnings.length === 1 ? 'a warning' : `${warnings.length} warnings`} about this
            configuration:
          </>
        }
      >
        <ul className="mt-1 flex list-disc flex-col gap-1 pl-4 marker:text-subtle-foreground">
          {warnings.map((warning, index) => (
            <li key={`${warning.code}:${warning.field}:${index}`}>
              {warning.field && (
                <code className="mr-1 rounded bg-surface-sunken px-1 py-0.5 font-mono text-xs text-foreground [overflow-wrap:anywhere]">
                  {warning.field}
                </code>
              )}{' '}
              {warning.message}
            </li>
          ))}
        </ul>
      </Alert>
    </div>
  )
}
