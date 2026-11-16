import { useEffect, useRef } from 'react'

import type { RouterConfigWarning } from '../utils/configSaveWarnings'
import styles from './ConfigSaveWarnings.module.css'

export default function ConfigSaveWarnings({ warnings }: { warnings: RouterConfigWarning[] }) {
  const notice = useRef<HTMLDivElement>(null)
  // A save made far down a page reports here, at the top of its section.
  useEffect(() => {
    if (warnings.length) notice.current?.scrollIntoView?.({ block: 'nearest' })
  }, [warnings])
  if (!warnings.length) return null
  return (
    <div className={styles.warnings} role="status" ref={notice}>
      <strong>
        Saved. The Router reports{' '}
        {warnings.length === 1 ? 'a warning' : `${warnings.length} warnings`} about this
        configuration:
      </strong>
      <ul>
        {warnings.map((warning, index) => (
          <li key={`${warning.code}:${warning.field}:${index}`}>
            {warning.field && <code>{warning.field}</code>} {warning.message}
          </li>
        ))}
      </ul>
    </div>
  )
}
