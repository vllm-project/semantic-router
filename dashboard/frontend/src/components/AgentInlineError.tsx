import { useEffect, useRef } from 'react'

import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

export default function AgentInlineError({ message }: { message: string }) {
  const ref = useRef<HTMLParagraphElement>(null)
  useEffect(() => {
    ref.current?.focus()
  }, [message])
  return (
    <p ref={ref} className={styles.inlineError} role="alert" tabIndex={-1}>
      <ProductIcon name="alert" />
      {message}
    </p>
  )
}
