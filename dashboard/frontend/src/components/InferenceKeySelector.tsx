import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import ProductIcon from './ProductIcon'
import styles from './InferenceKeySelector.module.css'

interface InferenceKeySelectorProps {
  className?: string
  disabled?: boolean
  label?: string
}

export default function InferenceKeySelector({
  className = '',
  disabled = false,
  label = 'API key',
}: InferenceKeySelectorProps) {
  const { keys, keysStatus, selectedKeyId, setSelectedKeyId } = useInferenceRoutingAccess()
  if (keys.length < 2) return null

  return (
    <label className={`${styles.control} ${className}`}>
      <ProductIcon name="key" aria-hidden="true" />
      <span>{label}</span>
      <select
        aria-label={`${label} context`}
        value={selectedKeyId}
        disabled={disabled || keysStatus !== 'ready'}
        onChange={(event) => setSelectedKeyId(event.target.value)}
      >
        {keys.map((key) => (
          <option key={key.keyId} value={key.keyId}>
            {key.name}
          </option>
        ))}
      </select>
    </label>
  )
}
