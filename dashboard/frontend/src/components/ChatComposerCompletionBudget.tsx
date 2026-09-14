import styles from './ChatComponent.module.css'

interface ChatComposerCompletionBudgetProps {
  value: number
  exactValue?: unknown
  onChange: (value: number) => void
}

export default function ChatComposerCompletionBudget({
  value,
  exactValue,
  onChange,
}: ChatComposerCompletionBudgetProps) {
  const hasExactBudget = exactValue !== undefined
  const effectiveValue = hasExactBudget ? String(exactValue) : String(value)
  const options = [...new Set(['2048', '8192', '16384', effectiveValue])]

  return (
    <label
      className={styles.completionBudget}
      title={
        hasExactBudget
          ? 'This probe preserves its explicit output budget, including reasoning tokens.'
          : 'Maximum output tokens per model call, including reasoning tokens.'
      }
    >
      <span>Output budget</span>
      <select
        aria-label="Output budget (tokens, including reasoning)"
        value={effectiveValue}
        disabled={hasExactBudget}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {Number(option).toLocaleString('en-US')}
          </option>
        ))}
      </select>
    </label>
  )
}
