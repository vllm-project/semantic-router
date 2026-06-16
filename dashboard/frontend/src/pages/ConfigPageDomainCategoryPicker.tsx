import { useMemo, useState } from 'react'
import styles from './ConfigPageDomainCategoryPicker.module.css'

const DEFAULT_MMLU_DOMAIN_CATEGORIES = [
  'biology',
  'business',
  'chemistry',
  'computer science',
  'economics',
  'engineering',
  'history',
  'law',
  'math',
  'philosophy',
  'physics',
  'psychology',
  'health',
  'other',
]

interface ConfigPageDomainCategoryPickerProps {
  value: string[]
  onChange: (value: string[]) => void
}

function normalizeCategory(value: string): string {
  return value.trim().replace(/\s+/g, ' ')
}

function dedupeCategories(values: string[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []

  for (const value of values) {
    const normalized = normalizeCategory(value)
    const key = normalized.toLowerCase()
    if (!normalized || seen.has(key)) {
      continue
    }
    seen.add(key)
    result.push(normalized)
  }

  return result
}

export default function ConfigPageDomainCategoryPicker({
  value,
  onChange,
}: ConfigPageDomainCategoryPickerProps) {
  const [customCategory, setCustomCategory] = useState('')
  const categories = useMemo(() => dedupeCategories(value), [value])
  const selectedKeys = useMemo(
    () => new Set(categories.map((category) => category.toLowerCase())),
    [categories],
  )

  const updateCategories = (nextCategories: string[]) => {
    onChange(dedupeCategories(nextCategories))
  }

  const togglePresetCategory = (category: string) => {
    const key = category.toLowerCase()
    if (selectedKeys.has(key)) {
      updateCategories(categories.filter((item) => item.toLowerCase() !== key))
      return
    }
    updateCategories([...categories, category])
  }

  const addCustomCategory = () => {
    const normalized = normalizeCategory(customCategory)
    if (!normalized || selectedKeys.has(normalized.toLowerCase())) {
      return
    }
    updateCategories([...categories, normalized])
    setCustomCategory('')
  }

  const removeCategory = (category: string) => {
    const key = category.toLowerCase()
    updateCategories(categories.filter((item) => item.toLowerCase() !== key))
  }

  return (
    <div className={styles.picker}>
      <div className={styles.optionGrid}>
        {DEFAULT_MMLU_DOMAIN_CATEGORIES.map((category) => {
          const selected = selectedKeys.has(category.toLowerCase())
          return (
            <label
              key={category}
              className={`${styles.option} ${selected ? styles.optionSelected : ''}`}
            >
              <input
                type="checkbox"
                checked={selected}
                onChange={() => togglePresetCategory(category)}
              />
              <span>{category}</span>
            </label>
          )
        })}
      </div>

      <div className={styles.customRow}>
        <input
          className={styles.customInput}
          value={customCategory}
          onChange={(event) => setCustomCategory(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault()
              addCustomCategory()
            }
          }}
          placeholder="Custom domain"
        />
        <button
          type="button"
          className={styles.addButton}
          onClick={addCustomCategory}
          disabled={!customCategory.trim()}
        >
          + Add
        </button>
      </div>

      {categories.length > 0 && (
        <div className={styles.selectedList}>
          {categories.map((category) => (
            <span key={category.toLowerCase()} className={styles.selectedTag}>
              <span className={styles.selectedTagText}>{category}</span>
              <button
                type="button"
                className={styles.removeButton}
                onClick={() => removeCategory(category)}
                aria-label={`Remove ${category}`}
              >
                x
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
