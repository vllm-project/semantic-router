import { useId, useMemo, useRef, useState, type KeyboardEvent, type WheelEvent } from 'react'

import styles from './ConfigPageRecipePicker.module.css'

export interface RecipePickerOption {
  description?: string
  meta: string
  name: string
}

interface Props {
  onChange: (name: string) => void
  options: RecipePickerOption[]
  value: string
}

export default function ConfigPageRecipePicker({ onChange, options, value }: Props) {
  const listId = useId()
  const [query, setQuery] = useState('')
  const [highlighted, setHighlighted] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return options
    return options.filter(
      (option) =>
        option.name.toLowerCase().includes(normalized) ||
        option.description?.toLowerCase().includes(normalized),
    )
  }, [options, query])

  const choose = (name: string) => {
    onChange(name)
    setQuery('')
  }

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (filtered.length === 0) return
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      const delta = event.key === 'ArrowDown' ? 1 : -1
      setHighlighted((current) => Math.max(0, Math.min(current + delta, filtered.length - 1)))
    } else if (event.key === 'Enter') {
      event.preventDefault()
      choose(filtered[Math.min(highlighted, filtered.length - 1)].name)
    }
  }

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    const list = listRef.current
    if (!list || list.scrollHeight <= list.clientHeight) return
    const previous = list.scrollTop
    list.scrollTop = Math.max(
      0,
      Math.min(list.scrollHeight - list.clientHeight, previous + event.deltaY),
    )
    if (list.scrollTop !== previous) event.preventDefault()
  }

  return (
    <div className={styles.root} onWheelCapture={onWheel}>
      <div className={styles.searchRow}>
        <input
          type="search"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value)
            setHighlighted(0)
          }}
          onKeyDown={onKeyDown}
          className={styles.search}
          placeholder="Find a ready Recipe"
          aria-label="Find a ready Recipe"
          aria-controls={listId}
        />
        <span className={styles.count}>{options.length} ready</span>
      </div>
      <div
        id={listId}
        ref={listRef}
        className={styles.list}
        role="listbox"
        aria-label="Ready Recipes"
      >
        {filtered.map((option, index) => {
          const selected = option.name === value
          return (
            <button
              key={option.name}
              type="button"
              role="option"
              aria-selected={selected}
              className={selected ? styles.optionSelected : styles.option}
              onClick={() => choose(option.name)}
              onMouseEnter={() => setHighlighted(index)}
            >
              <span className={styles.optionCopy}>
                <strong title={option.name}>{option.name}</strong>
                <span>{option.description || 'Ready to publish'}</span>
              </span>
              <span className={styles.meta}>{option.meta}</span>
            </button>
          )
        })}
        {filtered.length === 0 ? (
          <div className={styles.empty}>
            {options.length === 0
              ? 'Finish a Recipe before publishing a model.'
              : 'No Recipes match this search.'}
          </div>
        ) : null}
      </div>
    </div>
  )
}
