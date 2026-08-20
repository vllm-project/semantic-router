import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import type { ConfigSignals } from './configPageSupport'
import styles from './ConfigPageEntrypointsRecipesSection.module.css'
import { RECIPE_SIGNAL_FAMILIES, type RecipeSignalFamily } from './recipeSignalCatalog'

interface ConfigPageRecipePolicyEditorProps {
  value: ConfigSignals
  catalog?: ConfigSignals
  onChange: (value: ConfigSignals) => void
}

interface SignalEntry extends RecipeSignalFamily {
  name: string
  value: unknown
}

function entriesOf(signals?: ConfigSignals): SignalEntry[] {
  const record = (signals ?? {}) as unknown as Record<string, unknown>
  return RECIPE_SIGNAL_FAMILIES.flatMap((family) => {
    const values = record[family.key]
    if (!Array.isArray(values)) return []
    return values.flatMap((value) => {
      if (!value || typeof value !== 'object') return []
      const name = (value as { name?: unknown }).name
      return typeof name === 'string' && name.trim()
        ? [{ ...family, name: name.trim(), value }]
        : []
    })
  })
}

function signalKey(signal: Pick<SignalEntry, 'key' | 'name'>) {
  return `${signal.key}:${signal.name}`
}

export default function ConfigPageRecipePolicyEditor({
  value,
  catalog,
  onChange,
}: ConfigPageRecipePolicyEditorProps) {
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState('all')
  const selected = useMemo(() => entriesOf(value), [value])
  const selectedKeys = useMemo(() => new Set(selected.map(signalKey)), [selected])
  const available = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return entriesOf(catalog).filter(
      (entry) =>
        !selectedKeys.has(signalKey(entry)) &&
        (family === 'all' || entry.key === family) &&
        (!normalized ||
          entry.name.toLowerCase().includes(normalized) ||
          entry.type.toLowerCase().includes(normalized)),
    )
  }, [catalog, family, query, selectedKeys])

  const add = (entry: SignalEntry) => {
    const record = value as unknown as Record<string, unknown>
    const current = Array.isArray(record[entry.key]) ? (record[entry.key] as unknown[]) : []
    onChange({
      ...record,
      [entry.key]: [...current, structuredClone(entry.value)],
    } as unknown as ConfigSignals)
  }

  const remove = (entry: SignalEntry) => {
    const record = value as unknown as Record<string, unknown>
    const current = Array.isArray(record[entry.key]) ? (record[entry.key] as unknown[]) : []
    onChange({
      ...record,
      [entry.key]: current.filter(
        (item) =>
          !item || typeof item !== 'object' || (item as { name?: unknown }).name !== entry.name,
      ),
    } as unknown as ConfigSignals)
  }

  const catalogCount = entriesOf(catalog).length

  return (
    <div className={styles.signalComposer}>
      <header className={styles.signalComposerHeader}>
        <div>
          <span>Signal library</span>
          <h3>Choose what this recipe can see.</h3>
          <p>{RECIPE_SIGNAL_FAMILIES.length} signal types · one reusable catalog.</p>
        </div>
        <Link to="/config/signals">Manage signals</Link>
      </header>

      <section className={styles.selectedSignals} aria-labelledby="selected-signals-title">
        <div className={styles.signalSectionHeading}>
          <strong id="selected-signals-title">Selected</strong>
          <span>{selected.length}</span>
        </div>
        {selected.length ? (
          <div className={styles.signalSelectionGrid}>
            {selected.map((entry) => (
              <article key={signalKey(entry)} className={styles.signalSelectionCard}>
                <span>{entry.type}</span>
                <strong>{entry.name}</strong>
                <button
                  type="button"
                  onClick={() => remove(entry)}
                  aria-label={`Remove ${entry.name}`}
                >
                  Remove
                </button>
              </article>
            ))}
          </div>
        ) : (
          <p className={styles.signalEmpty}>No signals selected yet.</p>
        )}
      </section>

      <section className={styles.signalCatalog} aria-labelledby="signal-catalog-title">
        <div className={styles.signalSectionHeading}>
          <strong id="signal-catalog-title">Available</strong>
          <span>{catalogCount}</span>
        </div>
        <div className={styles.signalFilters}>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search signals"
            aria-label="Search signal library"
          />
          <select
            value={family}
            onChange={(event) => setFamily(event.target.value)}
            aria-label="Signal type"
          >
            <option value="all">All 20 types</option>
            {RECIPE_SIGNAL_FAMILIES.map((item) => (
              <option value={item.key} key={item.key}>
                {item.type}
              </option>
            ))}
          </select>
        </div>
        {available.length ? (
          <div className={styles.signalLibraryList}>
            {available.map((entry) => (
              <button type="button" key={signalKey(entry)} onClick={() => add(entry)}>
                <span>
                  <strong>{entry.name}</strong>
                  <small>{entry.type}</small>
                </span>
                <b aria-hidden="true">＋</b>
              </button>
            ))}
          </div>
        ) : (
          <div className={styles.signalEmptyState}>
            <strong>{catalogCount ? 'No matching signals' : 'Your signal library is empty'}</strong>
            <p>
              {catalogCount
                ? 'Try another name or type.'
                : 'Create a signal once, then reuse it in any recipe.'}
            </p>
            {!catalogCount ? <Link to="/config/signals">Create signal</Link> : null}
          </div>
        )}
      </section>
    </div>
  )
}
