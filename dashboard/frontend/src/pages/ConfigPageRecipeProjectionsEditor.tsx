import { useEffect, useState } from 'react'

import type { ConfigProjections } from './configPageSupport'
import styles from './ConfigPageEntrypointsRecipesSection.module.css'
import dialogStyles from './ConfigPageRecipeDialog.module.css'

interface Props {
  value?: ConfigProjections
  catalog?: ConfigProjections
  onChange: (value: ConfigProjections) => void
}

const families = ['partitions', 'scores', 'mappings'] as const

export default function ConfigPageRecipeProjectionsEditor({
  value = {},
  catalog = {},
  onChange,
}: Props) {
  const [selected, setSelected] = useState('')
  const [raw, setRaw] = useState(() => JSON.stringify(value, null, 2))
  const [error, setError] = useState('')
  useEffect(() => setRaw(JSON.stringify(value, null, 2)), [value])
  const available = families.flatMap((family) =>
    (catalog[family] ?? []).map((item) => ({ family, item })),
  )
  const add = () => {
    const [family, name] = selected.split(':', 2) as [(typeof families)[number], string]
    const source = available.find(
      (candidate) => candidate.family === family && candidate.item.name === name,
    )
    if (!source) return
    const current = value[family] ?? []
    if (!current.some((item) => item.name === name)) {
      onChange({ ...value, [family]: [...current, JSON.parse(JSON.stringify(source.item))] })
    }
    setSelected('')
  }
  const commitRaw = () => {
    try {
      onChange(JSON.parse(raw) as ConfigProjections)
      setError('')
    } catch {
      setError('Projection configuration must be valid JSON.')
    }
  }
  return (
    <div className={styles.decisionEditor}>
      <p className={styles.editorHint}>
        Reuse a projection from your library or configure partitions, scores, and mappings directly.
      </p>
      {available.length ? (
        <div className={styles.modelPoolHeader}>
          <label>
            <span>Reuse projection</span>
            <select value={selected} onChange={(event) => setSelected(event.target.value)}>
              <option value="">Choose from your library</option>
              {available.map(({ family, item }) => (
                <option key={`${family}:${item.name}`} value={`${family}:${item.name}`}>
                  {item.name} · {family}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className={styles.secondaryButton}
            disabled={!selected}
            onClick={add}
          >
            Add to recipe
          </button>
        </div>
      ) : null}
      <div className={styles.modelPool}>
        {families.flatMap((family) =>
          (value[family] ?? []).map((item, index) => (
            <article className={styles.modelReferenceCard} key={`${family}:${item.name}`}>
              <div>
                <strong>{item.name}</strong>
                <p className={styles.editorHint}>{family}</p>
              </div>
              <button
                type="button"
                className={styles.iconDangerButton}
                aria-label={`Remove ${item.name}`}
                onClick={() =>
                  onChange({
                    ...value,
                    [family]: (value[family] ?? []).filter(
                      (_, currentIndex) => currentIndex !== index,
                    ),
                  })
                }
              >
                ×
              </button>
            </article>
          )),
        )}
      </div>
      <details>
        <summary>Advanced projection configuration</summary>
        {error ? (
          <p className={styles.editorHint} role="alert">
            {error}
          </p>
        ) : null}
        <textarea
          className={dialogStyles.projectionJSON}
          value={raw}
          onChange={(event) => setRaw(event.target.value)}
          onBlur={commitRaw}
          rows={16}
          spellCheck={false}
          aria-label="Projection configuration JSON"
        />
      </details>
    </div>
  )
}
