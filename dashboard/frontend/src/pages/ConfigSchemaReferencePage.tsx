import React, { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import {
  configSchemaFields,
  configSchemaStructure,
  filterConfigSchemaIndex,
  focusedSchemaTitle,
  type ConfigSchemaIndex,
  type ConfigSchemaNode,
} from './configSchemaReferenceSupport'
import styles from './ConfigSchemaReferencePage.module.css'

interface SchemaSource {
  source: 'runtime' | 'bundled' | 'unknown'
  match: 'true' | 'false' | 'unknown'
}

const ConfigSchemaReferencePage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams()
  const [index, setIndex] = useState<ConfigSchemaIndex | null>(null)
  const [detail, setDetail] = useState<ConfigSchemaNode | null>(null)
  const [source, setSource] = useState<SchemaSource>({ source: 'unknown', match: 'unknown' })
  const [query, setQuery] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const selectedSection = searchParams.get('section')
  const selectedSurface = searchParams.get('surface')
  const selectionKind = selectedSection ? 'section' : selectedSurface ? 'surface' : null
  const selectionValue = selectedSection || selectedSurface

  useEffect(() => {
    const controller = new AbortController()
    const loadIndex = async () => {
      try {
        const response = await fetch('/api/router/config/schema?view=index', {
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`Schema index request failed (${response.status}).`)
        setIndex((await response.json()) as ConfigSchemaIndex)
        setSource({
          source:
            (response.headers.get('X-Vllm-Sr-Schema-Source') as SchemaSource['source']) ||
            'unknown',
          match:
            (response.headers.get('X-Vllm-Sr-Schema-Match') as SchemaSource['match']) || 'unknown',
        })
        setError(null)
      } catch (cause) {
        if ((cause as Error).name !== 'AbortError') {
          setError(cause instanceof Error ? cause.message : 'Schema index is unavailable.')
        }
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }
    void loadIndex()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!selectionKind || !selectionValue) {
      setDetail(null)
      return
    }
    const controller = new AbortController()
    const loadDetail = async () => {
      setIsLoading(true)
      try {
        const params = new URLSearchParams({ view: selectionKind })
        if (selectionKind === 'section') {
          params.set('path', selectionValue)
          params.set('expanded', 'true')
        } else {
          const [kind, ...nameParts] = selectionValue.split(':')
          params.set('kind', kind)
          params.set('name', nameParts.join(':'))
        }
        const response = await fetch(`/api/router/config/schema?${params}`, {
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`Schema detail request failed (${response.status}).`)
        setDetail((await response.json()) as ConfigSchemaNode)
        setError(null)
      } catch (cause) {
        if ((cause as Error).name !== 'AbortError') {
          setError(cause instanceof Error ? cause.message : 'Schema detail is unavailable.')
        }
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }
    void loadDetail()
    return () => controller.abort()
  }, [selectionKind, selectionValue])

  const filtered = useMemo(
    () => (index ? filterConfigSchemaIndex(index, query) : null),
    [index, query],
  )
  const fields = useMemo(() => (detail ? configSchemaFields(detail) : []), [detail])
  const structure = useMemo(() => (detail ? configSchemaStructure(detail) : null), [detail])
  const selectSection = (path: string) => setSearchParams({ section: path })
  const selectSurface = (kind: string, name: string) =>
    setSearchParams({ surface: `${kind}:${name}` })

  if (isLoading && !index) return <ProductLoadingState label="Loading configuration schema" />

  return (
    <main className={styles.page} data-testid="config-schema-reference-page">
      <header className={styles.masthead}>
        <div>
          <span className={styles.eyebrow}>Configuration contract</span>
          <h1>Schema reference</h1>
          <p>Browse the fields and routing surfaces supported by this deployed Router.</p>
        </div>
        <div className={styles.mastheadActions}>
          <span
            className={`${styles.sourceBadge} ${source.source === 'runtime' ? styles.live : ''}`}
          >
            <i />
            {source.source === 'runtime'
              ? 'Deployed Router'
              : source.source === 'bundled'
                ? 'Dashboard fallback'
                : 'Unknown source'}
          </span>
          <a
            className={styles.rawLink}
            href="/api/router/config/schema?view=full"
            target="_blank"
            rel="noreferrer"
          >
            Full JSON Schema <ProductIcon name="arrow-right" />
          </a>
        </div>
      </header>

      {source.source === 'runtime' && source.match === 'false' ? (
        <div className={styles.warning} role="status">
          <ProductIcon name="alert" />
          The deployed Router contract differs from this Dashboard build. This reference shows the
          runtime contract.
        </div>
      ) : null}
      {source.source === 'bundled' ? (
        <div className={styles.warning} role="status">
          <ProductIcon name="alert" />
          The Router schema endpoint is unavailable. This reference is using the Dashboard build
          fallback.
        </div>
      ) : null}
      {error ? (
        <div className={styles.error} role="alert">
          {error}
        </div>
      ) : null}

      {index ? (
        <section className={styles.identityStrip} aria-label="Schema identity">
          <div>
            <span>Config</span>
            <strong>{index.config_version}</strong>
          </div>
          <div>
            <span>Contract</span>
            <strong>{index.contract_version}</strong>
          </div>
          <div className={styles.schemaId}>
            <span>Schema ID</span>
            <strong>{index.schema_id}</strong>
          </div>
        </section>
      ) : null}

      <section className={styles.workspace}>
        <aside className={styles.directory} aria-label="Schema directory">
          <label className={styles.search}>
            <ProductIcon name="search" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter sections and surfaces"
              aria-label="Filter schema directory"
            />
          </label>

          <div className={styles.directoryGroup}>
            <h2>Config sections</h2>
            {filtered?.sections.map((section) => (
              <button
                type="button"
                key={section.path}
                className={
                  selectionKind === 'section' && selectionValue === section.path
                    ? styles.selected
                    : ''
                }
                onClick={() => selectSection(section.path)}
              >
                <span>{section.title}</span>
                <code>{section.path}</code>
              </button>
            ))}
          </div>

          {Object.entries(filtered?.surfaces ?? {}).map(([kind, catalog]) => (
            <div className={styles.directoryGroup} key={kind}>
              <h2>
                {kind}s <span>{catalog.names.length}</span>
              </h2>
              {catalog.names.map((name) => (
                <button
                  type="button"
                  key={`${kind}:${name}`}
                  className={
                    selectionKind === 'surface' && selectionValue === `${kind}:${name}`
                      ? styles.selected
                      : ''
                  }
                  onClick={() => selectSurface(kind, name)}
                >
                  <span>{name}</span>
                  <ProductIcon name="chevron-right" />
                </button>
              ))}
            </div>
          ))}
        </aside>

        <article className={styles.detail} aria-live="polite">
          {!selectionKind ? (
            <div className={styles.emptyState}>
              <ProductIcon name="code" />
              <h2>Select a contract surface</h2>
              <p>
                Choose a config section or routing surface. Only that schema branch will be loaded.
              </p>
            </div>
          ) : isLoading ? (
            <ProductLoadingState label="Loading schema branch" />
          ) : detail ? (
            <>
              <header className={styles.detailHeader}>
                <span>{detail['x-vllm-sr-view']?.view}</span>
                <h2>{focusedSchemaTitle(detail)}</h2>
                {detail['x-vllm-sr-surface']?.description || detail.description ? (
                  <p>{detail['x-vllm-sr-surface']?.description || detail.description}</p>
                ) : null}
              </header>
              {structure ? (
                <div className={styles.structure}>
                  <div>
                    <span>Shape</span>
                    <strong>{structure.type}</strong>
                  </div>
                  <p>{structure.description}</p>
                </div>
              ) : null}
              {fields.length ? (
                <>
                  <div className={styles.fieldsHeading}>
                    <h3>{structure?.fieldsLabel || 'Fields'}</h3>
                    <span>{fields.length}</span>
                  </div>
                  <div className={styles.fields}>
                    {fields.map((field) => (
                      <section className={styles.field} key={field.name}>
                        <div className={styles.fieldHeading}>
                          <code>{field.name}</code>
                          <span>{field.type}</span>
                          {field.required ? <b>required</b> : null}
                        </div>
                        {field.description ? <p>{field.description}</p> : null}
                        {field.details.length ? (
                          <div className={styles.fieldDetails}>
                            {field.details.map((value) => (
                              <code key={value}>{value}</code>
                            ))}
                          </div>
                        ) : null}
                      </section>
                    ))}
                  </div>
                </>
              ) : (
                <div className={styles.emptyFields}>
                  This object has no configurable parameters.
                </div>
              )}
              <details className={styles.rawDetail}>
                <summary>View raw schema</summary>
                <pre>{JSON.stringify(detail, null, 2)}</pre>
              </details>
            </>
          ) : null}
        </article>
      </section>
    </main>
  )
}

export default ConfigSchemaReferencePage
