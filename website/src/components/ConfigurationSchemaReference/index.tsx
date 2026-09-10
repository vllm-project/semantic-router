import React, { useMemo, useState } from 'react'

import routerConfigSchema from '../../../../src/semantic-router/pkg/configschema/router-config-v0.3.schema.json'
import styles from './styles.module.css'

interface SchemaNode {
  $ref?: string
  type?: string
  title?: string
  description?: string
  properties?: Record<string, SchemaNode>
  required?: string[]
  oneOf?: SchemaNode[]
  anyOf?: SchemaNode[]
  items?: SchemaNode
  enum?: unknown[]
  const?: unknown
}

interface SchemaDocument extends SchemaNode {
  '$id': string
  '$defs': Record<string, SchemaNode>
  'x-vllm-sr': {
    contract_version: string
    config_version: string
    signals: Array<{ type: string, display_name: string, schema_ref: string }>
    algorithms: Array<{
      type: string
      display_name: string
      schema_ref?: string
    }>
    plugins: Array<{ type: string, display_name: string, schema_ref: string }>
    projections: Array<{
      collection: string
      display_name: string
      schema_ref: string
    }>
  }
}

const document = routerConfigSchema as unknown as SchemaDocument

function resolve(node: SchemaNode): SchemaNode {
  if (!node.$ref?.startsWith('#/$defs/')) return node
  return document.$defs[node.$ref.slice('#/$defs/'.length)] ?? node
}

function concrete(node: SchemaNode): SchemaNode {
  const resolved = resolve(node)
  const alternatives = resolved.oneOf ?? resolved.anyOf
  if (!alternatives) return resolved
  return resolve(
    alternatives.find(candidate => candidate.type !== 'null') ?? resolved,
  )
}

function nodeType(node: SchemaNode): string {
  const value = concrete(node)
  if (value.type === 'array')
    return `array<${value.items ? nodeType(value.items) : 'value'}>`
  if (value.type) return value.type
  if (value.const !== undefined) return typeof value.const
  return 'value'
}

function displayName(value: string): string {
  const words = value.replaceAll('_', ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

const catalogs = [
  {
    kind: 'Signals',
    entries: document['x-vllm-sr'].signals.map(entry => ({
      name: entry.type,
      label: entry.display_name,
      ref: entry.schema_ref,
    })),
  },
  {
    kind: 'Algorithms',
    entries: document['x-vllm-sr'].algorithms.map(entry => ({
      name: entry.type,
      label: entry.display_name,
      ref: entry.schema_ref,
    })),
  },
  {
    kind: 'Plugins',
    entries: document['x-vllm-sr'].plugins.map(entry => ({
      name: entry.type,
      label: entry.display_name,
      ref: entry.schema_ref,
    })),
  },
  {
    kind: 'Projections',
    entries: document['x-vllm-sr'].projections.map(entry => ({
      name: entry.collection,
      label: entry.display_name,
      ref: entry.schema_ref,
    })),
  },
]

export default function ConfigurationSchemaReference() {
  const [query, setQuery] = useState('')
  const [selectedNode, setSelectedNode] = useState<SchemaNode | null>(null)
  const normalized = query.trim().toLowerCase()
  const sections = useMemo(
    () =>
      Object.entries(document.properties ?? {}).filter(([name, node]) => {
        const section = resolve(node)
        return [name, section.title, section.description]
          .filter(Boolean)
          .some(value => value!.toLowerCase().includes(normalized))
      }),
    [normalized],
  )
  const visibleCatalogs = useMemo(
    () =>
      catalogs
        .map(catalog => ({
          ...catalog,
          entries: catalog.entries.filter(entry =>
            [entry.name, entry.label, catalog.kind].some(value =>
              value.toLowerCase().includes(normalized),
            ),
          ),
        }))
        .filter(catalog => catalog.entries.length),
    [normalized],
  )
  const selected = selectedNode ? concrete(selectedNode) : null
  const required = new Set(selected?.required ?? [])

  return (
    <div className={styles.reference}>
      <header className={styles.identity}>
        <div>
          <span>Release contract</span>
          <strong>{document['x-vllm-sr'].config_version}</strong>
        </div>
        <div>
          <span>Contract</span>
          <strong>{document['x-vllm-sr'].contract_version}</strong>
        </div>
        <a
          href="https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/configschema/router-config-v0.3.schema.json"
          target="_blank"
          rel="noreferrer"
        >
          View canonical JSON ↗
        </a>
      </header>

      <label className={styles.search}>
        <span>Filter contract</span>
        <input
          type="search"
          value={query}
          onChange={event => setQuery(event.target.value)}
          placeholder="Try learning, weighted, jailbreak…"
        />
      </label>

      <div className={styles.layout}>
        <nav
          className={styles.directory}
          aria-label="Configuration schema directory"
        >
          <h3>Document sections</h3>
          {sections.map(([name, node]) => {
            const section = resolve(node)
            return (
              <button
                type="button"
                key={name}
                onClick={() => setSelectedNode(node)}
              >
                <span>{section.title || displayName(name)}</span>
                <code>{name}</code>
              </button>
            )
          })}
          {visibleCatalogs.map(catalog => (
            <section key={catalog.kind}>
              <h3>{catalog.kind}</h3>
              {catalog.entries.map(entry => (
                <button
                  type="button"
                  key={entry.name}
                  onClick={() =>
                    setSelectedNode(
                      entry.ref ? { $ref: entry.ref } : { type: 'object' },
                    )}
                >
                  <span>{entry.label}</span>
                  <code>{entry.name}</code>
                </button>
              ))}
            </section>
          ))}
        </nav>

        <article className={styles.detail}>
          {selected
            ? (
                <>
                  <h3>{selected.title || 'Schema fields'}</h3>
                  {selected.description ? <p>{selected.description}</p> : null}
                  <div className={styles.fields}>
                    {Object.entries(selected.properties ?? {}).map(
                      ([name, node]) => {
                        const field = concrete(node)
                        return (
                          <section key={name}>
                            <header>
                              <code>{name}</code>
                              <span>{nodeType(node)}</span>
                              {required.has(name) ? <b>required</b> : null}
                            </header>
                            {field.description ? <p>{field.description}</p> : null}
                            {field.enum?.length
                              ? (
                                  <small>
                                    Values:
                                    {field.enum.join(', ')}
                                  </small>
                                )
                              : null}
                          </section>
                        )
                      },
                    )}
                  </div>
                </>
              )
            : (
                <div className={styles.empty}>
                  <strong>Select a document section or routing surface</strong>
                  <span>
                    The reference expands one schema definition at a time.
                  </span>
                </div>
              )}
        </article>
      </div>
    </div>
  )
}
