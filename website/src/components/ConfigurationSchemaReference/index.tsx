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

type SchemaFieldEntry = [name: string, node: SchemaNode]

interface DirectoryEntry {
  id: string
  name: string
  label: string
  node: SchemaNode
}

interface DirectoryGroup {
  kind: string
  entries: DirectoryEntry[]
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
  const words = value.replace(/_/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function fieldRoot(node: SchemaNode): SchemaNode {
  const selected = concrete(node)
  return selected.type === 'array' && selected.items ? concrete(selected.items) : selected
}

function fieldEntries(node: SchemaNode): SchemaFieldEntry[] {
  const selected = concrete(node)
  const root = fieldRoot(node)
  const properties = Object.entries(root.properties ?? {})
  if (properties.length || root.type === 'object') return properties
  return [[selected.type === 'array' ? 'item' : 'value', root]]
}

function fieldsLabel(node: SchemaNode): string {
  const selected = concrete(node)
  if (selected.type === 'array') return 'Item fields'
  if (fieldRoot(node).type === 'object') return 'Fields'
  return 'Value'
}

const directoryGroups: DirectoryGroup[] = [
  {
    kind: 'Document sections',
    entries: Object.entries(document.properties ?? {}).map(([name, node]) => {
      const section = resolve(node)
      return {
        id: `section:${name}`,
        name,
        label: section.title || displayName(name),
        node,
      }
    }),
  },
  {
    kind: 'Signals',
    entries: document['x-vllm-sr'].signals.map(entry => ({
      id: `signal:${entry.type}`,
      name: entry.type,
      label: entry.display_name,
      node: { $ref: entry.schema_ref },
    })),
  },
  {
    kind: 'Algorithms',
    entries: document['x-vllm-sr'].algorithms.map(entry => ({
      id: `algorithm:${entry.type}`,
      name: entry.type,
      label: entry.display_name,
      node: entry.schema_ref ? { $ref: entry.schema_ref } : { type: 'object' },
    })),
  },
  {
    kind: 'Plugins',
    entries: document['x-vllm-sr'].plugins.map(entry => ({
      id: `plugin:${entry.type}`,
      name: entry.type,
      label: entry.display_name,
      node: { $ref: entry.schema_ref },
    })),
  },
  {
    kind: 'Projections',
    entries: document['x-vllm-sr'].projections.map(entry => ({
      id: `projection:${entry.collection}`,
      name: entry.collection,
      label: entry.display_name,
      node: { $ref: entry.schema_ref },
    })),
  },
]

export default function ConfigurationSchemaReference() {
  const [query, setQuery] = useState('')
  const [selectedEntryId, setSelectedEntryId] = useState(
    directoryGroups[0]?.entries.find(entry => entry.name === 'global')?.id
    ?? directoryGroups[0]?.entries[0]?.id
    ?? '',
  )
  const normalized = query.trim().toLowerCase()
  const visibleGroups = useMemo(
    () =>
      directoryGroups
        .map(catalog => ({
          ...catalog,
          entries: catalog.entries.filter((entry) => {
            const node = concrete(entry.node)
            return [entry.name, entry.label, catalog.kind, node.title, node.description]
              .filter(Boolean)
              .some(value => value!.toLowerCase().includes(normalized))
          }),
        }))
        .filter(catalog => catalog.entries.length),
    [normalized],
  )
  const visibleEntries = visibleGroups.flatMap(group => group.entries)
  const selectedEntry = visibleEntries.find(entry => entry.id === selectedEntryId)
    ?? visibleEntries[0]
    ?? null
  const selectedNode = selectedEntry?.node ?? null
  const selected = selectedNode ? concrete(selectedNode) : null
  const selectedRoot = selectedNode ? fieldRoot(selectedNode) : null
  const selectedFields = selectedNode ? fieldEntries(selectedNode) : []
  const required = new Set(selectedRoot?.required ?? [])

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
          {visibleGroups.map(catalog => (
            <section key={catalog.kind}>
              <h3>{catalog.kind}</h3>
              {catalog.entries.map(entry => (
                <button
                  type="button"
                  key={entry.id}
                  data-active={entry.id === selectedEntry?.id || undefined}
                  aria-pressed={entry.id === selectedEntry?.id}
                  onClick={() => setSelectedEntryId(entry.id)}
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
                  <header className={styles.detailHeading}>
                    <div>
                      <span>Selected contract</span>
                      <h3>
                        {selected.title || selectedRoot?.title || selectedEntry?.label || 'Schema fields'}
                      </h3>
                    </div>
                    <code>{selectedEntry?.name}</code>
                  </header>
                  {selected.description || selectedRoot?.description
                    ? <p>{selected.description || selectedRoot?.description}</p>
                    : null}
                  <div className={styles.shape}>
                    <span>Shape</span>
                    <code>{nodeType(selected)}</code>
                    <small>
                      {selected.type === 'array'
                        ? 'Each list item uses the fields below.'
                        : selectedRoot?.type === 'object'
                          ? 'This object accepts the fields below.'
                          : 'This section is a single value.'}
                    </small>
                  </div>
                  <div className={styles.fieldsHeading}>
                    <h4>{fieldsLabel(selected)}</h4>
                    <span>{selectedFields.length}</span>
                  </div>
                  {selectedFields.length
                    ? (
                        <div className={styles.fields}>
                          {selectedFields.map(([name, node]) => {
                            const field = concrete(node)
                            const isSynthetic = name === 'value' || name === 'item'
                            return (
                              <section key={name}>
                                <header>
                                  <code>{name}</code>
                                  <span>{nodeType(node)}</span>
                                  {required.has(name) || isSynthetic ? <b>required</b> : null}
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
                                {field.const !== undefined
                                  ? (
                                      <small>
                                        Fixed:
                                        {String(field.const)}
                                      </small>
                                    )
                                  : null}
                              </section>
                            )
                          })}
                        </div>
                      )
                    : <div className={styles.emptyFields}>No configurable fields.</div>}
                </>
              )
            : (
                <div className={styles.empty}>
                  <strong>No contract matches this filter</strong>
                  <span>
                    Clear the search or try a broader capability name.
                  </span>
                </div>
              )}
        </article>
      </div>
    </div>
  )
}
