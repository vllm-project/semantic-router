import React, { useMemo, useState } from 'react'

import openAPIDocument from '../../../static/openapi/apiserver/apiserver.openapi.json'
import SchemaView, { exampleValue, OpenAPISchema } from './SchemaView'
import styles from './styles.module.css'

type HTTPMethod = 'get' | 'post' | 'patch' | 'put' | 'delete'

interface OpenAPIMedia {
  schema?: OpenAPISchema
}

interface OpenAPIOperationPolicy {
  'x-vllm-sr-permission'?: string
  'x-vllm-sr-sensitivity'?: string
  'x-vllm-sr-audit-action'?: string
  'x-vllm-sr-plane'?: string
  'x-vllm-sr-audiences'?: string[]
  'x-vllm-sr-stability'?: string
  'x-vllm-sr-visibility'?: string
  'x-vllm-sr-plugin-operations'?: Array<{ plugin: string, mode: string }>
}

interface OpenAPIOperation extends OpenAPIOperationPolicy {
  summary?: string
  description?: string
  operationId?: string
  tags?: string[]
  security?: Array<Record<string, string[]>>
  parameters?: Array<{
    name: string
    in: string
    description?: string
    required?: boolean
    schema?: OpenAPISchema
  }>
  requestBody?: {
    description?: string
    required?: boolean
    content?: Record<string, OpenAPIMedia>
  }
  responses?: Record<string, {
    description?: string
    headers?: Record<string, { description?: string, schema?: OpenAPISchema }>
    content?: Record<string, OpenAPIMedia>
  }>
}

interface OpenAPISpec {
  openapi: string
  info: { title: string, description?: string, version: string }
  paths: Record<string, Partial<Record<HTTPMethod, OpenAPIOperation>>>
  components?: { schemas?: Record<string, OpenAPISchema> }
}

interface OperationEntry {
  id: string
  method: HTTPMethod
  path: string
  operation: OpenAPIOperation
}

const document = openAPIDocument as unknown as OpenAPISpec
const definitions = document.components?.schemas ?? {}
const methods: HTTPMethod[] = ['get', 'post', 'patch', 'put', 'delete']
const operations: OperationEntry[] = Object.entries(document.paths)
  .flatMap(([path, item]) => methods.flatMap((method) => {
    const operation = item[method]
    return operation ? [{ id: `${method}:${path}`, method, path, operation }] : []
  }))
  .sort((left, right) => left.path.localeCompare(right.path) || left.method.localeCompare(right.method))

function requestSchema(operation: OpenAPIOperation): OpenAPISchema | undefined {
  const media = Object.values(operation.requestBody?.content ?? {})[0]
  return media?.schema
}

function MediaSchemas({ content }: { content?: Record<string, OpenAPIMedia> }) {
  return (
    <div className={styles.mediaSchemas}>
      {Object.entries(content ?? {}).map(([mediaType, media]) => (
        <div key={mediaType}>
          <code className={styles.mediaType}>{mediaType}</code>
          <SchemaView name="body" schema={media.schema} definitions={definitions} />
        </div>
      ))}
    </div>
  )
}

function curlExample(entry: OperationEntry): string {
  const path = entry.path.replace(/\{([^}]+)\}/g, '<$1>')
  const requiredQuery = (entry.operation.parameters ?? [])
    .filter(parameter => parameter.in === 'query' && parameter.required)
    .map(parameter => `${encodeURIComponent(parameter.name)}=<${parameter.name}>`)
  const requestURL = requiredQuery.length ? `${path}?${requiredQuery.join('&')}` : path
  const parts = [
    `curl -sS -X ${entry.method.toUpperCase()}`,
    `  'http://localhost:8080${requestURL}'`,
  ]
  for (const parameter of entry.operation.parameters ?? []) {
    if (parameter.in === 'header' && parameter.required)
      parts.push(`  -H '${parameter.name}: <${parameter.name}>'`)
  }
  if (entry.operation.requestBody) {
    const mediaType = Object.keys(entry.operation.requestBody.content ?? {})[0] ?? 'application/json'
    if (mediaType === 'application/json') {
      const body = JSON.stringify(exampleValue(requestSchema(entry.operation), definitions), null, 2)
        .replace(/\u0027/g, '\u0027\\\u0027\u0027')
      parts.push(`  -H 'Content-Type: application/json'`)
      parts.push(`  -d '${body}'`)
    }
    else if (mediaType === 'multipart/form-data') {
      parts.push(`  -F 'file=@<path>'`)
      parts.push(`  -F 'purpose=<purpose>'`)
    }
  }
  return parts.join(' \\\n')
}

export default function OpenAPIReference() {
  const [query, setQuery] = useState('')
  const [methodFilter, setMethodFilter] = useState<'all' | HTTPMethod>('all')
  const [selectedID, setSelectedID] = useState(
    operations.find(entry => entry.path === '/api/v1' && entry.method === 'get')?.id
    ?? operations[0]?.id
    ?? '',
  )
  const normalized = query.trim().toLowerCase()
  const visible = useMemo(
    () => operations.filter((entry) => {
      if (methodFilter !== 'all' && entry.method !== methodFilter) return false
      return [
        entry.path, entry.method, entry.operation.summary, entry.operation.operationId,
        ...(entry.operation.tags ?? []),
        ...(entry.operation['x-vllm-sr-plugin-operations'] ?? []).flatMap(({ plugin, mode }) => [plugin, mode]),
      ]
        .filter(Boolean)
        .some(value => value!.toLowerCase().includes(normalized))
    }),
    [methodFilter, normalized],
  )
  const selected = visible.find(entry => entry.id === selectedID) ?? visible[0] ?? null

  return (
    <div className={styles.reference}>
      <header className={styles.identity}>
        <div>
          <span>OpenAPI</span>
          <strong>{document.openapi}</strong>
        </div>
        <div>
          <span>API version</span>
          <strong>{document.info.version}</strong>
        </div>
        <div>
          <span>Operations</span>
          <strong>{operations.length}</strong>
        </div>
        <a href="/openapi/apiserver/apiserver.openapi.json" target="_blank" rel="noreferrer">
          View generated JSON ↗
        </a>
      </header>

      <div className={styles.filters}>
        <label>
          <span>Find an operation</span>
          <input
            type="search"
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder="Try config, diagnostics, rag…"
          />
        </label>
        <label>
          <span>Method</span>
          <select
            value={methodFilter}
            onChange={event => setMethodFilter(event.target.value as 'all' | HTTPMethod)}
          >
            <option value="all">All methods</option>
            {methods.map(method => <option key={method} value={method}>{method.toUpperCase()}</option>)}
          </select>
        </label>
      </div>

      <div className={styles.layout}>
        <nav className={styles.directory} aria-label="Router API operations">
          <header>
            <h3>Operations</h3>
            <span>{visible.length}</span>
          </header>
          {visible.map(entry => (
            <button
              type="button"
              key={entry.id}
              data-active={entry.id === selected?.id || undefined}
              aria-pressed={entry.id === selected?.id}
              onClick={() => setSelectedID(entry.id)}
            >
              <b data-method={entry.method}>{entry.method.toUpperCase()}</b>
              <span>
                <code>{entry.path}</code>
                <small>{entry.operation.summary}</small>
              </span>
            </button>
          ))}
        </nav>

        <article key={selected?.id} className={styles.detail}>
          {selected
            ? (
                <>
                  <header className={styles.detailHeading}>
                    <b data-method={selected.method}>{selected.method.toUpperCase()}</b>
                    <div>
                      <code>{selected.path}</code>
                      <h3>{selected.operation.summary}</h3>
                    </div>
                  </header>
                  {selected.operation.operationId
                    ? (
                        <p className={styles.operationID}>
                          Operation ID ·
                          <code>{selected.operation.operationId}</code>
                        </p>
                      )
                    : null}

                  <div className={styles.accessContract}>
                    <div>
                      <span>Authentication</span>
                      <code>{selected.operation.security?.length ? 'runtime-configured bearer' : 'public'}</code>
                    </div>
                    <div>
                      <span>Permission</span>
                      <code>{selected.operation['x-vllm-sr-permission'] ?? 'unspecified'}</code>
                    </div>
                    <div>
                      <span>Sensitivity</span>
                      <code>{selected.operation['x-vllm-sr-sensitivity'] ?? 'unspecified'}</code>
                    </div>
                    {[
                      ['Capability', selected.operation.tags?.join(', ')],
                      ['Plane', selected.operation['x-vllm-sr-plane']],
                      ['Audiences', selected.operation['x-vllm-sr-audiences']?.join(', ')],
                      ['Stability', selected.operation['x-vllm-sr-stability']],
                      ['Visibility', selected.operation['x-vllm-sr-visibility']],
                    ].filter(([, value]) => value).map(([label, value]) => (
                      <div key={label}>
                        <span>{label}</span>
                        <code>{value}</code>
                      </div>
                    ))}
                    {selected.operation['x-vllm-sr-audit-action']
                      ? (
                          <div>
                            <span>Audit action</span>
                            <code>{selected.operation['x-vllm-sr-audit-action']}</code>
                          </div>
                        )
                      : null}
                  </div>

                  {selected.operation['x-vllm-sr-plugin-operations']?.length
                    ? (
                        <section className={styles.block}>
                          <h4>Plugin operations</h4>
                          <div className={styles.pluginOperations}>
                            {selected.operation['x-vllm-sr-plugin-operations'].map(({ plugin, mode }) => (
                              <div key={`${plugin}:${mode}`}>
                                <code>{plugin}</code>
                                <span>{mode}</span>
                              </div>
                            ))}
                          </div>
                        </section>
                      )
                    : null}

                  <section className={styles.block}>
                    <h4>Request</h4>
                    {selected.operation.parameters?.length
                      ? (
                          <div className={styles.parameters}>
                            {selected.operation.parameters.map(parameter => (
                              <div key={`${parameter.in}:${parameter.name}`}>
                                <header>
                                  <code>{parameter.name}</code>
                                  <span>{parameter.in}</span>
                                  {parameter.required ? <b>required</b> : null}
                                </header>
                                <SchemaView name="value" schema={parameter.schema} definitions={definitions} />
                                {parameter.description ? <p>{parameter.description}</p> : null}
                              </div>
                            ))}
                          </div>
                        )
                      : <p className={styles.muted}>No path, query, or header parameters.</p>}
                    {selected.operation.requestBody
                      ? (
                          <>
                            <div className={styles.bodyContract}>
                              <strong>
                                Body
                                {selected.operation.requestBody.required ? ' · required' : ''}
                              </strong>
                              {selected.operation.requestBody.description
                                ? <small>{selected.operation.requestBody.description}</small>
                                : null}
                            </div>
                            <MediaSchemas content={selected.operation.requestBody.content} />
                          </>
                        )
                      : null}
                    <pre><code>{curlExample(selected)}</code></pre>
                  </section>

                  <section className={styles.block}>
                    <h4>Responses</h4>
                    <div className={styles.responses}>
                      {Object.entries(selected.operation.responses ?? {}).map(([status, response]) => (
                        <details key={status} open={status.startsWith('2')}>
                          <summary>
                            <strong>{status}</strong>
                            <span>{response.description}</span>
                          </summary>
                          <div className={styles.responseDetails}>
                            {Object.keys(response.headers ?? {}).length
                              ? (
                                  <div className={styles.responseHeaders}>
                                    <strong>Headers</strong>
                                    {Object.entries(response.headers ?? {}).map(([name, header]) => (
                                      <SchemaView key={name} name={name} schema={{ ...header.schema, description: header.description }} definitions={definitions} />
                                    ))}
                                  </div>
                                )
                              : null}
                            {Object.keys(response.content ?? {}).length
                              ? <MediaSchemas content={response.content} />
                              : <p className={styles.muted}>No response body.</p>}
                          </div>
                        </details>
                      ))}
                    </div>
                  </section>
                </>
              )
            : (
                <div className={styles.empty}>
                  <strong>No operation matches this filter</strong>
                  <span>Clear the search or choose another method.</span>
                </div>
              )}
        </article>
      </div>
    </div>
  )
}
