import React, { useMemo, useState } from 'react'

import type { CatalogProtocol, CatalogProvider } from '../../data/modelHubCatalogTypes'
import { providerProtocolOperations } from '../../data/modelHubProviderOperations'
import { CatalogMark } from './ModelHubMark'
import { EmptyState, Pagination, readable, srOnlyClass } from './ModelHubPrimitives'
import styles from './modelHubProviders.module.css'

const PAGE_SIZE = 10

function ProviderCard({
  provider,
  protocols,
}: {
  provider: CatalogProvider
  protocols: Map<string, CatalogProtocol>
}) {
  const operations = provider.protocols.flatMap((protocolID) => {
    const protocol = protocols.get(protocolID)
    return protocol
      ? providerProtocolOperations(provider, protocol).map(operation => ({
          ...operation,
          protocol: protocol.display_name,
        }))
      : []
  })
  const builtInModelCount = provider.models?.length ?? 0

  return (
    <article className={styles.providerCard}>
      <div className={styles.identity}>
        <header>
          <CatalogMark presentation={provider.presentation} />
          <span>
            <strong>{provider.display_name}</strong>
            <code>{provider.id}</code>
          </span>
        </header>
        <div className={styles.providerStatus}>
          <span data-status={provider.conformance.status}>
            <i aria-hidden="true" />
            {readable(provider.conformance.status)}
            {provider.conformance.verified_at ? ` · ${provider.conformance.verified_at}` : ''}
          </span>
          <span>{readable(provider.support_tier)}</span>
        </div>
      </div>
      <div className={styles.summary}>
        <p>{provider.description}</p>
        {provider.default_base_url ? <code title={provider.default_base_url}>{provider.default_base_url}</code> : null}
        <dl className={styles.providerFacts}>
          <div>
            <dt>Models</dt>
            <dd>
              {builtInModelCount
                ? `${builtInModelCount} built-in ${builtInModelCount === 1 ? 'choice' : 'choices'}`
                : 'Enter a model ID'}
            </dd>
          </div>
          <div>
            <dt>Protocols</dt>
            <dd>{provider.protocols.length}</dd>
          </div>
          <div>
            <dt>Auth</dt>
            <dd>{readable(provider.auth.strategy)}</dd>
          </div>
        </dl>
      </div>
      <div className={styles.apiSurface}>
        <header>
          <strong>API surface</strong>
          <span>
            {operations.length}
            {' '}
            {operations.length === 1 ? 'operation' : 'operations'}
          </span>
        </header>
        {operations.length
          ? (
              <div className={styles.operationViewport}>
                <table className={styles.operationTable}>
                  <thead>
                    <tr>
                      <th>Protocol</th>
                      <th>Method</th>
                      <th>Operation</th>
                      <th>Path</th>
                    </tr>
                  </thead>
                  <tbody>
                    {operations.map(operation => (
                      <tr key={operation.reference}>
                        <td>{operation.protocol}</td>
                        <td><i>{operation.method}</i></td>
                        <td>{readable(operation.id)}</td>
                        <td><code title={operation.path}>{operation.path}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          : (
              <span className={styles.noOperations}>No published operations</span>
            )}
      </div>
    </article>
  )
}

export function ModelHubProviders({
  providers,
  protocols,
}: {
  providers: CatalogProvider[]
  protocols: CatalogProtocol[]
}) {
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(1)
  const protocolMap = useMemo(() => new Map(protocols.map(item => [item.id, item])), [protocols])
  const filtered = providers.filter((provider) => {
    const haystack = `${provider.display_name} ${provider.id} ${provider.description} ${provider.protocols.join(' ')}`.toLocaleLowerCase()
    return !query.trim() || haystack.includes(query.trim().toLocaleLowerCase())
  })
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const pageProviders = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  return (
    <div className={styles.providers}>
      <p className={styles.providerGuide}>
        Every provider below can be connected to Semantic Router. Choose a built-in model when one
        is listed, or enter any model ID that the provider serves.
      </p>
      <div className={styles.toolbar}>
        <label>
          <span className={srOnlyClass}>Search providers</span>
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <circle cx="8.5" cy="8.5" r="5.5" />
            <path d="m12.5 12.5 4 4" />
          </svg>
          <input
            type="search"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value)
              setPage(1)
            }}
            placeholder="Search providers"
          />
        </label>
      </div>
      <div className={styles.resultCount}>
        {filtered.length}
        {' '}
        providers
      </div>
      <div className={styles.providerGrid}>
        {pageProviders.map(provider => <ProviderCard key={provider.id} provider={provider} protocols={protocolMap} />)}
      </div>
      {!filtered.length ? <EmptyState title="No providers found" body="Try another search." /> : null}
      <Pagination page={page} pageCount={pageCount} total={filtered.length} pageSize={PAGE_SIZE} label="providers" onChange={setPage} />
    </div>
  )
}
