import * as React from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowUpRight,
  BookMarked,
  Braces,
  ChevronRight,
  FileJson,
  Link2,
  RotateCw,
  Search,
  X,
} from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  CodeBlock,
  CopyButton,
  EmptyState,
  IconTile,
  PageContainer,
  PageHeader,
  Skeleton,
  Tooltip,
  cn,
  useCopy,
} from '../ui'
import { SettingsAside, SettingsSection, useInSettingsLayout } from './SettingsLayout'
import {
  configSchemaChildFields,
  configSchemaExample,
  configSchemaFields,
  configSchemaStructure,
  filterConfigSchemaIndex,
  focusedSchemaTitle,
  type ConfigSchemaField,
  type ConfigSchemaIndex,
  type ConfigSchemaNode,
} from './configSchemaReferenceSupport'

interface SchemaSource {
  source: 'runtime' | 'bundled' | 'unknown'
  match: 'true' | 'false' | 'unknown'
}

const ease = [0.16, 1, 0.3, 1] as const

function anchorId(path: string): string {
  return `schema-field-${path.replace(/[^\w-]+/g, '-')}`
}

function DirectoryButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? 'true' : undefined}
      className={cn(
        'flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left text-[13px] outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
        active
          ? 'bg-accent text-foreground'
          : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
      )}
    >
      {children}
    </button>
  )
}

function Directory({
  index,
  query,
  onQuery,
  selectionKind,
  selectionValue,
  onSection,
  onSurface,
}: {
  index: ConfigSchemaIndex
  query: string
  onQuery: (value: string) => void
  selectionKind: 'section' | 'surface' | null
  selectionValue: string | null
  onSection: (path: string) => void
  onSurface: (kind: string, name: string) => void
}) {
  const filtered = React.useMemo(() => filterConfigSchemaIndex(index, query), [index, query])
  const [closed, setClosed] = React.useState<Set<string>>(() => new Set())
  const searching = Boolean(query.trim())
  const surfaceKinds = Object.entries(filtered.surfaces)
  const empty = filtered.sections.length === 0 && surfaceKinds.length === 0

  return (
    <aside
      aria-label="Schema directory"
      className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xs lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)]"
    >
      <div className="shrink-0 border-b border-border p-2.5">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-subtle-foreground" />
          <input
            value={query}
            onChange={(event) => onQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Escape') onQuery('')
            }}
            placeholder="Search sections and surfaces"
            aria-label="Filter schema directory"
            className="h-8 w-full rounded-md border border-input bg-surface pl-8 pr-7 text-[12.5px] shadow-xs outline-none transition-[border-color,box-shadow] placeholder:text-subtle-foreground hover:border-border-strong focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
          />
          {query ? (
            <button
              type="button"
              aria-label="Clear search"
              onClick={() => onQuery('')}
              className="absolute right-1.5 top-1/2 grid size-5 -translate-y-1/2 place-items-center rounded text-subtle-foreground hover:text-foreground"
            >
              <X className="size-3" />
            </button>
          ) : null}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
        {filtered.sections.length ? (
          <div className="mb-2">
            <p className="px-2 pb-1 pt-1.5 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
              Config sections
            </p>
            {filtered.sections.map((section) => (
              <DirectoryButton
                key={section.path}
                active={
                  selectionKind === 'section' &&
                  (selectionValue === section.path ||
                    Boolean(selectionValue?.startsWith(`${section.path}.`)))
                }
                onClick={() => onSection(section.path)}
              >
                <span className="min-w-0 flex-1 truncate">{section.title}</span>
                <code className="shrink-0 font-mono text-[11px] text-subtle-foreground">
                  {section.path}
                </code>
              </DirectoryButton>
            ))}
          </div>
        ) : null}
        {surfaceKinds.map(([kind, catalog]) => {
          const open = searching || !closed.has(kind)
          const regionId = `schema-surfaces-${kind}`
          return (
            <div key={kind} className="mb-1">
              <button
                type="button"
                aria-expanded={open}
                aria-controls={regionId}
                onClick={() =>
                  setClosed((current) => {
                    const next = new Set(current)
                    if (next.has(kind)) next.delete(kind)
                    else next.add(kind)
                    return next
                  })
                }
                className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
              >
                <ChevronRight className={cn('size-3 transition-transform', open && 'rotate-90')} />
                <span className="flex-1">{kind}s</span>
                <span className="font-mono tabular normal-case tracking-normal">
                  {catalog.names.length}
                </span>
              </button>
              {open ? (
                <div id={regionId} className="ml-3 border-l border-border pl-1">
                  {catalog.names.map((name) => (
                    <DirectoryButton
                      key={name}
                      active={selectionKind === 'surface' && selectionValue === `${kind}:${name}`}
                      onClick={() => onSurface(kind, name)}
                    >
                      <span className="min-w-0 flex-1 truncate font-mono text-[12px]">{name}</span>
                    </DirectoryButton>
                  ))}
                </div>
              ) : null}
            </div>
          )
        })}
        {empty ? (
          <p className="px-3 py-8 text-center text-xs text-subtle-foreground">
            Nothing matches “{query}”.
          </p>
        ) : null}
      </div>
    </aside>
  )
}

function FieldRow({
  document,
  field,
  path,
  depth,
  onOpenSection,
}: {
  document: ConfigSchemaNode
  field: ConfigSchemaField
  path: string
  depth: number
  onOpenSection: (path: string) => void
}) {
  const children = React.useMemo(
    () => (depth < 5 ? configSchemaChildFields(document, field.node, depth) : []),
    [depth, document, field.node],
  )
  const [open, setOpen] = React.useState(false)
  const regionId = `${anchorId(path)}-children`
  return (
    <li id={anchorId(path)} className="scroll-mt-6">
      <div className="group flex flex-col gap-1.5 px-4 py-3">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {children.length ? (
            <button
              type="button"
              aria-expanded={open}
              aria-controls={regionId}
              aria-label={`${open ? 'Hide' : 'Show'} the fields of ${field.name}`}
              onClick={() => setOpen((current) => !current)}
              className="-ml-1 grid size-5 place-items-center rounded text-subtle-foreground outline-none transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
            >
              <ChevronRight className={cn('size-3.5 transition-transform', open && 'rotate-90')} />
            </button>
          ) : null}
          {field.path ? (
            <button
              type="button"
              onClick={() => onOpenSection(field.path!)}
              className="rounded font-mono text-[13px] font-medium text-foreground underline-offset-4 outline-none hover:text-primary hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/40"
            >
              {field.name}
            </button>
          ) : (
            <code className="font-mono text-[13px] font-medium text-foreground">{field.name}</code>
          )}
          <span className="font-mono text-xs text-chart-1">{field.type}</span>
          {field.required ? (
            <Badge tone="warning" size="sm">
              required
            </Badge>
          ) : null}
          {children.length ? (
            <span className="text-xs text-subtle-foreground">
              {children.length} {children.length === 1 ? 'field' : 'fields'}
            </span>
          ) : null}
          <span className="ml-auto flex items-center gap-0.5">
            <span className="opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
              <CopyButton value={path} label={`Copy ${path}`} />
            </span>
            {field.path ? (
              <Tooltip content={`Open ${field.path}`}>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Open ${field.path}`}
                  onClick={() => onOpenSection(field.path!)}
                >
                  <ChevronRight />
                </Button>
              </Tooltip>
            ) : null}
          </span>
        </div>
        {field.description ? (
          <p className="max-w-3xl text-[13px] leading-relaxed text-muted-foreground">
            {field.description}
          </p>
        ) : null}
        {field.details.length ? (
          <div className="flex flex-wrap gap-1.5">
            {field.details.map((value) => (
              <code
                key={value}
                className="rounded-md border border-border bg-surface-subtle px-1.5 py-0.5 font-mono text-[11.5px] text-foreground/80"
              >
                {value}
              </code>
            ))}
          </div>
        ) : null}
      </div>
      {open && children.length ? (
        <motion.ul
          id={regionId}
          initial={{ opacity: 0, y: -2 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, ease }}
          className="mb-2 ml-6 divide-y divide-border rounded-lg border border-border bg-surface-subtle/40"
        >
          {children.map((child) => (
            <FieldRow
              key={child.name}
              document={document}
              field={child}
              path={`${path}.${child.name}`}
              depth={depth + 1}
              onOpenSection={onOpenSection}
            />
          ))}
        </motion.ul>
      ) : null}
    </li>
  )
}

function OnThisPage({
  fields,
  basePath,
  hasExample,
}: {
  fields: ConfigSchemaField[]
  basePath: string
  hasExample: boolean
}) {
  const go = (id: string) =>
    window.document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  const linkClass =
    'block w-full truncate rounded-r-md py-1 pl-3 pr-1.5 text-left text-[12.5px] text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40'
  return (
    <nav aria-label="On this page" className="flex flex-col gap-2">
      <p className="px-2.5 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
        On this page
      </p>
      <ul className="ml-2.5 flex flex-col border-l border-border">
        <li>
          <button type="button" className={linkClass} onClick={() => go('schema-overview')}>
            Overview
          </button>
        </li>
        {fields.slice(0, 24).map((field) => (
          <li key={field.name}>
            <button
              type="button"
              className={cn(linkClass, 'font-mono text-[12px]')}
              onClick={() => go(anchorId(`${basePath}.${field.name}`))}
            >
              {field.name}
            </button>
          </li>
        ))}
        {fields.length > 24 ? (
          <li className="py-1 pl-3 text-xs text-subtle-foreground">+{fields.length - 24} more</li>
        ) : null}
        {hasExample ? (
          <li>
            <button type="button" className={linkClass} onClick={() => go('schema-example')}>
              Example
            </button>
          </li>
        ) : null}
      </ul>
    </nav>
  )
}

function Detail({
  detail,
  loading,
  selectionKind,
  selectionValue,
  onOpenSection,
}: {
  detail: ConfigSchemaNode | null
  loading: boolean
  selectionKind: 'section' | 'surface' | null
  selectionValue: string | null
  onOpenSection: (path: string) => void
}) {
  const { copy, copied } = useCopy()
  const fields = React.useMemo(() => (detail ? configSchemaFields(detail) : []), [detail])
  const structure = React.useMemo(() => (detail ? configSchemaStructure(detail) : null), [detail])
  const example = React.useMemo(() => (detail ? configSchemaExample(detail) : ''), [detail])
  const raw = React.useMemo(() => (detail ? JSON.stringify(detail, null, 2) : ''), [detail])
  const basePath =
    selectionKind === 'section'
      ? (selectionValue ?? '')
      : (selectionValue ?? '').split(':').slice(1).join(':')

  if (!selectionKind) {
    return (
      <div className="rounded-xl border border-border bg-card shadow-xs">
        <EmptyState
          icon={Braces}
          tone="slate"
          title="Pick a section or a routing surface"
          description="Only that branch of the schema loads, with its fields, types, defaults and a YAML example."
        />
      </div>
    )
  }
  if (loading || !detail) {
    return (
      <div
        className="flex flex-col gap-3 rounded-xl border border-border bg-card p-5 shadow-xs"
        aria-label="Loading the schema branch"
      >
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-6 w-56" />
        <Skeleton className="h-3 w-80" />
        <div className="mt-3 flex flex-col gap-3">
          {[0, 1, 2, 3].map((row) => (
            <Skeleton key={row} className="h-12" />
          ))}
        </div>
      </div>
    )
  }
  const view = detail['x-vllm-sr-view']
  const description = detail['x-vllm-sr-surface']?.description || detail.description
  return (
    <motion.article
      key={selectionValue}
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease }}
      aria-live="polite"
      className="flex min-w-0 flex-col gap-4"
    >
      <SettingsAside>
        <OnThisPage fields={fields} basePath={basePath} hasExample={Boolean(example)} />
      </SettingsAside>
      <section
        id="schema-overview"
        className="scroll-mt-6 rounded-xl border border-border bg-card shadow-xs"
      >
        <header className="flex flex-col gap-2 border-b border-border px-5 py-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={view?.view === 'section' ? 'brand' : 'violet'} size="sm">
              {view?.view === 'surface' ? `${view.kind} surface` : 'config section'}
            </Badge>
            {selectionKind === 'section' ? (
              <nav
                aria-label="Schema path"
                className="flex min-w-0 items-center gap-0.5 font-mono text-xs"
              >
                {basePath.split('.').map((segment, position, all) => {
                  const target = all.slice(0, position + 1).join('.')
                  const last = position === all.length - 1
                  return (
                    <React.Fragment key={target}>
                      {position ? <ChevronRight className="size-3 text-subtle-foreground" /> : null}
                      {last ? (
                        <span className="text-foreground/80">{segment}</span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onOpenSection(target)}
                          className="rounded text-subtle-foreground outline-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
                        >
                          {segment}
                        </button>
                      )}
                    </React.Fragment>
                  )
                })}
              </nav>
            ) : (
              <code className="font-mono text-xs text-subtle-foreground">{selectionValue}</code>
            )}
            <CopyButton
              value={selectionKind === 'section' ? basePath : (selectionValue ?? '')}
              label="Copy the schema path"
            />
            <Tooltip content={copied ? 'Link copied' : 'Copy a link to this view'}>
              <Button
                variant="ghost"
                size="icon-xs"
                className="ml-auto"
                aria-label="Copy a link to this view"
                onClick={() => copy(window.location.href)}
              >
                <Link2 />
              </Button>
            </Tooltip>
          </div>
          <h3 className="text-lg font-semibold tracking-[-0.015em]">
            {focusedSchemaTitle(detail)}
          </h3>
          {description ? (
            <p className="max-w-3xl text-[13px] leading-relaxed text-muted-foreground">
              {description}
            </p>
          ) : null}
          {structure ? (
            <p className="text-xs text-muted-foreground">
              <span className="font-mono text-chart-1">{structure.type}</span> ·{' '}
              {structure.description}
            </p>
          ) : null}
        </header>
        {fields.length ? (
          <>
            <div className="flex items-center gap-2 border-b border-border px-5 py-2.5">
              <span className="text-[13px] font-semibold">
                {structure?.fieldsLabel ?? 'Fields'}
              </span>
              <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
                {fields.length}
              </span>
              <span className="ml-auto text-xs text-subtle-foreground">
                {fields.some((field) => field.path)
                  ? 'Open a field to see its own fields'
                  : 'Expand a field to walk its nested fields'}
              </span>
            </div>
            <ul className="divide-y divide-border">
              {fields.map((field) => (
                <FieldRow
                  key={field.name}
                  document={detail}
                  field={field}
                  path={basePath ? `${basePath}.${field.name}` : field.name}
                  depth={0}
                  onOpenSection={onOpenSection}
                />
              ))}
            </ul>
          </>
        ) : (
          <p className="px-5 py-6 text-[13px] text-muted-foreground">
            This object has no configurable parameters.
          </p>
        )}
      </section>
      <section id="schema-example" className="scroll-mt-6">
        <CodeBlock
          title="Example"
          lineNumbers
          tabs={[
            { label: 'YAML', language: 'yaml', code: example || '# no fields' },
            { label: 'JSON Schema', language: 'json', code: raw },
          ]}
        />
      </section>
    </motion.article>
  )
}

/** The configuration contract the Router accepts, branch by branch, with types, defaults and examples. */
export default function ConfigSchemaReferencePage() {
  const inSettings = useInSettingsLayout()
  const [searchParams, setSearchParams] = useSearchParams()
  const [index, setIndex] = React.useState<ConfigSchemaIndex | null>(null)
  const [detail, setDetail] = React.useState<ConfigSchemaNode | null>(null)
  const [source, setSource] = React.useState<SchemaSource>({ source: 'unknown', match: 'unknown' })
  const [query, setQuery] = React.useState('')
  const [error, setError] = React.useState<string | null>(null)
  const [indexLoading, setIndexLoading] = React.useState(true)
  const [detailLoading, setDetailLoading] = React.useState(false)
  const [attempt, setAttempt] = React.useState(0)
  const selectedSection = searchParams.get('section')
  const selectedSurface = searchParams.get('surface')
  const selectionKind = selectedSection ? 'section' : selectedSurface ? 'surface' : null
  const selectionValue = selectedSection || selectedSurface

  React.useEffect(() => {
    const controller = new AbortController()
    const loadIndex = async () => {
      setIndexLoading(true)
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
        if (!controller.signal.aborted) setIndexLoading(false)
      }
    }
    void loadIndex()
    return () => controller.abort()
  }, [attempt])

  React.useEffect(() => {
    if (!selectionKind || !selectionValue) {
      setDetail(null)
      return
    }
    const controller = new AbortController()
    const loadDetail = async () => {
      setDetailLoading(true)
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
        if (!controller.signal.aborted) setDetailLoading(false)
      }
    }
    void loadDetail()
    return () => controller.abort()
  }, [selectionKind, selectionValue])

  const selectSection = (path: string) => setSearchParams({ section: path })
  const selectSurface = (kind: string, name: string) =>
    setSearchParams({ surface: `${kind}:${name}` })

  const sourceBadge =
    source.source === 'runtime' ? (
      <Badge tone="success" dot>
        Deployed Router
      </Badge>
    ) : source.source === 'bundled' ? (
      <Badge tone="warning" dot>
        Dashboard fallback
      </Badge>
    ) : (
      <Badge tone="neutral" dot>
        Unknown source
      </Badge>
    )
  const actions = (
    <>
      {index ? sourceBadge : null}
      <Button asChild variant="outline" size="sm">
        <a href="/api/router/config/schema?view=full" target="_blank" rel="noreferrer">
          <FileJson />
          Full JSON Schema
          <ArrowUpRight />
        </a>
      </Button>
    </>
  )

  const body = (
    <div data-testid="config-schema-reference-page" className="flex min-w-0 flex-col gap-4">
      {source.source === 'runtime' && source.match === 'false' ? (
        <Alert
          tone="warning"
          title="The Router runs a different contract than this Dashboard build"
        >
          This reference shows the deployed Router&apos;s contract. Upgrade the Dashboard to edit
          fields the Router added.
        </Alert>
      ) : null}
      {source.source === 'bundled' ? (
        <Alert tone="warning" title="The Router schema endpoint is unavailable">
          This reference uses the contract bundled with the Dashboard. Start the Router to see the
          contract it actually accepts.
        </Alert>
      ) : null}
      {error ? (
        <div role="alert">
          <Alert
            tone="danger"
            title="The schema could not be loaded"
            actions={
              index ? null : (
                <Button
                  variant="outline"
                  size="xs"
                  onClick={() => setAttempt((value) => value + 1)}
                >
                  <RotateCw />
                  Retry
                </Button>
              )
            }
          >
            {error}
          </Alert>
        </div>
      ) : null}
      {index ? (
        <dl className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-border bg-card px-4 py-3 text-xs shadow-xs">
          <div className="flex items-center gap-2">
            <dt className="text-subtle-foreground">Config</dt>
            <dd className="font-mono font-medium">{index.config_version}</dd>
          </div>
          <div className="flex items-center gap-2">
            <dt className="text-subtle-foreground">Contract</dt>
            <dd className="font-mono font-medium">{index.contract_version}</dd>
          </div>
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <dt className="shrink-0 text-subtle-foreground">Schema ID</dt>
            <dd className="flex min-w-0 items-center gap-1">
              <span className="truncate font-mono">{index.schema_id}</span>
              <CopyButton value={index.schema_id} label="Copy the schema ID" />
            </dd>
          </div>
        </dl>
      ) : null}
      {indexLoading && !index ? (
        <div
          className="grid gap-4 lg:grid-cols-[minmax(0,17rem)_minmax(0,1fr)]"
          aria-label="Loading the schema"
        >
          <Skeleton className="h-[420px] rounded-xl" />
          <Skeleton className="h-[420px] rounded-xl" />
        </div>
      ) : index ? (
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
          <Directory
            index={index}
            query={query}
            onQuery={setQuery}
            selectionKind={selectionKind}
            selectionValue={selectionValue}
            onSection={selectSection}
            onSurface={selectSurface}
          />
          <Detail
            detail={detail}
            loading={detailLoading}
            selectionKind={selectionKind}
            selectionValue={selectionValue}
            onOpenSection={selectSection}
          />
        </div>
      ) : !error ? (
        <EmptyState
          icon={BookMarked}
          tone="slate"
          title="No schema to show"
          description="The Router returned an empty contract."
        />
      ) : null}
    </div>
  )

  if (inSettings) {
    return (
      <SettingsSection
        title="Schema reference"
        description="The configuration contract this Router accepts: every section and routing surface, with types, defaults and examples."
        actions={actions}
      >
        {body}
      </SettingsSection>
    )
  }
  return (
    <PageContainer>
      <PageHeader
        icon={<IconTile icon={BookMarked} tone="slate" size="lg" />}
        title="Schema reference"
        description="The configuration contract this Router accepts, with types, defaults and examples."
        actions={actions}
      />
      {body}
    </PageContainer>
  )
}
