import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CheckCheck,
  Eye,
  FlaskConical,
  Info,
  MessagesSquare,
  Pencil,
  Play,
  RefreshCw,
  ShieldCheck,
  Split,
} from 'lucide-react'

import type {
  RecipeProbeDetail,
  RecipeProbePage,
  RecipeProbeRunPlan,
  RecipeProbeSummary,
  RecipeProbeValidationResult,
} from '../types/recipe'
import {
  Alert,
  Badge,
  Button,
  Card,
  DataTable,
  DataToolbar,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  IconTile,
  SectionLabel,
  Skeleton,
  Tooltip,
  cn,
  useRetainedValue,
  type Column,
} from '../ui'
import {
  createRecipeProbeRunPlan,
  getRecipeProbe,
  listRecipeProbes,
  validateRecipeProbe,
} from '../utils/recipeApi'
import {
  ExpectedRoute,
  RecipeProbeDetailPanel,
  RecipeProbeValidation,
  ValidationBadge,
} from './ConfigPageMoMProbeResults'
import ConfigPageTablePager from './ConfigPageTablePager'
import {
  mapRecipeProbeFacetOptions,
  RECIPE_PROBE_PAGE_SIZE,
  validationOutcome,
} from './configPageMoMProbesSupport'

interface ConfigPageMoMProbesPanelProps {
  onLaunch: (intent: 'run' | 'edit', plan: RecipeProbeRunPlan) => void
  recipeRevision: number
  activeId: string | null
  onOpen: (id: string) => void
  onClose: () => void
  /** Reports the probe total and failures for the page tab. */
  onSummary?: (summary: { total: number; failing: number } | null) => void
  onOpenRecipes?: () => void
}

type FacetKey = 'recipe' | 'decision' | 'tag' | 'model' | 'shape'

const FACETS: Array<{ key: FacetKey; label: string; facet: keyof RecipeProbePage['facets'] }> = [
  { key: 'recipe', label: 'Recipe', facet: 'recipes' },
  { key: 'decision', label: 'Decision', facet: 'decisions' },
  { key: 'tag', label: 'Tag', facet: 'tags' },
  { key: 'model', label: 'Model', facet: 'models' },
  { key: 'shape', label: 'Shape', facet: 'shapes' },
]

const emptyFacets: Record<FacetKey, string> = {
  recipe: '',
  decision: '',
  tag: '',
  model: '',
  shape: '',
}

/** The API filters by one value per facet: the newest choice replaces the last. */
const singleChoice = (current: string, next: string[]) =>
  next.length === 0 ? '' : (next.find((value) => value !== current) ?? next[next.length - 1])

export default function ConfigPageMoMProbesPanel({
  onLaunch,
  recipeRevision,
  activeId,
  onOpen,
  onClose,
  onSummary,
  onOpenRecipes,
}: ConfigPageMoMProbesPanelProps) {
  const [page, setPage] = useState(1)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [facets, setFacets] = useState(emptyFacets)
  const [result, setResult] = useState<RecipeProbePage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [detail, setDetail] = useState<RecipeProbeDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [validation, setValidation] = useState<Record<string, RecipeProbeValidationResult>>({})
  const [activeAction, setActiveAction] = useState<string | null>(null)
  const [actionError, setActionError] = useState<Record<string, string>>({})
  const [sweep, setSweep] = useState<{ done: number; total: number } | null>(null)
  const actionControllerRef = useRef<AbortController | null>(null)

  useEffect(
    () => () => {
      actionControllerRef.current?.abort()
    },
    [],
  )

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const nextQuery = query.trim()
      if (nextQuery === debouncedQuery) return
      setPage(1)
      setDebouncedQuery(nextQuery)
    }, 250)
    return () => window.clearTimeout(timer)
  }, [debouncedQuery, query])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void listRecipeProbes(
      {
        page,
        pageSize: RECIPE_PROBE_PAGE_SIZE,
        query: debouncedQuery,
        recipe: facets.recipe,
        decision: facets.decision,
        tag: facets.tag,
        model: facets.model,
        requestShape: facets.shape,
      },
      controller.signal,
    )
      .then(setResult)
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setResult(null)
        setError(reason instanceof Error ? reason.message : 'Failed to load Recipe probes.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [debouncedQuery, facets, page, recipeRevision, refreshKey])

  const items = useMemo(() => result?.items ?? [], [result])
  const selected = items.find((probe) => probe.id === activeId) ?? null
  const shown = useRetainedValue(selected)

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      setDetailError(null)
      return
    }
    const controller = new AbortController()
    setDetail(null)
    setDetailLoading(true)
    setDetailError(null)
    void getRecipeProbe(selected.decision_id, selected.variant_id, controller.signal)
      .then(setDetail)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setDetailError(reason instanceof Error ? reason.message : 'Failed to load probe detail.')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false)
      })
    return () => controller.abort()
  }, [selected])

  useEffect(() => {
    setValidation({})
    setActionError({})
  }, [recipeRevision, result?.recipe_digest])

  const failing = Object.values(validation).filter(
    (entry) => validationOutcome(entry) !== 'pass',
  ).length
  useEffect(() => {
    onSummary?.(result ? { total: result.total, failing } : null)
  }, [failing, onSummary, result])

  const hasFilters = Boolean(debouncedQuery || Object.values(facets).some(Boolean))
  const clearFilters = () => {
    setQuery('')
    setDebouncedQuery('')
    setFacets(emptyFacets)
    setPage(1)
  }

  const beginAction = (key: string) => {
    const controller = new AbortController()
    actionControllerRef.current?.abort()
    actionControllerRef.current = controller
    setActiveAction(key)
    return controller
  }
  const endAction = (controller: AbortController) => {
    if (actionControllerRef.current === controller) {
      actionControllerRef.current = null
      setActiveAction(null)
    }
  }

  const launchProbe = async (intent: 'run' | 'edit', probe: RecipeProbeSummary) => {
    if (!result) return
    const controller = beginAction(`${intent}:${probe.id}`)
    setActionError((current) => ({ ...current, [probe.id]: '' }))
    try {
      const plan = await createRecipeProbeRunPlan(
        probe.decision_id,
        probe.variant_id,
        result.recipe_digest,
        controller.signal,
      )
      if (controller.signal.aborted) return
      onLaunch(intent, plan)
    } catch (reason) {
      if (reason instanceof Error && reason.name === 'AbortError') return
      setActionError((current) => ({
        ...current,
        [probe.id]: reason instanceof Error ? reason.message : `Failed to ${intent} probe.`,
      }))
    } finally {
      endAction(controller)
    }
  }

  const validateOne = async (probe: RecipeProbeSummary, signal: AbortSignal, digest: string) => {
    setActionError((current) => ({ ...current, [probe.id]: '' }))
    setValidation((current) => {
      const next = { ...current }
      delete next[probe.id]
      return next
    })
    try {
      const next = await validateRecipeProbe(probe.decision_id, probe.variant_id, digest, signal)
      if (signal.aborted) return
      setValidation((current) => ({ ...current, [probe.id]: next }))
    } catch (reason) {
      if (reason instanceof Error && reason.name === 'AbortError') return
      setActionError((current) => ({
        ...current,
        [probe.id]: reason instanceof Error ? reason.message : 'Failed to validate probe.',
      }))
    }
  }

  const validateProbe = async (probe: RecipeProbeSummary) => {
    if (!result) return
    const controller = beginAction(`validate:${probe.id}`)
    try {
      await validateOne(probe, controller.signal, result.recipe_digest)
    } finally {
      endAction(controller)
    }
  }

  // Validation checks routing only, one probe at a time, without calling a backend.
  const validatePage = async () => {
    if (!result || items.length === 0) return
    const controller = beginAction('validate:page')
    setSweep({ done: 0, total: items.length })
    try {
      for (const [index, probe] of items.entries()) {
        if (controller.signal.aborted) return
        await validateOne(probe, controller.signal, result.recipe_digest)
        setSweep({ done: index + 1, total: items.length })
      }
    } finally {
      setSweep(null)
      endAction(controller)
    }
  }

  const columns = useMemo<Column<RecipeProbeSummary>[]>(
    () => [
      {
        id: 'probe',
        header: 'Probe',
        cell: (probe) => (
          <div className="flex min-w-0 max-w-[300px] flex-col gap-0.5">
            <span className="flex min-w-0 items-center gap-1.5">
              <span className="truncate font-mono text-[12.5px] font-medium">
                {probe.variant_id}
              </span>
              {!probe.playground.enabled ? (
                <Badge size="sm" tone="outline">
                  Eval only
                </Badge>
              ) : null}
            </span>
            <span className="truncate text-xs text-muted-foreground">
              {probe.display_prompt || probe.query_preview || 'Structured message fixture'}
            </span>
          </div>
        ),
      },
      {
        id: 'expected',
        header: 'Expected route',
        cell: (probe) => (
          <Badge
            tone="outline"
            className="max-w-[220px] font-mono"
            title={`${probe.expected.recipe || 'default'} · ${probe.expected.algorithm || probe.expected.alias || 'route only'}`}
          >
            <Split className="text-chart-3" />
            <span className="truncate">{probe.expected.decision}</span>
          </Badge>
        ),
      },
      {
        id: 'shape',
        header: 'Shape and tags',
        hideBelow: 'xl',
        cell: (probe) => {
          const chips = [
            ...probe.request_shapes.map((value) => ({ value, tone: 'info' as const })),
            ...probe.tags.map((value) => ({ value, tone: 'outline' as const })),
          ]
          return (
            <span
              className="flex max-w-[240px] items-center gap-1 overflow-hidden"
              title={chips.map((chip) => chip.value).join(', ')}
            >
              {chips.slice(0, 3).map((chip) => (
                <Badge key={`${chip.tone}-${chip.value}`} size="sm" tone={chip.tone}>
                  {chip.value}
                </Badge>
              ))}
              {chips.length > 3 ? (
                <span className="shrink-0 text-[11px] text-subtle-foreground">
                  +{chips.length - 3}
                </span>
              ) : null}
            </span>
          )
        },
      },
      {
        id: 'result',
        header: 'Validation',
        width: '150px',
        cell: (probe) =>
          activeAction === `validate:${probe.id}` ? (
            <Skeleton className="h-5 w-24" />
          ) : actionError[probe.id] ? (
            <Tooltip content={actionError[probe.id]}>
              <span className="inline-flex">
                <Badge tone="warning">Error</Badge>
              </span>
            </Tooltip>
          ) : (
            <ValidationBadge result={validation[probe.id]} />
          ),
      },
    ],
    [actionError, activeAction, validation],
  )

  if (error) {
    const unmanaged = /not a managed recipe/i.test(error)
    return (
      <Card className="py-2">
        <EmptyState
          icon={FlaskConical}
          tone="orange"
          title={unmanaged ? 'Probes come with a managed recipe' : 'Probes are unavailable'}
          description={
            unmanaged
              ? 'The running configuration is not a managed recipe package, so it ships no probes. Activate a recipe package on the Recipes tab to validate its probes.'
              : error
          }
          actions={
            unmanaged ? (
              onOpenRecipes ? (
                <Button onClick={onOpenRecipes}>Open recipes</Button>
              ) : undefined
            ) : (
              <Button variant="outline" onClick={() => setRefreshKey((current) => current + 1)}>
                <RefreshCw />
                Retry
              </Button>
            )
          }
        />
      </Card>
    )
  }

  const validatedOnPage = items.filter((probe) => validation[probe.id]).length
  const passedOnPage = items.filter(
    (probe) => validation[probe.id] && validationOutcome(validation[probe.id]) === 'pass',
  ).length
  const busy = activeAction !== null

  return (
    <div className="flex flex-col gap-4">
      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-3 px-5 pb-3 pt-4">
          <IconTile icon={FlaskConical} tone="orange" />
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
            <span className="text-sm font-semibold tabular">
              {sweep
                ? `Validating ${Math.min(sweep.done + 1, sweep.total)} of ${sweep.total}…`
                : validatedOnPage
                  ? `${passedOnPage} of ${validatedOnPage} probes pass`
                  : result
                    ? `${result.total.toLocaleString()} ${result.total === 1 ? 'probe' : 'probes'}`
                    : 'Loading probes…'}
            </span>
            {sweep ? null : validatedOnPage ? (
              <Badge tone={passedOnPage === validatedOnPage ? 'success' : 'danger'}>
                {passedOnPage === validatedOnPage
                  ? 'All routed as expected'
                  : `${validatedOnPage - passedOnPage} failing`}
              </Badge>
            ) : result ? (
              <Badge tone="neutral">Not validated yet</Badge>
            ) : null}
            <Tooltip
              content={
                <span className="flex max-w-72 flex-col gap-0.5 py-0.5 leading-snug">
                  <span>
                    Verified requests and the route each should take. Validate checks routing only;
                    Run sends the exact request in Playground.
                  </span>
                  {result?.recipe_digest ? (
                    <span className="font-mono opacity-70">
                      {result.recipe_digest.slice(0, 23)}…
                    </span>
                  ) : null}
                </span>
              }
            >
              <button
                type="button"
                aria-label="About probes"
                className="grid size-6 place-items-center rounded-md text-subtle-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <Info className="size-3.5" />
              </button>
            </Tooltip>
          </div>
          <Button
            onClick={() => void validatePage()}
            disabled={busy || items.length === 0}
            loading={Boolean(sweep)}
          >
            {sweep ? null : <ShieldCheck />}
            {sweep ? 'Validating…' : 'Validate this page'}
          </Button>
        </div>
        {items.length ? (
          <div className="px-5 pb-4" aria-hidden>
            <div className="flex h-1.5 gap-1">
              {items.map((probe) => {
                const entry = validation[probe.id]
                const outcome = entry ? validationOutcome(entry) : null
                return (
                  <span
                    key={probe.id}
                    className={cn(
                      'flex-1 rounded-full transition-colors duration-300',
                      activeAction === `validate:${probe.id}`
                        ? 'animate-pulse bg-info'
                        : outcome === 'pass'
                          ? 'bg-success'
                          : outcome === 'mismatch'
                            ? 'bg-destructive'
                            : outcome === 'unavailable'
                              ? 'bg-warning'
                              : 'bg-muted',
                    )}
                  />
                )
              })}
            </div>
          </div>
        ) : null}
      </Card>

      <div>
        <DataToolbar
          search={query}
          onSearchChange={setQuery}
          placeholder="Query, decision, variant or tag…"
          filters={
            <>
              {FACETS.map(({ key, label, facet }) => {
                const options = mapRecipeProbeFacetOptions(result?.facets[facet] ?? {}, facets[key])
                if (options.length === 0) return null
                return (
                  <FilterPill
                    key={key}
                    label={label}
                    selected={facets[key] ? [facets[key]] : []}
                    onChange={(next) => {
                      setPage(1)
                      setFacets((current) => ({
                        ...current,
                        [key]: singleChoice(current[key], next),
                      }))
                    }}
                    options={options.map(([value, count]) => ({ value, label: value, count }))}
                  />
                )
              })}
              {hasFilters ? (
                <Button variant="ghost" size="sm" onClick={clearFilters}>
                  Clear filters
                </Button>
              ) : null}
            </>
          }
          actions={
            <span className="text-xs text-muted-foreground">
              {result ? `${result.total.toLocaleString()} results` : ''}
            </span>
          }
        />
        <DataTable
          data={items}
          columns={columns}
          getRowId={(probe) => probe.id}
          activeRowId={activeId ?? undefined}
          onRowClick={(probe) => onOpen(probe.id)}
          loading={loading && !result}
          rowActions={(probe) => (
            <>
              <DropdownMenuItem onSelect={() => onOpen(probe.id)}>
                <Eye />
                View details
              </DropdownMenuItem>
              <DropdownMenuItem disabled={busy} onSelect={() => void validateProbe(probe)}>
                <CheckCheck />
                Validate route
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                disabled={busy || !probe.playground.enabled}
                onSelect={() => void launchProbe('run', probe)}
              >
                <Play />
                Run in Playground
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={busy || !probe.editable || !probe.playground.enabled}
                onSelect={() => void launchProbe('edit', probe)}
              >
                <Pencil />
                Edit in Playground
              </DropdownMenuItem>
            </>
          )}
          className={cn(loading && result && 'opacity-70 transition-opacity')}
          empty={
            <EmptyState
              compact
              icon={FlaskConical}
              tone="orange"
              title={hasFilters ? 'No probes match' : 'This recipe has no probes'}
              description={
                hasFilters
                  ? 'Try another search or clear the filters.'
                  : 'Probes are verified requests a recipe package ships with its decisions.'
              }
              actions={
                hasFilters ? (
                  <Button variant="outline" size="sm" onClick={clearFilters}>
                    Clear filters
                  </Button>
                ) : undefined
              }
            />
          }
        />
        {result ? (
          <ConfigPageTablePager
            itemLabel="probes"
            page={result.page}
            totalPages={Math.max(result.total_pages, 1)}
            total={result.total}
            start={(result.page - 1) * result.page_size}
            end={Math.min(result.total, result.page * result.page_size)}
            onPageChange={setPage}
            disabled={loading}
          />
        ) : null}
      </div>

      <DetailSheet
        open={Boolean(selected)}
        onOpenChange={(open) => (open ? null : onClose())}
        size="lg"
      >
        {shown ? (
          <>
            <DetailSheetHeader
              icon={<IconTile icon={FlaskConical} tone="orange" size="lg" />}
              title={<span className="font-mono">{shown.variant_id}</span>}
              badges={
                <>
                  {validation[shown.id] ? <ValidationBadge result={validation[shown.id]} /> : null}
                  {!shown.playground.enabled ? <Badge tone="outline">Eval only</Badge> : null}
                </>
              }
              description={
                <>
                  <span className="font-mono">{shown.decision_id}</span>
                  <span className="text-subtle-foreground">·</span>
                  <span>{shown.request_shapes.join(', ') || 'text'}</span>
                </>
              }
            />
            <DetailSheetBody className="flex flex-col gap-5 px-5 py-5">
              {actionError[shown.id] ? (
                <div role="alert">
                  <Alert tone="danger" title="The action failed">
                    {actionError[shown.id]}
                  </Alert>
                </div>
              ) : null}
              {!shown.playground.enabled ? (
                <Alert tone="info" title="Eval-only probe">
                  {shown.playground.reason}
                </Alert>
              ) : null}
              {validation[shown.id] ? (
                <RecipeProbeValidation result={validation[shown.id]} />
              ) : null}
              <section className="flex flex-col gap-2">
                <SectionLabel>Expected route</SectionLabel>
                <ExpectedRoute expected={shown.expected} />
              </section>
              <section className="flex flex-col gap-2">
                <SectionLabel>Request</SectionLabel>
                {detailLoading ? (
                  <div className="flex flex-col gap-2" aria-label="Loading probe">
                    <Skeleton className="h-4 w-40" />
                    <Skeleton className="h-20 w-full" />
                  </div>
                ) : detailError ? (
                  <div role="alert">
                    <Alert tone="danger">{detailError}</Alert>
                  </div>
                ) : detail ? (
                  <RecipeProbeDetailPanel detail={detail} />
                ) : null}
              </section>
            </DetailSheetBody>
            <DetailSheetFooter>
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                loading={activeAction === `validate:${shown.id}`}
                onClick={() => void validateProbe(shown)}
              >
                {activeAction === `validate:${shown.id}` ? null : <CheckCheck />}
                Validate route
              </Button>
              <div className="ml-auto flex items-center gap-2">
                <Tooltip
                  content={
                    !shown.playground.enabled
                      ? shown.playground.reason
                      : shown.editable
                        ? 'Edit this probe in Playground'
                        : 'This structured probe does not end in a user message'
                  }
                >
                  <span>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={busy || !shown.editable || !shown.playground.enabled}
                      onClick={() => void launchProbe('edit', shown)}
                    >
                      <Pencil />
                      Edit in Playground
                    </Button>
                  </span>
                </Tooltip>
                <Button
                  size="sm"
                  disabled={busy || !shown.playground.enabled}
                  loading={activeAction === `run:${shown.id}`}
                  onClick={() => void launchProbe('run', shown)}
                >
                  {activeAction === `run:${shown.id}` ? null : <MessagesSquare />}
                  Run in Playground
                </Button>
              </div>
            </DetailSheetFooter>
          </>
        ) : null}
      </DetailSheet>
    </div>
  )
}
