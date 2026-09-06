import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { MutableRefObject, RefObject } from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  modelHubCapabilities,
  modelHubCreators,
  modelHubProviders,
  modelHubRows,
  modelHubStats,
  paginateModelHubRows,
  resolveModelHubSelection,
  type ModelHubFilters,
  type ModelHubView,
} from './modelHubSupport'

const initialFilters: ModelHubFilters = {
  query: '',
  kind: 'all',
  distribution: 'all',
  lifecycle: 'supported',
  publisher: 'all',
  provider: 'all',
  capability: 'all',
  sort: 'released',
}

export const compactModelHubLayoutQuery = '(max-width: 1050px)'
const focusableElementSelector =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

const useCompactModelHubLayout = (): boolean => {
  const [compact, setCompact] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(compactModelHubLayoutQuery).matches,
  )

  useEffect(() => {
    const media = window.matchMedia(compactModelHubLayoutQuery)
    const update = (): void => setCompact(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  return compact
}

function keepFocusInDialog(event: KeyboardEvent, dialog?: HTMLElement | null): void {
  if (event.key !== 'Tab' || !dialog) return
  const focusable = Array.from(
    dialog.querySelectorAll<HTMLElement>(focusableElementSelector),
  ).filter((element) => !element.hidden)
  if (!focusable.length) {
    event.preventDefault()
    dialog.focus()
    return
  }
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  if (!dialog.contains(document.activeElement)) {
    event.preventDefault()
    const destination = event.shiftKey ? last : first
    destination.focus()
  } else if (event.shiftKey && document.activeElement === first) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
}

function useMobileDetailDialog(
  compact: boolean,
  detailOpen: boolean,
  closeDetail: () => void,
  detailLayerRef: RefObject<HTMLDivElement>,
  detailCloseRef: RefObject<HTMLButtonElement>,
  detailTriggerRef: MutableRefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!compact || !detailOpen) return undefined
    const dialog = detailLayerRef.current?.querySelector<HTMLElement>('[role="dialog"]')
    const detailTrigger = detailTriggerRef.current
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    detailCloseRef.current?.focus()
    const handleDialogKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeDetail()
        return
      }
      keepFocusInDialog(event, dialog)
    }
    window.addEventListener('keydown', handleDialogKey)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', handleDialogKey)
      detailTrigger?.focus()
    }
  }, [closeDetail, compact, detailCloseRef, detailLayerRef, detailOpen, detailTriggerRef])
}

export function useModelHubPageController(catalog: BuiltInModelCatalog) {
  const [filters, setFilters] = useState<ModelHubFilters>(initialFilters)
  const [view, setView] = useState<ModelHubView>('list')
  const [selectedID, setSelectedID] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const compact = useCompactModelHubLayout()
  const [pageSize, setPageSize] = useState(10)
  const [detailOpen, setDetailOpen] = useState(false)
  const detailCloseRef = useRef<HTMLButtonElement>(null)
  const detailLayerRef = useRef<HTMLDivElement>(null)
  const detailTriggerRef = useRef<HTMLElement | null>(null)
  const stats = useMemo(() => modelHubStats(catalog), [catalog])
  const creators = useMemo(() => modelHubCreators(catalog), [catalog])
  const providers = useMemo(() => modelHubProviders(catalog), [catalog])
  const capabilities = useMemo(() => modelHubCapabilities(catalog), [catalog])
  const rows = useMemo(() => modelHubRows(catalog, filters), [catalog, filters])
  const pagination = useMemo(
    () => paginateModelHubRows(rows, page, pageSize),
    [page, pageSize, rows],
  )
  const selected = resolveModelHubSelection(
    view === 'benchmarks' ? rows : pagination.items,
    selectedID,
  )

  useEffect(() => {
    setPage(1)
    if (!compact) setDetailOpen(false)
  }, [compact])

  const closeDetail = useCallback((): void => setDetailOpen(false), [])
  useMobileDetailDialog(
    compact,
    detailOpen,
    closeDetail,
    detailLayerRef,
    detailCloseRef,
    detailTriggerRef,
  )

  const updateFilters = (patch: Partial<ModelHubFilters>): void => {
    setFilters((current) => ({ ...current, ...patch }))
    setPage(1)
  }
  const resetFilters = (): void => {
    setFilters(initialFilters)
    setPage(1)
  }
  const updatePageSize = (next: number): void => {
    setPageSize(next)
    setPage(1)
  }
  const selectModel = (id: string): void => {
    setSelectedID(id)
    if (!compact) return
    detailTriggerRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    setDetailOpen(true)
  }

  return {
    filters,
    view,
    setView,
    compact,
    detailOpen,
    detailCloseRef,
    detailLayerRef,
    stats,
    creators,
    providers,
    capabilities,
    rows,
    pagination,
    selected,
    setPage,
    closeDetail,
    updateFilters,
    resetFilters,
    updatePageSize,
    selectModel,
  }
}
