import { useRef, useState, type Dispatch, type SetStateAction } from 'react'

import type {
  EvaluationCatalog,
  EvaluationRun,
  EvaluationRunLedgerWarning,
} from '../types/evaluationPlane'

export interface EvaluationRunPageState {
  nextCursor: string | null
  totalRuns: number
  warningCount: number
}

interface EvaluationLoadPendingState {
  catalog: boolean
  runs: boolean
}

interface MutableValue<T> {
  current: T
}

export interface EvaluationPlaneHookState {
  catalog: EvaluationCatalog | null
  setCatalog: Dispatch<SetStateAction<EvaluationCatalog | null>>
  runs: EvaluationRun[]
  setRuns: Dispatch<SetStateAction<EvaluationRun[]>>
  runsLoaded: boolean
  setRunsLoaded: Dispatch<SetStateAction<boolean>>
  runLedgerComplete: boolean
  setRunLedgerComplete: Dispatch<SetStateAction<boolean>>
  runLedgerWarnings: EvaluationRunLedgerWarning[]
  setRunLedgerWarnings: Dispatch<SetStateAction<EvaluationRunLedgerWarning[]>>
  runPage: EvaluationRunPageState
  setRunPage: Dispatch<SetStateAction<EvaluationRunPageState>>
  loadPending: EvaluationLoadPendingState
  setLoadPending: Dispatch<SetStateAction<EvaluationLoadPendingState>>
  refreshing: boolean
  setRefreshing: Dispatch<SetStateAction<boolean>>
  loadingMoreRuns: boolean
  setLoadingMoreRuns: Dispatch<SetStateAction<boolean>>
  loadingAllRuns: boolean
  setLoadingAllRuns: Dispatch<SetStateAction<boolean>>
  runPollingPaused: boolean
  setRunPollingPaused: Dispatch<SetStateAction<boolean>>
  catalogError: string | null
  setCatalogError: Dispatch<SetStateAction<string | null>>
  runsError: string | null
  setRunsError: Dispatch<SetStateAction<string | null>>
  lastUpdatedAt: Date | null
  setLastUpdatedAt: Dispatch<SetStateAction<Date | null>>
  mutationPending: boolean
  setMutationPending: Dispatch<SetStateAction<boolean>>
  mutationKey: string | null
  setMutationKey: Dispatch<SetStateAction<string | null>>
  mutationError: string | null
  setMutationError: Dispatch<SetStateAction<string | null>>
  catalogRequestVersion: MutableValue<number>
  runsRequestVersion: MutableValue<number>
  catalogController: MutableValue<AbortController | null>
  runsController: MutableValue<AbortController | null>
  runsRefreshPromise: MutableValue<Promise<boolean> | null>
  mutationLock: MutableValue<boolean>
  loadedPageCount: MutableValue<number>
  loadingMoreRequest: MutableValue<boolean>
}

export function useEvaluationPlaneHookState(): EvaluationPlaneHookState {
  const [catalog, setCatalog] = useState<EvaluationCatalog | null>(null)
  const [runs, setRuns] = useState<EvaluationRun[]>([])
  const [runsLoaded, setRunsLoaded] = useState(false)
  const [runLedgerComplete, setRunLedgerComplete] = useState(false)
  const [runLedgerWarnings, setRunLedgerWarnings] = useState<EvaluationRunLedgerWarning[]>([])
  const [runPage, setRunPage] = useState<EvaluationRunPageState>({
    nextCursor: null,
    totalRuns: 0,
    warningCount: 0,
  })
  const [loadPending, setLoadPending] = useState({ catalog: true, runs: true })
  const [refreshing, setRefreshing] = useState(false)
  const [loadingMoreRuns, setLoadingMoreRuns] = useState(false)
  const [loadingAllRuns, setLoadingAllRuns] = useState(false)
  const [runPollingPaused, setRunPollingPaused] = useState(false)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [runsError, setRunsError] = useState<string | null>(null)
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null)
  const [mutationPending, setMutationPending] = useState(false)
  const [mutationKey, setMutationKey] = useState<string | null>(null)
  const [mutationError, setMutationError] = useState<string | null>(null)
  const catalogRequestVersion = useRef(0)
  const runsRequestVersion = useRef(0)
  const catalogController = useRef<AbortController | null>(null)
  const runsController = useRef<AbortController | null>(null)
  const runsRefreshPromise = useRef<Promise<boolean> | null>(null)
  const mutationLock = useRef(false)
  const loadedPageCount = useRef(0)
  const loadingMoreRequest = useRef(false)

  return {
    catalog,
    setCatalog,
    runs,
    setRuns,
    runsLoaded,
    setRunsLoaded,
    runLedgerComplete,
    setRunLedgerComplete,
    runLedgerWarnings,
    setRunLedgerWarnings,
    runPage,
    setRunPage,
    loadPending,
    setLoadPending,
    refreshing,
    setRefreshing,
    loadingMoreRuns,
    setLoadingMoreRuns,
    loadingAllRuns,
    setLoadingAllRuns,
    runPollingPaused,
    setRunPollingPaused,
    catalogError,
    setCatalogError,
    runsError,
    setRunsError,
    lastUpdatedAt,
    setLastUpdatedAt,
    mutationPending,
    setMutationPending,
    mutationKey,
    setMutationKey,
    mutationError,
    setMutationError,
    catalogRequestVersion,
    runsRequestVersion,
    catalogController,
    runsController,
    runsRefreshPromise,
    mutationLock,
    loadedPageCount,
    loadingMoreRequest,
  }
}
