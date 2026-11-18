// Test Query functionality always uses backend verification.
import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { TestQueryResult, ParsedTopology } from '../types'
import { testQueryDryRun } from '../utils/api'
import { parsePreviewEnvelope, type PreviewOptions } from '../previewRequest'

export function useTestQuery(
  _topologyData: ParsedTopology | null,
  routingModel?: string,
  recipe?: string,
) {
  const [testQuery, setQuery] = useState('')
  const [requestContext, setContext] = useState('')
  const [testResult, setTestResult] = useState<TestQueryResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const activeRequest = useRef<AbortController | null>(null)
  const parsed = useMemo(() => {
    try {
      return { envelope: parsePreviewEnvelope(requestContext), error: undefined }
    } catch (error) {
      return {
        envelope: undefined,
        error: error instanceof Error ? error.message : 'Invalid request context.',
      }
    }
  }, [requestContext])
  const disabledReason =
    recipe && !routingModel
      ? 'This recipe has no Chat entrypoint. Select a recipe with an entrypoint to run a preview.'
      : parsed.error
  const hasInput = Boolean(testQuery.trim() || parsed.envelope?.messages?.length)

  const clearResult = useCallback(() => {
    activeRequest.current?.abort()
    activeRequest.current = null
    setTestResult(null)
    setIsLoading(false)
  }, [])

  useEffect(() => {
    clearResult()
    return () => activeRequest.current?.abort()
  }, [routingModel, recipe, clearResult])

  const setTestQuery = useCallback(
    (value: string) => {
      clearResult()
      setQuery(value)
    },
    [clearResult],
  )
  const setRequestContext = useCallback(
    (value: string) => {
      clearResult()
      setContext(value)
    },
    [clearResult],
  )

  const runTest = useCallback(async () => {
    if (!hasInput || disabledReason) return
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    setTestResult(null)
    setIsLoading(true)
    try {
      const result = await runTestQueryPreview(
        testQuery,
        routingModel,
        { ...parsed.envelope, recipe },
        controller.signal,
      )
      if (!controller.signal.aborted && activeRequest.current === controller) setTestResult(result)
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null
        setIsLoading(false)
      }
    }
  }, [testQuery, routingModel, recipe, parsed.envelope, hasInput, disabledReason])

  return {
    testQuery,
    setTestQuery,
    requestContext,
    setRequestContext,
    testResult,
    isLoading,
    runTest,
    clearResult,
    disabledReason,
    hasInput,
  }
}

export async function runTestQueryPreview(
  query: string,
  model?: string,
  options: PreviewOptions = {},
  signal?: AbortSignal,
): Promise<TestQueryResult> {
  try {
    const result = await testQueryDryRun(query, model, options, signal)
    return { ...result, mode: 'dry-run' }
  } catch (error) {
    return {
      query,
      mode: 'dry-run',
      matchedSignals: [],
      matchedDecision: null,
      matchedModels: [],
      highlightedPath: ['client'],
      isAccurate: false,
      warning:
        error instanceof Error ? `Preview unavailable: ${error.message}` : 'Preview unavailable',
    }
  }
}
