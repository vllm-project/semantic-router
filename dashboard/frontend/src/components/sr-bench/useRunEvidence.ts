import { useEffect, useRef, useState } from 'react'
import { benchApi } from './api'
import { active } from './model'
import type { CallRecord, CaseResult, PageState, Report, Run, RunEvent } from './types'

const emptyPage: PageState = { total: null, nextCursor: null, loading: false, error: '' }
const initialPage: PageState = { ...emptyPage, loading: true }

function readFailure(cause: unknown) {
  return cause instanceof Error ? cause.message : 'Refresh to retry the read.'
}

export function useRunEvidence(id: string, revision: number) {
  const [run, setRun] = useState<Run | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [reportRead, setReportRead] = useState({ loading: true, error: '' })
  const [results, setResults] = useState<CaseResult[]>([])
  const [events, setEvents] = useState<RunEvent[]>([])
  const [calls, setCalls] = useState<CallRecord[]>([])
  const [resultsPage, setResultsPage] = useState<PageState>(initialPage)
  const [callsPage, setCallsPage] = useState<PageState>(initialPage)
  const [error, setError] = useState('')
  const [readAt, setReadAt] = useState<string | null>(null)
  const control = useRef<{
    controller: AbortController
    resultsExpanded: boolean
    callsExpanded: boolean
  } | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const currentControl = { controller, resultsExpanded: false, callsExpanded: false }
    control.current = currentControl
    let timer: ReturnType<typeof setTimeout> | undefined
    let eventCursor = 0
    let lastUpdated = ''
    setRun(null)
    setReport(null)
    setReportRead({ loading: true, error: '' })
    setResults([])
    setEvents([])
    setCalls([])
    setResultsPage(initialPage)
    setCallsPage(initialPage)
    setError('')

    async function readEvents() {
      const all: RunEvent[] = []
      let cursor = eventCursor
      // The service limits event pages to 1,000. Drain saved pages in order,
      // including terminal runs, without ever replaying a model request.
      while (!controller.signal.aborted) {
        const page = await benchApi.events(id, cursor, controller.signal)
        if (!page.events.length) break
        all.push(...page.events)
        const next = Math.max(...page.events.map((event) => event.seq ?? 0))
        if (next <= cursor) break
        cursor = next
        if (page.events.length < 1000) break
      }
      // Commit only the events this read can display. A later page failure must
      // leave the cursor unchanged so the next read also recovers earlier pages.
      eventCursor = cursor
      return all
    }

    async function load() {
      let terminal = false
      let evidenceIncomplete = false
      try {
        const current = await benchApi.run(id, controller.signal)
        if (controller.signal.aborted) return
        setRun(current)
        setReadAt(new Date().toLocaleTimeString())
        terminal = !active(current.status)
        if (current.updated_at !== lastUpdated) {
          setReportRead({ loading: true, error: '' })
          if (!currentControl.resultsExpanded)
            setResultsPage((previous) => ({ ...previous, loading: true, error: '' }))
          if (!currentControl.callsExpanded)
            setCallsPage((previous) => ({ ...previous, loading: true, error: '' }))
          const responses = await Promise.allSettled([
            benchApi.report(id, controller.signal),
            currentControl.resultsExpanded
              ? Promise.resolve(null)
              : benchApi.results(id, 0, controller.signal),
            currentControl.callsExpanded
              ? Promise.resolve(null)
              : benchApi.calls(id, 0, controller.signal),
            readEvents(),
          ])
          if (controller.signal.aborted) return
          if (responses[0].status === 'fulfilled') setReport(responses[0].value)
          setReportRead({
            loading: false,
            error: responses[0].status === 'rejected' ? readFailure(responses[0].reason) : '',
          })
          if (
            responses[1].status === 'fulfilled' &&
            responses[1].value &&
            !currentControl.resultsExpanded
          ) {
            const page = responses[1].value
            setResults(page.results)
            setResultsPage({
              total: page.total,
              nextCursor: page.next_cursor,
              loading: false,
              error: '',
            })
          } else if (responses[1].status === 'rejected' && !currentControl.resultsExpanded) {
            const reason = readFailure(responses[1].reason)
            setResultsPage((previous) => ({
              ...previous,
              loading: false,
              error: `Case results are unavailable: ${reason}`,
            }))
          }
          if (
            responses[2].status === 'fulfilled' &&
            responses[2].value &&
            !currentControl.callsExpanded
          ) {
            const page = responses[2].value
            setCalls(page.calls)
            setCallsPage({
              total: page.total,
              nextCursor: page.next_cursor,
              loading: false,
              error: '',
            })
          } else if (responses[2].status === 'rejected' && !currentControl.callsExpanded) {
            const reason = readFailure(responses[2].reason)
            setCallsPage((previous) => ({
              ...previous,
              loading: false,
              error: `Call records are unavailable: ${reason}`,
            }))
          }
          if (responses[3].status === 'fulfilled') {
            const newEvents = responses[3].value
            setEvents((previous) => [...previous, ...newEvents])
          }
          const failed = responses.find((response) => response.status === 'rejected')
          evidenceIncomplete = !!failed
          setError(
            failed?.status === 'rejected'
              ? `Some run evidence is unavailable: ${failed.reason instanceof Error ? failed.reason.message : 'Refresh to retry the read.'}`
              : '',
          )
          if (!failed) lastUpdated = current.updated_at
        }
      } catch (cause) {
        if (controller.signal.aborted) return
        evidenceIncomplete = true
        setError(cause instanceof Error ? cause.message : 'Run could not be loaded.')
      }
      if (!controller.signal.aborted && (!terminal || evidenceIncomplete))
        timer = setTimeout(() => void load(), evidenceIncomplete ? 5000 : 2500)
    }
    void load()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [id, revision])

  async function loadMoreResults() {
    const current = control.current
    if (
      !current ||
      current.controller.signal.aborted ||
      resultsPage.loading ||
      resultsPage.nextCursor === null
    )
      return
    current.resultsExpanded = true
    setResultsPage((previous) => ({ ...previous, loading: true, error: '' }))
    try {
      const page = await benchApi.results(id, resultsPage.nextCursor, current.controller.signal)
      if (current.controller.signal.aborted) return
      setResults((previous) => {
        const unique = new Map(previous.map((row) => [`${row.target_id}\0${row.case_id}`, row]))
        for (const row of page.results) unique.set(`${row.target_id}\0${row.case_id}`, row)
        return [...unique.values()]
      })
      setResultsPage({ total: page.total, nextCursor: page.next_cursor, loading: false, error: '' })
    } catch (cause) {
      if (!current.controller.signal.aborted)
        setResultsPage((previous) => ({
          ...previous,
          loading: false,
          error: cause instanceof Error ? cause.message : 'Could not load the next result page.',
        }))
    }
  }

  async function loadMoreCalls() {
    const current = control.current
    if (
      !current ||
      current.controller.signal.aborted ||
      callsPage.loading ||
      callsPage.nextCursor === null
    )
      return
    current.callsExpanded = true
    setCallsPage((previous) => ({ ...previous, loading: true, error: '' }))
    try {
      const page = await benchApi.calls(id, callsPage.nextCursor, current.controller.signal)
      if (current.controller.signal.aborted) return
      setCalls((previous) => {
        const unique = new Map(previous.map((row) => [row.id, row]))
        for (const row of page.calls) unique.set(row.id, row)
        return [...unique.values()]
      })
      setCallsPage({ total: page.total, nextCursor: page.next_cursor, loading: false, error: '' })
    } catch (cause) {
      if (!current.controller.signal.aborted)
        setCallsPage((previous) => ({
          ...previous,
          loading: false,
          error: cause instanceof Error ? cause.message : 'Could not load the next call page.',
        }))
    }
  }

  return {
    run,
    report,
    reportRead,
    results,
    events,
    calls,
    error,
    readAt,
    resultsPage,
    callsPage,
    loadMoreResults,
    loadMoreCalls,
  }
}
