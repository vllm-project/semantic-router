import { useEffect, useRef, useState } from 'react'
import { benchApi } from './api'
import { active } from './model'
import type { CallRecord, CaseResult, PageState, Report, Run, RunEvent } from './types'

const emptyPage: PageState = { total: null, nextCursor: null, loading: false, error: '' }
const initialPage: PageState = { ...emptyPage, loading: true }

function readFailure(cause: unknown) {
  return cause instanceof Error ? cause.message : 'Refresh to retry the read.'
}

function eventPage(events: RunEvent[], loaded = 0, after = 0): PageState {
  const next = Math.max(after, ...events.map((event) => event.seq ?? event.sequence ?? 0))
  if (events.length >= 1000 && next <= after)
    throw new Error('Saved event page did not provide an advancing cursor.')
  return {
    total: events.length < 1000 ? loaded + events.length : null,
    nextCursor: events.length >= 1000 ? next : null,
    loading: false,
    error: '',
  }
}

export function useRunEvidence(id: string, revision: number) {
  const [run, setRun] = useState<Run | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [reportRead, setReportRead] = useState({ loading: true, error: '' })
  const [results, setResults] = useState<CaseResult[]>([])
  const [events, setEvents] = useState<RunEvent[]>([])
  const [eventsPage, setEventsPage] = useState<PageState>(initialPage)
  const [calls, setCalls] = useState<CallRecord[]>([])
  const [resultsPage, setResultsPage] = useState<PageState>(initialPage)
  const [callsPage, setCallsPage] = useState<PageState>(initialPage)
  const [error, setError] = useState('')
  const [readAt, setReadAt] = useState<string | null>(null)
  const control = useRef<{
    controller: AbortController
    resultsExpanded: boolean
    callsExpanded: boolean
    eventsLoaded: boolean
    eventsLoading: boolean
  } | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const currentControl = {
      controller,
      resultsExpanded: false,
      callsExpanded: false,
      eventsLoaded: false,
      eventsLoading: false,
    }
    control.current = currentControl
    let timer: ReturnType<typeof setTimeout> | undefined
    let lastUpdated = ''
    setRun(null)
    setReport(null)
    setReportRead({ loading: true, error: '' })
    setResults([])
    setEvents([])
    setEventsPage(initialPage)
    setCalls([])
    setResultsPage(initialPage)
    setCallsPage(initialPage)
    setError('')

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
          if (!currentControl.eventsLoaded)
            setEventsPage((previous) => ({ ...previous, loading: true, error: '' }))
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
            currentControl.eventsLoaded
              ? Promise.resolve(null)
              : benchApi.events(id, 0, controller.signal).then((response) => ({
                  events: response.events,
                  page: eventPage(response.events),
                })),
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
          if (responses[3].status === 'fulfilled' && responses[3].value) {
            const page = responses[3].value.events
            setEvents(page)
            currentControl.eventsLoaded = true
            setEventsPage(responses[3].value.page)
          } else if (responses[3].status === 'rejected') {
            const reason = readFailure(responses[3].reason)
            setEventsPage((previous) => ({
              ...previous,
              loading: false,
              error: reason,
            }))
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

  async function loadMoreEvents() {
    const current = control.current
    if (
      !current ||
      current.controller.signal.aborted ||
      current.eventsLoading ||
      eventsPage.nextCursor === null
    )
      return
    current.eventsLoading = true
    setEventsPage((previous) => ({ ...previous, loading: true, error: '' }))
    try {
      const response = await benchApi.events(id, eventsPage.nextCursor, current.controller.signal)
      if (current.controller.signal.aborted) return
      const page = response.events
      const state = eventPage(page, events.length, eventsPage.nextCursor)
      setEvents((previous) => [...previous, ...page])
      setEventsPage(state)
    } catch (cause) {
      if (!current.controller.signal.aborted)
        setEventsPage((previous) => ({ ...previous, loading: false, error: readFailure(cause) }))
    } finally {
      current.eventsLoading = false
    }
  }

  return {
    run,
    report,
    reportRead,
    results,
    events,
    eventsPage,
    loadMoreEvents,
    calls,
    error,
    readAt,
    resultsPage,
    callsPage,
    loadMoreResults,
    loadMoreCalls,
  }
}
