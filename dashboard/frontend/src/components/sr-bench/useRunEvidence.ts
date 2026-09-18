import { useEffect, useState } from 'react'
import { benchApi } from './api'
import { active } from './model'
import type { CallRecord, CaseResult, Report, Run, RunEvent } from './types'

export function useRunEvidence(id: string, revision: number) {
  const [run, setRun] = useState<Run | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [results, setResults] = useState<CaseResult[]>([])
  const [events, setEvents] = useState<RunEvent[]>([])
  const [calls, setCalls] = useState<CallRecord[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    let eventCursor = 0
    let lastUpdated = ''
    setRun(null)
    setReport(null)
    setResults([])
    setEvents([])
    setCalls([])
    setError('')

    async function readEvents() {
      const all: RunEvent[] = []
      // The service limits event pages to 1,000. Drain saved pages in order,
      // including terminal runs, without ever replaying a model request.
      while (!controller.signal.aborted) {
        const page = await benchApi.events(id, eventCursor, controller.signal)
        if (!page.events.length) break
        all.push(...page.events)
        const next = Math.max(...page.events.map((event) => event.seq ?? 0))
        if (next <= eventCursor) break
        eventCursor = next
        if (page.events.length < 1000) break
      }
      return all
    }

    async function load() {
      let terminal = false
      try {
        const current = await benchApi.run(id, controller.signal)
        if (controller.signal.aborted) return
        setRun(current)
        terminal = !active(current.status)
        if (current.updated_at !== lastUpdated) {
          const responses = await Promise.allSettled([
            benchApi.report(id, controller.signal),
            benchApi.results(id, controller.signal),
            benchApi.calls(id, controller.signal),
            readEvents(),
          ])
          if (controller.signal.aborted) return
          if (responses[0].status === 'fulfilled') setReport(responses[0].value)
          if (responses[1].status === 'fulfilled') setResults(responses[1].value.results)
          if (responses[2].status === 'fulfilled') setCalls(responses[2].value.calls)
          if (responses[3].status === 'fulfilled') {
            const newEvents = responses[3].value
            setEvents((previous) => [...previous, ...newEvents])
          }
          const failed = responses.find((response) => response.status === 'rejected')
          setError(
            failed?.status === 'rejected'
              ? `Some run evidence is unavailable: ${failed.reason instanceof Error ? failed.reason.message : 'Refresh to retry the read.'}`
              : '',
          )
          if (!failed) lastUpdated = current.updated_at
        }
      } catch (cause) {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'Run could not be loaded.')
      }
      if (!controller.signal.aborted && !terminal) timer = setTimeout(() => void load(), 2500)
    }
    void load()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [id, revision])
  return { run, report, results, events, calls, error }
}
