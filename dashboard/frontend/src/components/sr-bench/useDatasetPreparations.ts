import { useCallback, useEffect, useRef, useState } from 'react'
import {
  datasetPreparationApi,
  preparationBenchmarkIDs,
  preparationIsActive,
  type DatasetPreparationJob,
  type PreparationBenchmark,
  type PreparationRequest,
} from './datasetPreparationApi'

/** Jobs are always polled; the downloadable catalog is read only while `catalogOpen`. */
export function useDatasetPreparations(onCompleted: () => void, catalogOpen = true) {
  const [benchmarks, setBenchmarks] = useState<PreparationBenchmark[]>([])
  const [preparations, setPreparations] = useState<DatasetPreparationJob[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [readError, setReadError] = useState('')
  const [submitError, setSubmitError] = useState('')
  const [revision, setRevision] = useState(0)
  const completed = useRef(new Set<string>())
  const submittingRef = useRef(false)
  const refresh = useCallback(() => setRevision((value) => value + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    async function load() {
      let busy = false
      let failed = false
      try {
        const [options, history] = await Promise.all([
          catalogOpen ? datasetPreparationApi.options(controller.signal) : null,
          datasetPreparationApi.list(controller.signal),
        ])
        if (controller.signal.aborted) return
        if (options) setBenchmarks(options.benchmarks)
        setPreparations(history.preparations)
        setReadError('')
        busy = history.preparations.some(preparationIsActive)
        let changed = false
        for (const preparation of history.preparations) {
          if (preparation.status === 'completed' && !completed.current.has(preparation.id)) {
            completed.current.add(preparation.id)
            changed = true
          }
        }
        if (changed) onCompleted()
      } catch (cause) {
        failed = true
        if (!controller.signal.aborted)
          setReadError(
            cause instanceof Error ? cause.message : 'Could not read dataset preparations.',
          )
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
          timer = setTimeout(() => void load(), busy ? 2000 : failed ? 5000 : 15000)
        }
      }
    }
    void load()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [onCompleted, revision, catalogOpen])

  async function prepare(body: PreparationRequest) {
    if (submittingRef.current) return
    submittingRef.current = true
    setSubmitting(true)
    setSubmitError('')
    try {
      const { preparation } = await datasetPreparationApi.prepare(body)
      setPreparations((previous) => [
        preparation,
        ...previous.filter((item) => item.id !== preparation.id),
      ])
    } catch (cause) {
      setSubmitError(
        `${cause instanceof Error ? cause.message : 'Could not submit dataset preparation.'} Check the preparation history before trying again.`,
      )
    } finally {
      submittingRef.current = false
      setSubmitting(false)
      setLoading(true)
      // A lost submission response may still have created a service-owned job.
      // Reconcile history; never automatically repeat the POST.
      refresh()
    }
  }

  return { benchmarks, preparations, loading, submitting, readError, submitError, prepare, refresh }
}

/** Resubmits a failed job's own benchmarks, profile and seed; only ever on an explicit click. */
export function retryPreparation(
  state: Pick<ReturnType<typeof useDatasetPreparations>, 'prepare'>,
  job: DatasetPreparationJob,
) {
  const ids = preparationBenchmarkIDs(job)
  if (!ids.length) return
  void state.prepare(
    job.benchmarks
      ? { benchmarks: ids, profile: job.profile, seed: job.seed ?? undefined }
      : {
          benchmark: ids[0],
          profile: job.profile,
          seed: job.seed ?? undefined,
          limit: job.limit ?? undefined,
        },
  )
}
