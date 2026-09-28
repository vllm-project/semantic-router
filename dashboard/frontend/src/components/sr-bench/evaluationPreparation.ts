import { SrBenchRequestError } from './api'
import {
  datasetPreparationApi,
  preparationBenchmarkIDs,
  type DatasetPreparationJob,
  type PreparationProfile,
} from './datasetPreparationApi'
import type { Dataset } from './types'

export interface EvaluationPreparationProgress {
  phase: string
  job?: DatasetPreparationJob
}

function pause(signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    signal.throwIfAborted()
    const abort = () => {
      clearTimeout(timer)
      reject(signal.reason)
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort)
      resolve()
    }, 1500)
    signal.addEventListener('abort', abort, { once: true })
  })
}

function matches(
  job: DatasetPreparationJob,
  body: { benchmarks: string[]; profile: PreparationProfile; seed: number },
) {
  return (
    !!job.benchmarks &&
    job.profile === body.profile &&
    job.seed === body.seed &&
    JSON.stringify([...preparationBenchmarkIDs(job)].sort()) === JSON.stringify(body.benchmarks)
  )
}

/** Observe a service-owned preparation; leaving this page does not cancel its work. */
export async function prepareEvaluationDataset(
  body: { benchmarks: string[]; profile: PreparationProfile; seed: number },
  signal: AbortSignal,
  onProgress: (progress: EvaluationPreparationProgress) => void,
): Promise<Dataset> {
  const request = { ...body, benchmarks: [...new Set(body.benchmarks)].sort() }
  let job: DatasetPreparationJob | undefined
  onProgress({ phase: 'resolving_datasets' })
  // Only a definitive busy rejection can be retried automatically. A lost POST
  // response is reconciled by reads and never followed by another implicit POST.
  while (!job) {
    signal.throwIfAborted()
    // Completed batches are deliberately revalidated by the service. Remember
    // existing jobs so an ambiguous response cannot adopt stale success.
    const before = await datasetPreparationApi.list(signal)
    const previousIDs = new Set(before.preparations.map((candidate) => candidate.id))
    try {
      job = (await datasetPreparationApi.prepare(request, signal)).preparation
    } catch (cause) {
      signal.throwIfAborted()
      if (
        cause instanceof SrBenchRequestError &&
        cause.status === 409 &&
        cause.code === 'preparation_busy'
      ) {
        onProgress({ phase: 'waiting_for_slot' })
        await pause(signal)
        continue
      }
      if (cause instanceof SrBenchRequestError && cause.code === 'preparation_unavailable')
        throw cause
      if (cause instanceof SrBenchRequestError && cause.status < 500 && cause.status !== 408)
        throw cause
      const history = await datasetPreparationApi.list(signal).catch(() => null)
      signal.throwIfAborted()
      job = history?.preparations.find(
        (candidate) => !previousIDs.has(candidate.id) && matches(candidate, request),
      )
      if (!job) throw cause
    }
  }
  const id = job.id
  while (true) {
    signal.throwIfAborted()
    if (job.id !== id || !matches(job, request))
      throw new Error(
        'The preparation no longer matches this evaluation. Review the selection again.',
      )
    onProgress({ phase: job.phase, job })
    if (job.status === 'failed')
      throw new Error(job.error || 'Could not prepare the evaluation data.')
    if (job.status === 'completed') {
      const dataset = job.dataset
      if (
        !dataset ||
        dataset.profile !== request.profile ||
        dataset.seed !== request.seed ||
        JSON.stringify([...(dataset.benchmarks ?? [])].sort()) !==
          JSON.stringify(request.benchmarks)
      )
        throw new Error('The prepared dataset differs from the selected evaluation scope.')
      return dataset
    }
    await pause(signal)
    job = (await datasetPreparationApi.get(id, signal)).preparation
  }
}
