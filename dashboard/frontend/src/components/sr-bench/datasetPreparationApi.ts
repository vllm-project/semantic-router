import { request } from './api'
import type { Dataset } from './types'

export type PreparationProfile = 'smoke' | 'quick' | 'standard'

export interface PreparationBenchmark {
  id: string
  name: string
  profiles: Record<PreparationProfile, number>
  source_url: string
  access_note: string
  dependencies: string[]
}

interface PreparationOptions {
  profile: PreparationProfile
  seed?: number
}

export type PreparationRequest = PreparationOptions &
  (
    | { benchmark: string; benchmarks?: never; limit?: number }
    | { benchmarks: string[]; benchmark?: never; limit?: never }
  )

export interface DatasetPreparationJob extends PreparationOptions {
  id: string
  benchmark?: string
  benchmarks?: string[]
  limit?: number
  status: 'queued' | 'running' | 'completed' | 'failed'
  phase: string
  created_at: string
  updated_at: string
  dataset?: Dataset
  error?: string
  items?: Array<{
    benchmark: string
    status: 'queued' | 'running' | 'completed' | 'failed'
    phase: string
    reused: boolean
    source_ids: string[]
    dataset_id?: string
    error_code?: string
    error?: string
  }>
}

export const preparationBenchmarkIDs = (job: DatasetPreparationJob): string[] =>
  job.benchmarks ?? (job.benchmark ? [job.benchmark] : [])

export const datasetPreparationApi = {
  options: (signal?: AbortSignal) =>
    request<{ benchmarks: PreparationBenchmark[] }>('/dataset-preparations/options', { signal }),
  list: (signal?: AbortSignal) =>
    request<{ preparations: DatasetPreparationJob[] }>('/dataset-preparations', { signal }),
  get: (id: string, signal?: AbortSignal) =>
    request<{ preparation: DatasetPreparationJob }>(
      `/dataset-preparations/${encodeURIComponent(id)}`,
      { signal },
    ),
  prepare: (body: PreparationRequest, signal?: AbortSignal) =>
    request<{ preparation: DatasetPreparationJob }>('/dataset-preparations', {
      method: 'POST',
      body: JSON.stringify(body),
      signal,
    }),
}

export function preparationIsActive(job: DatasetPreparationJob): boolean {
  return job.status === 'queued' || job.status === 'running'
}

export function preparationPhase(job: DatasetPreparationJob): string {
  if (job.status === 'completed') return 'Ready to evaluate'
  if (job.status === 'failed') return 'Preparation failed'
  return (
    {
      queued: 'Waiting to start',
      resolving_datasets: 'Checking available datasets',
      checking_dependencies: 'Checking required dependencies',
      installing_dependencies: 'Installing required dependencies',
      downloading: 'Downloading source data',
      freezing: 'Freezing the selected questions',
      composing: 'Finalizing the evaluation questions',
    }[job.phase] ?? 'Preparing dataset'
  )
}
