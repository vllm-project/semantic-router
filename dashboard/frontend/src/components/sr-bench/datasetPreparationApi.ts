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

export interface PreparationRequest {
  benchmark: string
  profile: PreparationProfile
  seed?: number
  limit?: number
}

export interface DatasetPreparationJob extends PreparationRequest {
  id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  phase: string
  created_at: string
  updated_at: string
  dataset?: Dataset
  error?: string
}

export const datasetPreparationApi = {
  options: (signal?: AbortSignal) =>
    request<{ benchmarks: PreparationBenchmark[] }>('/dataset-preparations/options', { signal }),
  list: (signal?: AbortSignal) =>
    request<{ preparations: DatasetPreparationJob[] }>('/dataset-preparations', { signal }),
  prepare: (body: PreparationRequest) =>
    request<{ preparation: DatasetPreparationJob }>('/dataset-preparations', {
      method: 'POST',
      body: JSON.stringify(body),
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
      checking_dependencies: 'Checking required dependencies',
      installing_dependencies: 'Installing required dependencies',
      downloading: 'Downloading source data',
      freezing: 'Freezing the selected questions',
    }[job.phase] ?? 'Preparing dataset'
  )
}
