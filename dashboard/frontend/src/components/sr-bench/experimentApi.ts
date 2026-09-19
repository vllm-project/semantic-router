import { request } from './api'
import type { ExperimentRunContext } from './types'

export interface Experiment {
  id: string
  name: string
  created_at: string
  updated_at: string
  run_count?: number
  active_run_count?: number
}
export interface ExperimentMember {
  run_id: string
  role: ExperimentRunContext['role']
  hypothesis: string
  linked_at: string
}
export interface ExperimentPage {
  experiment: Experiment
  members: ExperimentMember[]
  next_cursor: number | null
  has_more: boolean
}
const path = (id: string) => `/experiments/${encodeURIComponent(id)}`
export const experimentApi = {
  list: (after = 0, signal?: AbortSignal) =>
    request<{ experiments: Experiment[]; next_cursor: number | null; has_more: boolean }>(
      `/experiments?after=${after}&limit=20`,
      { signal },
    ),
  create: (name: string, key: string) =>
    request<Experiment>('/experiments', {
      method: 'POST',
      body: JSON.stringify({ name, idempotency_key: key }),
    }),
  runs: (id: string, after = 0, signal?: AbortSignal) =>
    request<ExperimentPage>(`${path(id)}/runs?after=${after}&limit=20`, { signal }),
  delete: (id: string) =>
    request<{
      id: string
      deleted: true
      unlinked_runs: number
      runs_deleted: 0
      model_requests: 0
    }>(path(id), { method: 'DELETE' }),
  attach: (id: string, run: string, role: ExperimentRunContext['role'], hypothesis: string) =>
    request<Experiment>(`${path(id)}/runs`, {
      method: 'POST',
      body: JSON.stringify({ run_id: run, role, hypothesis }),
    }),
}
