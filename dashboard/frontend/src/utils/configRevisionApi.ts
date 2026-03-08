import type { ConfigVersion } from '@/types/dsl'

const REVISION_API_BASE = '/api/router/config/revisions'

export interface ConfigRevisionSummary {
  id: string
  parentRevisionId?: string
  status: string
  source?: string
  summary?: string
  createdBy?: string
  runtimeTarget?: string
  lastDeployStatus?: string
  lastDeployMessage?: string
  activatedAt?: string
  lastDeployedAt?: string
  createdAt: string
  updatedAt: string
}

export interface ConfigRevisionDetail extends ConfigRevisionSummary {
  document?: unknown
  runtimeConfigYAML?: string
  metadata?: Record<string, unknown>
  message?: string
}

export interface SaveConfigRevisionDraftRequest {
  id?: string
  parentRevisionId?: string
  source?: string
  summary?: string
  dslSource?: string
  document?: unknown
  runtimeConfigYAML?: string
  metadata?: Record<string, unknown>
}

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.text()
  if (!body) {
    return `HTTP ${response.status}: ${response.statusText}`
  }

  try {
    const parsed = JSON.parse(body) as { error?: string; message?: string }
    return parsed.message || parsed.error || body
  } catch {
    return body
  }
}

async function readJSON<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readErrorMessage(response))
  }

  return response.json()
}

async function postRevisionJSON<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  return readJSON<T>(response)
}

function revisionHistoryTimestamp(revision: ConfigRevisionSummary): string {
  return revision.activatedAt || revision.lastDeployedAt || revision.updatedAt || revision.createdAt
}

function isRevisionHistoryEntry(revision: ConfigRevisionSummary): boolean {
  return Boolean(revision.activatedAt) || ['active', 'superseded', 'rolled_back'].includes(revision.status)
}

export function formatConfigRevisionLabel(id: string | undefined): string {
  if (!id) {
    return 'unknown revision'
  }
  return id.length <= 12 ? id : id.slice(0, 8)
}

export function revisionToConfigVersion(revision: ConfigRevisionSummary): ConfigVersion {
  return {
    id: revision.id,
    version: revision.id,
    timestamp: revisionHistoryTimestamp(revision),
    source: revision.source || 'revision',
    filename: revision.summary || revision.id,
    parentRevisionId: revision.parentRevisionId,
    status: revision.status,
    summary: revision.summary,
    createdBy: revision.createdBy,
    runtimeTarget: revision.runtimeTarget,
    lastDeployStatus: revision.lastDeployStatus,
    lastDeployMessage: revision.lastDeployMessage,
    activatedAt: revision.activatedAt,
  }
}

export async function listConfigRevisions(): Promise<ConfigRevisionSummary[]> {
  const response = await fetch(REVISION_API_BASE)
  return readJSON<ConfigRevisionSummary[]>(response)
}

export async function listConfigVersionHistory(): Promise<ConfigVersion[]> {
  const revisions = await listConfigRevisions()
  return revisions.filter(isRevisionHistoryEntry).map(revisionToConfigVersion)
}

export async function saveConfigRevisionDraft(
  request: SaveConfigRevisionDraftRequest,
): Promise<ConfigRevisionDetail> {
  return postRevisionJSON<ConfigRevisionDetail>(`${REVISION_API_BASE}/draft`, request)
}

export async function validateConfigRevision(id: string): Promise<ConfigRevisionDetail> {
  return postRevisionJSON<ConfigRevisionDetail>(`${REVISION_API_BASE}/validate`, { id })
}

export async function activateConfigRevision(id: string): Promise<ConfigRevisionDetail> {
  return postRevisionJSON<ConfigRevisionDetail>(`${REVISION_API_BASE}/activate`, { id })
}

export async function createAndActivateConfigRevision(
  request: SaveConfigRevisionDraftRequest,
): Promise<ConfigRevisionDetail> {
  const draft = await saveConfigRevisionDraft(request)
  if (!draft.id) {
    throw new Error('Revision draft response did not include an id')
  }
  await validateConfigRevision(draft.id)
  return activateConfigRevision(draft.id)
}
