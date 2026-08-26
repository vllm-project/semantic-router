import { describe, expect, it, vi } from 'vitest'

import type { AgentSession } from '../generated/managementApiContract'
import type { AgentManagementApi } from '../utils/agentManagementApi'
import {
  deleteAgentSession,
  type AgentSessionLifecycleApi,
} from './useAgentSessionRuntime'

type AgentSessionDetail = Awaited<ReturnType<AgentManagementApi['getSession']>>

function detail(status: AgentSession['status'], revision: number): AgentSessionDetail {
  return {
    data: { status } as AgentSession,
    etag: `"agent:${revision}"`,
    status: 200,
    mediaType: 'application/vnd.vllm-semantic-router.management.v1+json',
  }
}

describe('deleteAgentSession', () => {
  it('closes an active session before deleting its next revision', async () => {
    const current = detail('active', 4)
    const closed = detail('closed', 5)
    const api = {
      getSession: vi.fn().mockResolvedValue(current),
      patchSession: vi.fn().mockResolvedValue(closed),
      deleteSession: vi.fn().mockResolvedValue(undefined),
    } as unknown as AgentSessionLifecycleApi

    await deleteAgentSession(api, 'session-1')

    expect(api.getSession).toHaveBeenCalledWith('session-1')
    expect(api.patchSession).toHaveBeenCalledWith(
      'session-1',
      { status: 'closed' },
      current.etag,
    )
    expect(api.deleteSession).toHaveBeenCalledWith('session-1', closed.etag)
  })

  it('deletes a closed session without repeating the transition', async () => {
    const current = detail('closed', 7)
    const api = {
      getSession: vi.fn().mockResolvedValue(current),
      patchSession: vi.fn(),
      deleteSession: vi.fn().mockResolvedValue(undefined),
    } as unknown as AgentSessionLifecycleApi

    await deleteAgentSession(api, 'session-2')

    expect(api.patchSession).not.toHaveBeenCalled()
    expect(api.deleteSession).toHaveBeenCalledWith('session-2', current.etag)
  })
})
