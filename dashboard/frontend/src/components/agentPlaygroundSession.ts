import type { AgentSessionInput, AgentSessionMode } from '../generated/managementApiContract'
import type { RouterModelOption } from '../utils/routerModelSelection'

interface BuildAgentSessionInputOptions {
  keyId: string
  mode: AgentSessionMode
  effectiveTeamId?: string
  model: RouterModelOption
  title: string
}

// An Agent session is immutable with respect to its inference authority. Keep
// the selected key, its team context, and its visible target in one request so
// later turns cannot drift to another key-scoped catalog.
export function buildAgentSessionInput({
  keyId,
  mode,
  effectiveTeamId,
  model,
  title,
}: BuildAgentSessionInputOptions): AgentSessionInput {
  return {
    keyId,
    mode,
    ...(effectiveTeamId ? { effectiveTeamId } : {}),
    target: { kind: model.kind === 'individual' ? 'model' : 'entrypoint', id: model.id },
    title,
  }
}
