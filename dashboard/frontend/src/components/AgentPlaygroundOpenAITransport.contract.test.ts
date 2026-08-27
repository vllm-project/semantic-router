import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8')

describe('Playground inference transport contract', () => {
  it('uses one durable session kernel for Chat and Builder without a third Agent mode', () => {
    const playground = readSource('./AgentPlayground.tsx')
    const menu = readSource('./AgentComposerMenu.tsx')
    const timeline = readSource('./AgentTimeline.tsx')

    expect(playground).toContain('mode: agentSessionMode(activeMode)')
    expect(playground).toContain('await runtime.sendTurn(content, sessionId)')
    expect(playground).not.toContain('useOpenAIPlaygroundRuntime')
    expect(playground).not.toContain("activeMode === 'agent'")
    expect(playground).not.toContain('fallbackHistory:')
    expect(menu).not.toContain('<strong>Agent</strong>')
    expect(menu).toContain('<strong>Builder</strong>')
    expect(timeline).toContain('<AgentRouterMetadata')
  })
})
