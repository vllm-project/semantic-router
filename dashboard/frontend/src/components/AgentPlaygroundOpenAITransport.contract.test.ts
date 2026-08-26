import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8')

describe('Playground inference transport contract', () => {
  it('keeps Chat on public OpenAI streaming and uses Agent turns only for explicit tool modes', () => {
    const playground = readSource('./AgentPlayground.tsx')
    const menu = readSource('./AgentComposerMenu.tsx')
    const messages = readSource('../utils/playgroundInferenceMessages.ts')
    const timeline = readSource('./AgentTimeline.tsx')
    const transport = readSource('../utils/openAIChatCompletions.ts')

    expect(playground).toContain("if (activeMode !== 'chat')")
    expect(playground).toContain('mode: agentSessionMode(activeMode)')
    expect(playground).toContain('await runtime.sendTurn(content, sessionId)')
    expect(playground).toContain('sessionId = inference.createSession({')
    expect(playground).toContain('await inference.send({')
    expect(playground).not.toContain('fallbackHistory:')
    expect(transport).toContain("method: 'POST'")
    expect(transport).toContain('stream: true')
    expect(transport).toContain("Accept: 'text/event-stream'")
    expect(transport).toContain('Authorization: `Bearer ${accessToken}`')
    expect(transport).not.toContain('/management/v1/agent-sessions')
    expect(menu).toContain('<strong>Agent</strong>')
    expect(menu).toContain('Search the web and use tools')
    expect(menu).toContain('onAgentChange(!agentEnabled)')
    expect(messages).toContain('Router completed the stream without assistant text.')
    expect(timeline).toContain('<HeaderDisplay headers={message.metadata.headers} />')
    expect(timeline).toContain('<AgentRouterMetadata')
  })
})
