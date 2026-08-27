import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Agent Playground presentation', () => {
  it('opens as a focused canvas with restrained composer focus styling', () => {
    const playground = readSource('./AgentPlayground.tsx')
    const modelHook = readSource('./usePlaygroundRoutingModel.ts')
    const timeline = readSource('./AgentTimeline.tsx')
    const queue = readSource('./AgentComposerQueue.tsx')
    const styles = readSource('./AgentPlayground.module.css')
    const emptyMarkStyles = styles.slice(
      styles.indexOf('.emptyMark {'),
      styles.indexOf('.emptyMark img'),
    )

    expect(playground).toContain('const [sidebarOpen, setSidebarOpen] = useState(false)')
    expect(playground).toContain("user?.name.trim() || user?.email.split('@')[0] || 'there'")
    expect(timeline).toContain('src="/vllm.png"')
    expect(timeline).toContain('<h1>Welcome, {userName}</h1>')
    expect(timeline).toContain('One prompt. The right model path.')
    expect(timeline).toContain('Describe the outcome. We’ll compose the model path.')
    expect(timeline).not.toContain('Put your models to work.')
    expect(playground).toContain('routing.refresh().catch(() => undefined)')
    expect(playground).toContain('Try again')
    expect(playground).toContain('const MAX_QUEUED_TURNS = 8')
    expect(playground).toContain('void submitTurn(next)')
    expect(queue).toContain('aria-label="Queued messages"')
    expect(emptyMarkStyles).toContain('width: 74px')
    expect(playground).not.toContain('topbarIdentity')
    expect(playground).not.toContain('canManageRouting')
    expect(modelHook).toContain('includeIndividualModels: true')
    expect(emptyMarkStyles).not.toContain('border:')
    expect(emptyMarkStyles).not.toContain('background:')
    expect(styles).toMatch(/\.composer:focus-within\s*{[^}]*border-color:[^}]*box-shadow:/s)
    expect(styles).toMatch(
      /\.composer textarea:focus,[\s\S]*?\.composer textarea:focus-visible\s*{[^}]*box-shadow:\s*none/,
    )
    expect(styles).toMatch(/\.composerQueue\s*{[^}]*max-height:\s*126px/s)
    expect(styles).toMatch(/\.composerQueue > header > span\s*{[^}]*font-size:\s*0\.58rem/s)
  })
})
