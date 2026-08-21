import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Config Builder dashboard experience', () => {
  it('uses a coordinated workspace layout without emoji controls', () => {
    const dashboard = readSource('./builderPageDashboardView.tsx')
    const detail = readSource('./builderPageEntityDetailView.tsx')
    const styles = readSource('./BuilderPage.module.css')

    expect(dashboard).toContain('Routing workspace')
    expect(dashboard).toContain('dashboardWorkspace')
    expect(dashboard).toContain('dashboardRail')
    expect(dashboard).toContain('<svg viewBox="0 0 20 20" aria-hidden="true">')
    expect(dashboard).not.toMatch(/[📐📝🤖]/u)
    expect(detail).not.toContain('🔍')
    expect(styles).toContain('grid-template-columns: minmax(0, 2.15fr) minmax(250px, 0.85fr);')
  })
})
