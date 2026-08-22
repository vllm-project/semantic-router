import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('landing page product story', () => {
  it('presents the restored Mixture-of-Models story with the new workload promise', () => {
    const source = readSource('./LandingPage.tsx')

    expect(source).toContain('Build your')
    expect(source).toContain('Mixture-of-Models.')
    expect(source).toContain('Compose heterogeneous LLMs into personalized model paths.')
    expect(source).toContain('Match workload with right model on right hardware')
    expect(source).toContain('Understand every request')
    expect(source).toContain('Make preference executable')
    expect(source).toContain('Compose the model path')
  })

  it('keeps the restored page rooted in a full-height canvas', () => {
    const styles = readSource('./LandingPage.module.css')

    expect(styles).toMatch(/\.container\s*{[^}]*min-height: 100vh;[^}]*overflow-x: hidden;/s)
  })

  it('uses the red-black ambient motion behind the restored three-step story', () => {
    const page = readSource('./LandingPage.tsx')
    const styles = readSource('./LandingPage.module.css')

    expect(page).toContain('ColorBends')
    expect(page).toContain('DASHBOARD_MOTION_COLORS')
    expect(page).toContain('DASHBOARD_COLOR_BENDS_MOTION')
    expect(page).toContain('landing-motion-background')
    expect(page.match(/className=\{styles\.routingStep\}/g)).toHaveLength(3)
    expect(styles).toMatch(/\.backgroundEffect\s*{[^}]*position: fixed;[^}]*opacity: 0\.85;/s)
  })
})
