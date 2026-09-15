import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('Dashboard capability settings fail closed', () => {
  it('restores settings for HttpOnly-cookie sessions authenticated by user state', () => {
    const source = readFileSync(new URL('./ReadonlyContext.tsx', import.meta.url), 'utf8')

    expect(source).toContain('const { isAuthenticated } = useAuth()')
    expect(source).toContain('if (!isAuthenticated)')
    expect(source).toContain('}, [isAuthenticated])')
    expect(source).not.toContain('const { token } = useAuth()')
  })

  it('closes every mutation capability before each fetch and leaves failures closed', () => {
    const source = readFileSync(new URL('./ReadonlyContext.tsx', import.meta.url), 'utf8')
    const fetchStart = source.indexOf("const response = await fetch('/api/settings'")

    expect(source).toContain('const [isReadonly, setIsReadonly] = useState(true)')
    expect(source).toContain('const [serverReadonly, setServerReadonly] = useState(true)')
    expect(source).toContain(
      'const [runtimeConfigWritable, setRuntimeConfigWritable] = useState(false)',
    )
    expect(source).toContain(
      'const [recipeStoreWritable, setRecipeStoreWritable] = useState(false)',
    )
    for (const reset of [
      'setIsReadonly(true)',
      'setServerReadonly(true)',
      'setRuntimeConfigWritable(false)',
      'setRecipeStoreWritable(false)',
    ]) {
      expect(source.indexOf(reset)).toBeGreaterThan(-1)
      expect(source.indexOf(reset)).toBeLessThan(fetchStart)
    }
    expect(source).toContain('decodeDashboardSettings(await response.json())')
    expect(source).not.toContain('Older Dashboard responses')
    expect(source).not.toContain('!effectiveReadonly')
    expect(source).toContain('if (controller.signal.aborted) return')
    expect(source).toContain('return () => controller.abort()')
  })
})
