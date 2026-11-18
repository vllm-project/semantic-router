import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('configuration delete confirmation contracts', () => {
  it.each([
    ['./ConfigPageDecisionsSection.tsx', 'pendingDelete', 'deleteError'],
    ['./ConfigPageEntrypointsRecipesSection.tsx', 'entrypointPendingDelete', 'deleteError'],
    ['./ConfigPageProjectionsSection.tsx', 'pendingDelete', 'deleteError'],
  ])('uses the shared confirmation flow in %s', (path, pendingState, errorState) => {
    const source = readSource(path)

    expect(source).toContain("import ConfirmDialog from '../components/ConfirmDialog'")
    expect(source).toContain('<ConfirmDialog')
    expect(source).toContain(pendingState)
    expect(source).toContain(errorState)
    expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
  })

  it('uses the dedicated bulk-safe model delete dialog', () => {
    const source = readSource('./ConfigPageModelsSection.tsx')

    expect(source).toContain("import ModelDeleteDialog from './ModelDeleteDialog'")
    expect(source).toContain('<ModelDeleteDialog')
    expect(source).toContain('modelsPendingDelete')
    expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
  })

  it('deletes every projection kind through the one confirmed removal', () => {
    const source = readSource('./ConfigPageProjectionsSection.tsx')

    expect(source).toContain('const confirmDelete = async () =>')
    expect(source).toContain(
      'removeProjection(scopedConfig, pendingDelete.kind, pendingDelete.name)',
    )
  })

  it.each([['./ConfigPageSignalsSection.tsx', 'pendingDelete', 'confirmDelete']])(
    'removes native browser confirmation from %s',
    (path, targetState, confirmAction) => {
      const source = readSource(path)

      expect(source).toContain('<ConfirmDialog')
      expect(source).toContain(targetState)
      expect(source).toContain(confirmAction)
      expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
      expect(source).not.toMatch(/\b(?:window\.)?alert\s*\(/)
    },
  )
})
