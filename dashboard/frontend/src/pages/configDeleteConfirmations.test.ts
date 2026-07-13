import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('configuration delete confirmation contracts', () => {
  it.each([
    ['./ConfigPageDecisionsSection.tsx', 'decisionDeletePending', 'decisionDeleteError'],
    ['./ConfigPageModelsSection.tsx', 'reasoningFamilyDeletePending', 'reasoningFamilyDeleteError'],
    ['./ConfigPageProjectionsSection.tsx', 'projectionDeletePending', 'projectionDeleteError'],
  ])('uses the shared confirmation flow in %s', (path, pendingState, errorState) => {
    const source = readSource(path)

    expect(source).toContain("import ConfirmDialog from '../components/ConfirmDialog'")
    expect(source).toContain('<ConfirmDialog')
    expect(source).toContain(pendingState)
    expect(source).toContain(errorState)
    expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
  })

  it('covers every projection surface with the same stateful delete flow', () => {
    const source = readSource('./ConfigPageProjectionsSection.tsx')

    expect(source).toContain("{ kind: 'partition', name: partition.name }")
    expect(source).toContain("{ kind: 'score', name: score.name }")
    expect(source).toContain("{ kind: 'mapping', name: mapping.name }")
    expect(source).toContain('confirmDeleteProjection')
  })

  it.each([
    ['./ConfigPageSignalsSection.tsx', 'signalsPendingDelete', 'confirmDeleteSignals'],
    ['./ConfigPageTaxonomyClassifiers.tsx', 'deleteTarget', 'confirmDelete'],
    ['./ConfigPageLegacyCategoriesSection.tsx', 'removeTarget', 'confirmRemoveModel'],
    ['../components/ClawRoomChat.tsx', 'roomPendingDelete', 'handleDeleteRoom'],
  ])('removes native browser confirmation from %s', (path, targetState, confirmAction) => {
    const source = readSource(path)

    expect(source).toContain('<ConfirmDialog')
    expect(source).toContain(targetState)
    expect(source).toContain(confirmAction)
    expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
    expect(source).not.toMatch(/\b(?:window\.)?alert\s*\(/)
  })
})
