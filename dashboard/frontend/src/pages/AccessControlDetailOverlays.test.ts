import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('access detail relationship authority', () => {
  it('loads entity relationships independently from the current table page', () => {
    const detail = readSource('./AccessEntityDetail.tsx')
    const overlays = readSource('./AccessControlDetailOverlays.tsx')

    expect(detail).toContain('inferenceAccessApi.userMemberships(id)')
    expect(detail).toContain('inferenceAccessApi.teamMembers(id)')
    expect(detail).toContain('inferenceAccessApi.ownedKeys(kind, id)')
    expect(detail).toContain("loadMore('accessAssignments')")
    expect(detail).not.toContain('users.find(')
    expect(detail).not.toContain('teams.find(')
    expect(overlays).not.toContain('canManageSelfServiceKey')
  })

  it('resolves API key ownership and policies by canonical detail endpoints', () => {
    const detail = readSource('./APIKeyDetail.tsx')

    expect(detail).toContain('inferenceAccessApi.userSummary')
    expect(detail).toContain('inferenceAccessApi.teamSummary')
    expect(detail).toContain('inferenceAccessApi.groupSummary')
    expect(detail).toContain('inferenceAccessApi.budgetSummary')
    expect(detail).not.toContain('ownerLabel(')
    expect(detail).not.toContain('users.find(')
    expect(detail).not.toContain('teams.find(')
  })
})
