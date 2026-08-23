import { describe, expect, it } from 'vitest'
import type { AccessBudget, AccessTeam, AccessUser } from '../utils/inferenceAccessApi'
import { inheritedKeyBudget } from './AccessControlFormSupport'

const budgets: AccessBudget[] = [
  {
    id: 'team-budget',
    name: 'Team default',
    description: '',
    rules: [
      {
        metric: 'requests',
        algorithm: 'sliding_log',
        limit: '12',
        window: 'PT1M',
        accounting: 'request',
        enforcement: 'enforce',
      },
    ],
    enabled: true,
    assignmentCount: 1,
  },
  {
    id: 'user-budget',
    name: 'User default',
    description: '',
    rules: [
      {
        metric: 'cost',
        algorithm: 'sliding_log',
        limit: '20',
        window: 'PT8H',
        accounting: 'response_actual',
        enforcement: 'enforce',
      },
    ],
    enabled: true,
    assignmentCount: 1,
  },
]

const team: AccessTeam = {
  id: 'team-1',
  name: 'Team',
  description: '',
  status: 'active',
  members: [{ teamId: 'team-1', userId: 'user-1', role: 'member' }],
  accessGroupIds: ['models'],
  budgetId: 'team-budget',
}

const user: AccessUser = {
  id: 'user-1',
  name: 'Invited user',
  email: 'user@example.test',
  status: 'active',
  accessGroupIds: [],
  memberships: team.members,
}

describe('inheritedKeyBudget', () => {
  it('uses the Team default for an invited user key without a personal override', () => {
    expect(
      inheritedKeyBudget(
        { ownerId: user.id, contextTeamId: team.id },
        'user',
        [user],
        [team],
        budgets,
      ),
    ).toEqual({ budget: budgets[0], source: 'team' })
  })

  it('uses the User default before the Team default', () => {
    expect(
      inheritedKeyBudget(
        { ownerId: user.id, contextTeamId: team.id },
        'user',
        [{ ...user, budgetId: 'user-budget' }],
        [team],
        budgets,
      ),
    ).toEqual({ budget: budgets[1], source: 'user' })
  })
})
