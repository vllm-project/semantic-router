import type {
  AccessAPIKey,
  AccessBudget,
  AccessTeam,
  AccessUser,
} from '../utils/inferenceAccessApi'
import type { AccessEditor } from './AccessControlPageSupport'

export const ACCESS_EDITOR_TITLES: Record<
  AccessEditor['kind'],
  { eyebrow: string; create: string; edit: string; description: string }
> = {
  user: {
    eyebrow: 'Identity',
    create: 'Add user',
    edit: 'Edit user',
    description: 'Set the personal model and quota policy that keys may inherit.',
  },
  team: {
    eyebrow: 'Identity',
    create: 'Create team',
    edit: 'Edit team',
    description: 'Choose Team defaults, then add members and their Team roles.',
  },
  key: {
    eyebrow: 'Credential',
    create: 'Create API key',
    edit: 'API key',
    description: 'Name it, choose one owner, and keep overrides optional.',
  },
  group: {
    eyebrow: 'Model policy',
    create: 'Create access group',
    edit: 'Edit access group',
    description: 'Create a reusable collection of visible models.',
  },
  budget: {
    eyebrow: 'Rate limit',
    create: 'Create budget',
    edit: 'Edit budget',
    description: 'Create a reusable quota for users, Teams, or API keys.',
  },
}

export function toLocalDateTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000)
  return local.toISOString().slice(0, 16)
}

export function inheritedKeyBudget(
  value: Partial<AccessAPIKey>,
  ownerType: 'user' | 'team',
  users: AccessUser[],
  teams: AccessTeam[],
  budgets: AccessBudget[],
): { budget?: AccessBudget; source?: 'user' | 'team' } {
  if (ownerType === 'user') {
    const userBudgetId = users.find((user) => user.id === value.ownerId)?.budgetId
    if (userBudgetId) {
      return { budget: budgets.find((budget) => budget.id === userBudgetId), source: 'user' }
    }
    const teamBudgetId = teams.find((team) => team.id === value.contextTeamId)?.budgetId
    if (teamBudgetId) {
      return { budget: budgets.find((budget) => budget.id === teamBudgetId), source: 'team' }
    }
    return {}
  }
  const teamBudgetId = teams.find((team) => team.id === value.ownerId)?.budgetId
  if (!teamBudgetId) return {}
  return { budget: budgets.find((budget) => budget.id === teamBudgetId), source: 'team' }
}
