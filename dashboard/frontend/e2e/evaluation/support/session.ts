import type { Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from '../../support/auth'

const evaluationUser = {
  id: 'user-eval-1',
  email: 'eval@example.com',
  name: 'Eval User',
  role: 'read',
  permissions: [
    'config.read',
    'evaluation.read',
    'evaluation.run',
    'evaluation.write',
    'logs.read',
    'topology.read',
  ],
}

type EvaluationSessionSettings = {
  readonlyMode?: boolean
  serverReadonly?: boolean
}

export async function mockEvaluationUserSession(
  page: Page,
  settings: EvaluationSessionSettings = {},
): Promise<void> {
  await mockAuthenticatedAppShell(page, {
    user: evaluationUser,
    settings: { readonlyMode: false, serverReadonly: false, ...settings },
  })
}
