export interface DashboardSettings {
  readonlyMode: boolean
  serverReadonly: boolean
  runtimeConfigWritable: boolean
  recipeStoreWritable: boolean
  setupMode: boolean
  platform: string
  envoyUrl: string
  routerEvalEndpoint: string
  evaluationAvailable: boolean
  evaluationUnavailableReason: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/** Decode the same-version backend contract without inferring capabilities. */
export const decodeDashboardSettings = (value: unknown): DashboardSettings => {
  if (!isRecord(value)) throw new Error('Dashboard settings must be an object')

  const booleanFields = [
    'readonlyMode',
    'serverReadonly',
    'runtimeConfigWritable',
    'recipeStoreWritable',
    'setupMode',
    'evaluationAvailable',
  ] as const
  const stringFields = [
    'platform',
    'envoyUrl',
    'routerEvalEndpoint',
    'evaluationUnavailableReason',
  ] as const

  for (const field of booleanFields) {
    if (typeof value[field] !== 'boolean') {
      throw new Error(`Dashboard settings field ${field} must be boolean`)
    }
  }
  for (const field of stringFields) {
    if (typeof value[field] !== 'string') {
      throw new Error(`Dashboard settings field ${field} must be string`)
    }
  }

  return {
    readonlyMode: value.readonlyMode as boolean,
    serverReadonly: value.serverReadonly as boolean,
    runtimeConfigWritable: value.runtimeConfigWritable as boolean,
    recipeStoreWritable: value.recipeStoreWritable as boolean,
    setupMode: value.setupMode as boolean,
    platform: value.platform as string,
    envoyUrl: value.envoyUrl as string,
    routerEvalEndpoint: value.routerEvalEndpoint as string,
    evaluationAvailable: value.evaluationAvailable as boolean,
    evaluationUnavailableReason: value.evaluationUnavailableReason as string,
  }
}
