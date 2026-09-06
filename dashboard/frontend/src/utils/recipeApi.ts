import type {
  RecipeDescriptor,
  RecipeActivationConfirmation,
  RecipeActivationPlan,
  RecipePackageActivationResult,
  RecipePackageDeactivationResult,
  RecipePackageErrorCode,
  RecipePackageList,
  RecipePackageListFilters,
  RecipeProbeDetail,
  RecipeProbeListFilters,
  RecipeProbePage,
  RecipeProbeRunPlan,
  RecipeProbeValidationResult,
} from '../types/recipe'

export class RecipeApiError extends Error {
  readonly code?: RecipePackageErrorCode
  readonly status: number

  constructor(message: string, status: number, code?: RecipePackageErrorCode) {
    super(message)
    this.name = 'RecipeApiError'
    this.code = code
    this.status = status
  }
}

async function readJson<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T

  const fallback = `${response.status} ${response.statusText}`.trim()
  try {
    const body = (await response.json()) as { error?: RecipePackageErrorCode; message?: string }
    throw new RecipeApiError(body.message || body.error || fallback, response.status, body.error)
  } catch (error) {
    if (error instanceof SyntaxError) throw new RecipeApiError(fallback, response.status)
    throw error
  }
}

export function buildRecipeProbeQuery(filters: RecipeProbeListFilters): string {
  const query = new URLSearchParams()
  query.set('page', String(filters.page ?? 1))
  query.set('page_size', String(filters.pageSize ?? 50))
  if (filters.query?.trim()) query.set('q', filters.query.trim())
  if (filters.recipe) query.set('recipe', filters.recipe)
  if (filters.decision) query.set('decision', filters.decision)
  if (filters.tag) query.set('tag', filters.tag)
  if (filters.model) query.set('model', filters.model)
  if (filters.requestShape) query.set('shape', filters.requestShape)
  return query.toString()
}

export async function getActiveRecipe(signal?: AbortSignal): Promise<RecipeDescriptor> {
  const response = await fetch('/api/recipe', { signal })
  const payload = (await response.json()) as RecipeDescriptor | { error?: string; message?: string }
  if ('managed' in payload && typeof payload.managed === 'boolean') return payload
  const message = 'error' in payload ? payload.error : 'message' in payload ? payload.message : ''
  throw new Error(message || `${response.status} ${response.statusText}`.trim())
}

export function buildRecipePackageQuery(filters: RecipePackageListFilters): string {
  const query = new URLSearchParams()
  query.set('page', String(filters.page ?? 1))
  query.set('page_size', String(filters.pageSize ?? 25))
  if (filters.query?.trim()) query.set('q', filters.query.trim())
  return query.toString()
}

export async function listRecipePackages(
  filters: RecipePackageListFilters = {},
  signal?: AbortSignal,
): Promise<RecipePackageList> {
  const query = buildRecipePackageQuery(filters)
  return readJson<RecipePackageList>(await fetch(`/api/recipe/packages?${query}`, { signal }))
}

export async function activateRecipePackage(
  recipeDigest: string,
  acknowledgeWarnings: boolean,
  signal?: AbortSignal,
  confirmation?: RecipeActivationConfirmation,
): Promise<RecipePackageActivationResult> {
  return readJson<RecipePackageActivationResult>(
    await fetch('/api/recipe/activate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recipe_digest: recipeDigest,
        ...(acknowledgeWarnings ? { acknowledge_warnings: true } : {}),
        ...confirmation,
      }),
      signal,
    }),
  )
}

export async function deactivateRecipePackage(
  signal?: AbortSignal,
  confirmation?: RecipeActivationConfirmation,
): Promise<RecipePackageDeactivationResult> {
  return readJson<RecipePackageDeactivationResult>(
    await fetch('/api/recipe/deactivate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(confirmation ?? {}),
      signal,
    }),
  )
}

export async function previewRecipePackageActivation(
  recipeDigest: string,
  acknowledgeWarnings: boolean,
  signal?: AbortSignal,
): Promise<RecipeActivationPlan> {
  return readJson<RecipeActivationPlan>(
    await fetch('/api/recipe/activate/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recipe_digest: recipeDigest,
        ...(acknowledgeWarnings ? { acknowledge_warnings: true } : {}),
      }),
      signal,
    }),
  )
}

export async function previewRecipePackageDeactivation(
  signal?: AbortSignal,
): Promise<RecipeActivationPlan> {
  return readJson<RecipeActivationPlan>(
    await fetch('/api/recipe/deactivate/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
      signal,
    }),
  )
}

export async function listRecipeProbes(
  filters: RecipeProbeListFilters,
  signal?: AbortSignal,
): Promise<RecipeProbePage> {
  const query = buildRecipeProbeQuery(filters)
  return readJson<RecipeProbePage>(await fetch(`/api/recipe/probes?${query}`, { signal }))
}

function probePath(decision: string, variant: string): string {
  return `/api/recipe/probes/${encodeURIComponent(decision)}/${encodeURIComponent(variant)}`
}

export async function getRecipeProbe(
  decision: string,
  variant: string,
  signal?: AbortSignal,
): Promise<RecipeProbeDetail> {
  return readJson<RecipeProbeDetail>(await fetch(probePath(decision, variant), { signal }))
}

export async function validateRecipeProbe(
  decision: string,
  variant: string,
  recipeDigest: string,
  signal?: AbortSignal,
): Promise<RecipeProbeValidationResult> {
  const response = await fetch(`${probePath(decision, variant)}/validate`, {
    method: 'POST',
    headers: { 'If-Match': `"${recipeDigest}"` },
    signal,
  })
  const payload = (await response.json()) as
    | RecipeProbeValidationResult
    | { error?: string; message?: string }
  if ('passed' in payload && typeof payload.passed === 'boolean') return payload
  const message = 'error' in payload ? payload.error : 'message' in payload ? payload.message : ''
  throw new Error(message || `${response.status} ${response.statusText}`.trim())
}

export async function createRecipeProbeRunPlan(
  decision: string,
  variant: string,
  recipeDigest: string,
  signal?: AbortSignal,
): Promise<RecipeProbeRunPlan> {
  return readJson<RecipeProbeRunPlan>(
    await fetch(`${probePath(decision, variant)}/run-plan`, {
      method: 'POST',
      headers: { 'If-Match': `"${recipeDigest}"` },
      signal,
    }),
  )
}
