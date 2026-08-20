import type { RecipeConfig } from '../pages/configPageSupport'

export interface RecipeDraft extends RecipeConfig {
  createdAt?: string
  updatedAt?: string
}

const errorMessage = async (response: Response) =>
  (await response.text()) || `Request failed (${response.status})`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const recipeDraftApi = {
  list: () => request<{ items: RecipeDraft[] }>('/api/recipe-drafts'),
  save: (draft: RecipeConfig) =>
    request<RecipeDraft>(`/api/recipe-drafts/${encodeURIComponent(draft.name)}`, {
      method: 'PUT',
      body: JSON.stringify(draft),
    }),
  remove: (name: string) =>
    request<void>(`/api/recipe-drafts/${encodeURIComponent(name)}`, { method: 'DELETE' }),
}

export const announceRecipeDraftChange = () =>
  window.dispatchEvent(new CustomEvent('recipe-drafts-changed'))
