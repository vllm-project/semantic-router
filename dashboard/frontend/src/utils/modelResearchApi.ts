import type {
  ModelResearchCampaign,
  ModelResearchCreateRequest,
  ModelResearchEvent,
  ModelResearchRecipesResponse,
} from '../types/modelResearch'

const API_BASE = '/api/model-research'

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status}: ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export async function getRecipes(): Promise<ModelResearchRecipesResponse> {
  const response = await fetch(`${API_BASE}/recipes`)
  return handleResponse<ModelResearchRecipesResponse>(response)
}

export async function listCampaigns(): Promise<ModelResearchCampaign[]> {
  const response = await fetch(`${API_BASE}/campaigns`)
  return handleResponse<ModelResearchCampaign[]>(response)
}

export async function getCampaign(campaignId: string): Promise<ModelResearchCampaign> {
  const response = await fetch(`${API_BASE}/campaigns/${campaignId}`)
  return handleResponse<ModelResearchCampaign>(response)
}

export async function createCampaign(
  payload: ModelResearchCreateRequest
): Promise<ModelResearchCampaign> {
  const response = await fetch(`${API_BASE}/campaigns`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse<ModelResearchCampaign>(response)
}

export async function stopCampaign(campaignId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/campaigns/${campaignId}/stop`, {
    method: 'POST',
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status}: ${response.statusText}`)
  }
}

export function subscribeToCampaignEvents(
  campaignId: string,
  onEvent: (event: ModelResearchEvent) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): () => void {
  let cancelled = false
  let currentSource: EventSource | null = null
  let retries = 0
  const maxRetries = 10

  const connect = () => {
    if (cancelled) return

    const source = new EventSource(`${API_BASE}/campaigns/${campaignId}/events`)
    currentSource = source

    source.addEventListener('connected', () => {
      retries = 0
    })

    source.addEventListener('event', (event) => {
      try {
        onEvent(JSON.parse(event.data) as ModelResearchEvent)
      } catch (error) {
        console.error('Failed to parse model research event', error)
      }
    })

    source.addEventListener('completed', () => {
      source.close()
      currentSource = null
      onComplete()
    })

    source.onerror = () => {
      source.close()
      currentSource = null
      if (cancelled) return

      retries += 1
      if (retries > maxRetries) {
        onError(new Error('Model research event stream disconnected'))
        return
      }

      const delay = Math.min(1000 * Math.pow(2, retries - 1), 15000)
      window.setTimeout(connect, delay)
    }
  }

  connect()

  return () => {
    cancelled = true
    if (currentSource) {
      currentSource.close()
      currentSource = null
    }
  }
}

