/**
 * Agent API - API functions for agent playground
 */

import type {
  AgentSession,
  AgentStreamEvent,
  CreateSessionRequest,
  CreateSessionResponse,
  StepAnnotation,
  AnnotationRating,
} from '../types/agent'

// Base URL for API calls - empty for same-origin
const API_BASE = ''

/**
 * Create a new agent session
 */
export async function createSession(request: CreateSessionRequest): Promise<CreateSessionResponse> {
  const response = await fetch(`${API_BASE}/api/agent/sessions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || 'Failed to create session')
  }

  return response.json()
}

/**
 * Get session details
 */
export async function getSession(sessionId: string): Promise<AgentSession> {
  const response = await fetch(`${API_BASE}/api/agent/sessions/${sessionId}`)

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || 'Failed to get session')
  }

  return response.json()
}

/**
 * Cancel a running session
 */
export async function cancelSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/agent/sessions/${sessionId}`, {
    method: 'DELETE',
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || 'Failed to cancel session')
  }
}

/**
 * Subscribe to session events via SSE
 * Returns cleanup function
 */
export function subscribeToSession(
  sessionId: string,
  onEvent: (event: AgentStreamEvent) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/api/agent/sessions/${sessionId}/stream`)

  eventSource.onopen = () => {
    console.log('[AgentStream] Connected to session:', sessionId)
  }

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as AgentStreamEvent
      onEvent(data)

      // Check for terminal events
      if (
        data.type === 'session_completed' ||
        data.type === 'session_failed' ||
        data.type === 'session_cancelled'
      ) {
        eventSource.close()
        onComplete()
      }
    } catch (err) {
      console.error('[AgentStream] Failed to parse event:', err)
    }
  }

  eventSource.onerror = (err) => {
    console.error('[AgentStream] Error:', err)
    eventSource.close()
    onError(new Error('Connection lost'))
  }

  // Return cleanup function
  return () => {
    eventSource.close()
  }
}

/**
 * Submit annotation for a step
 */
export async function submitAnnotation(
  sessionId: string,
  stepId: string,
  rating: AnnotationRating,
  feedback?: string
): Promise<StepAnnotation> {
  const response = await fetch(`${API_BASE}/api/agent/annotations`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      step_id: stepId,
      rating,
      feedback,
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || 'Failed to submit annotation')
  }

  return response.json()
}

/**
 * Get annotations for a session
 */
export async function getAnnotations(sessionId: string): Promise<StepAnnotation[]> {
  const response = await fetch(`${API_BASE}/api/agent/sessions/${sessionId}/annotations`)

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || 'Failed to get annotations')
  }

  return response.json()
}

/**
 * Export session data
 */
export async function exportSession(
  sessionId: string,
  format: 'json' | 'jsonl' = 'json'
): Promise<Blob> {
  const response = await fetch(
    `${API_BASE}/api/agent/sessions/${sessionId}/export?format=${format}`
  )

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || 'Failed to export session')
  }

  return response.blob()
}

/**
 * List recent sessions
 */
export async function listSessions(limit = 10): Promise<AgentSession[]> {
  const response = await fetch(`${API_BASE}/api/agent/sessions?limit=${limit}`)

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || 'Failed to list sessions')
  }

  return response.json()
}
