/**
 * Agent Hooks - Custom hooks for agent playground
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  AgentSession,
  AgentStep,
  AgentStreamEvent,
  CreateSessionRequest,
  SessionStatus,
  AgentPlaygroundSettings,
} from '../types/agent'
import {
  createSession as createSessionFromType,
  isSessionTerminal,
} from '../types/agent'
import * as api from '../utils/agentApi'

/**
 * Hook for managing agent session state
 */
export function useAgentSession() {
  const [session, setSession] = useState<AgentSession | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Start a new session
  const startSession = useCallback(async (request: CreateSessionRequest): Promise<string | null> => {
    setLoading(true)
    setError(null)

    // Create local session first for immediate UI feedback
    const localSession = createSessionFromType(
      request.task,
      request.mode || 'chat',
      request.model || ''
    )
    localSession.status = 'running'
    localSession.max_steps = request.max_steps || 30
    setSession(localSession)

    try {
      // Create session on backend
      const response = await api.createSession(request)

      // Update session with server-provided ID
      setSession(prev => prev ? { ...prev, id: response.session_id } : null)

      return response.session_id
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start session'
      setError(errorMessage)
      setSession(prev => prev ? { ...prev, status: 'failed', error: errorMessage } : null)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  // Cancel the current session
  const cancelSession = useCallback(async () => {
    if (!session || isSessionTerminal(session.status)) {
      return
    }

    // Abort any pending requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    try {
      if (session.id) {
        await api.cancelSession(session.id)
      }
      setSession(prev => prev ? { ...prev, status: 'cancelled' } : null)
    } catch (err) {
      console.error('Failed to cancel session:', err)
      // Still mark as cancelled locally
      setSession(prev => prev ? { ...prev, status: 'cancelled' } : null)
    }
  }, [session])

  // Add a step to the session
  const addStep = useCallback((step: AgentStep) => {
    setSession(prev => {
      if (!prev) return null
      return {
        ...prev,
        steps: [...prev.steps, step],
        step_count: prev.steps.length + 1,
      }
    })
  }, [])

  // Update a step in the session
  const updateStep = useCallback((stepId: string, updates: Partial<AgentStep>) => {
    setSession(prev => {
      if (!prev) return null
      return {
        ...prev,
        steps: prev.steps.map(step =>
          step.id === stepId ? { ...step, ...updates } : step
        ),
      }
    })
  }, [])

  // Update session status
  const updateStatus = useCallback((status: SessionStatus, error?: string) => {
    setSession(prev => {
      if (!prev) return null
      return {
        ...prev,
        status,
        error,
        completed_at: isSessionTerminal(status) ? Date.now() : prev.completed_at,
      }
    })
  }, [])

  // Set final response
  const setFinalResponse = useCallback((response: string) => {
    setSession(prev => {
      if (!prev) return null
      return {
        ...prev,
        final_response: response,
      }
    })
  }, [])

  // Reset session
  const resetSession = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setSession(null)
    setError(null)
    setLoading(false)
  }, [])

  // Clear error
  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    session,
    loading,
    error,
    startSession,
    cancelSession,
    addStep,
    updateStep,
    updateStatus,
    setFinalResponse,
    resetSession,
    clearError,
    abortControllerRef,
  }
}

/**
 * Hook for SSE streaming subscription
 */
export function useAgentStream(
  sessionId: string | null,
  enabled = true,
  onEvent?: (event: AgentStreamEvent) => void
) {
  const [connected, setConnected] = useState(false)
  const [completed, setCompleted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    if (!sessionId || !enabled) {
      return
    }

    const cleanup = api.subscribeToSession(
      sessionId,
      (event) => {
        setConnected(true)
        setError(null)
        if (onEvent) {
          onEvent(event)
        }
      },
      () => {
        setCompleted(true)
        setConnected(false)
      },
      (err) => {
        setError(err.message)
        setConnected(false)
      }
    )

    cleanupRef.current = cleanup

    return () => {
      cleanup()
      cleanupRef.current = null
    }
  }, [sessionId, enabled, onEvent])

  const disconnect = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current()
      cleanupRef.current = null
      setConnected(false)
    }
  }, [])

  return { connected, completed, error, disconnect }
}

/**
 * Hook for agent playground settings
 */
export function useAgentSettings(initialSettings?: Partial<AgentPlaygroundSettings>) {
  const [settings, setSettings] = useState<AgentPlaygroundSettings>({
    visualizationMode: 'steps',
    agentMode: 'chat',
    model: '',
    maxSteps: 30,
    enableWebSearch: true,
    systemPrompt: '',
    autoScroll: true,
    ...initialSettings,
  })

  const updateSetting = useCallback(<K extends keyof AgentPlaygroundSettings>(
    key: K,
    value: AgentPlaygroundSettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }, [])

  const toggleVisualization = useCallback(() => {
    setSettings(prev => ({
      ...prev,
      visualizationMode: prev.visualizationMode === 'steps' ? 'direct' : 'steps',
    }))
  }, [])

  const resetSettings = useCallback(() => {
    setSettings({
      visualizationMode: 'steps',
      agentMode: 'chat',
      model: '',
      maxSteps: 30,
      enableWebSearch: true,
      systemPrompt: '',
      autoScroll: true,
    })
  }, [])

  return {
    settings,
    updateSetting,
    toggleVisualization,
    resetSettings,
  }
}

/**
 * Hook for selected step state
 */
export function useSelectedStep() {
  const [selectedStep, setSelectedStep] = useState<AgentStep | null>(null)

  const selectStep = useCallback((step: AgentStep | null) => {
    setSelectedStep(step)
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedStep(null)
  }, [])

  return {
    selectedStep,
    selectStep,
    clearSelection,
  }
}

/**
 * Combined hook for full agent playground state
 * This provides a convenient way to access all agent-related state
 */
export function useAgentPlayground(initialSettings?: Partial<AgentPlaygroundSettings>) {
  const sessionHook = useAgentSession()
  const settingsHook = useAgentSettings(initialSettings)
  const selectionHook = useSelectedStep()

  // Stream subscription with event handling
  const handleStreamEvent = useCallback((event: AgentStreamEvent) => {
    switch (event.type) {
      case 'step_started':
      case 'step_updated':
        if (event.step) {
          // Check if step already exists
          const existingStep = sessionHook.session?.steps.find(s => s.id === event.step!.id)
          if (existingStep) {
            sessionHook.updateStep(event.step.id, event.step)
          } else {
            sessionHook.addStep(event.step)
          }
        }
        break
      case 'step_completed':
        if (event.step) {
          sessionHook.updateStep(event.step.id, { ...event.step, status: 'completed' })
        }
        break
      case 'session_completed':
        sessionHook.updateStatus('completed')
        if (event.session?.final_response) {
          sessionHook.setFinalResponse(event.session.final_response)
        }
        break
      case 'session_failed':
        sessionHook.updateStatus('failed', event.error)
        break
      case 'session_cancelled':
        sessionHook.updateStatus('cancelled')
        break
    }
  }, [sessionHook])

  const streamHook = useAgentStream(
    sessionHook.session?.id || null,
    sessionHook.session?.status === 'running',
    handleStreamEvent
  )

  // Start a new agent run
  const startAgent = useCallback(async (task: string) => {
    const { agentMode, model, maxSteps, enableWebSearch, systemPrompt } = settingsHook.settings

    selectionHook.clearSelection()

    const sessionId = await sessionHook.startSession({
      task,
      mode: agentMode,
      model: model || undefined,
      max_steps: maxSteps,
      enable_web_search: enableWebSearch,
      system_prompt: systemPrompt || undefined,
    })

    return sessionId
  }, [sessionHook, settingsHook.settings, selectionHook])

  // Stop the current agent run
  const stopAgent = useCallback(() => {
    sessionHook.cancelSession()
    streamHook.disconnect()
  }, [sessionHook, streamHook])

  // Reset everything
  const resetAgent = useCallback(() => {
    sessionHook.resetSession()
    selectionHook.clearSelection()
  }, [sessionHook, selectionHook])

  return {
    // Session state
    session: sessionHook.session,
    loading: sessionHook.loading,
    error: sessionHook.error,

    // Settings
    settings: settingsHook.settings,
    updateSetting: settingsHook.updateSetting,
    toggleVisualization: settingsHook.toggleVisualization,

    // Selection
    selectedStep: selectionHook.selectedStep,
    selectStep: selectionHook.selectStep,

    // Stream state
    connected: streamHook.connected,

    // Actions
    startAgent,
    stopAgent,
    resetAgent,
    clearError: sessionHook.clearError,

    // Low-level access for custom handling
    addStep: sessionHook.addStep,
    updateStep: sessionHook.updateStep,
    abortControllerRef: sessionHook.abortControllerRef,
  }
}
