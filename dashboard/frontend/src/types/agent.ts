/**
 * Agent Playground - Type Definitions
 * Types for agentic workflow visualization and execution
 */

import type { ToolCall, ToolResult } from '../tools/types'

// ========== Step Types ==========

/**
 * Agent step types representing the ReAct pattern
 */
export type AgentStepType = 'thought' | 'action' | 'observation' | 'final_answer'

/**
 * Step execution status
 */
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed'

/**
 * Action data for agent steps
 */
export interface AgentAction {
  /** Action type - usually 'function' for tool calls */
  type: string
  /** Tool/function name being called */
  name: string
  /** Arguments passed to the tool */
  arguments: Record<string, unknown>
  /** Code representation (for code-based agents) */
  code?: string
}

/**
 * Observation data from tool execution
 */
export interface AgentObservation {
  /** Result content from tool execution */
  content: unknown
  /** Error message if execution failed */
  error?: string
  /** Screenshot data (base64) for computer-use agents */
  screenshot?: string
  /** Execution duration in milliseconds */
  duration_ms: number
}

/**
 * Individual step in the agent's reasoning process
 */
export interface AgentStep {
  /** Unique step identifier */
  id: string
  /** Step index (0-based) */
  index: number
  /** Timestamp when step was created */
  timestamp: number
  /** Type of step */
  type: AgentStepType
  /** Current execution status */
  status: StepStatus
  /** Thought content (for 'thought' type) */
  thought?: string
  /** Action data (for 'action' type) */
  action?: AgentAction
  /** Observation data (for 'observation' type) */
  observation?: AgentObservation
  /** Final answer content (for 'final_answer' type) */
  answer?: string
  /** Original tool call data (for action steps) */
  toolCall?: ToolCall
  /** Tool result data (for observation steps) */
  toolResult?: ToolResult
}

// ========== Session Types ==========

/**
 * Agent operation modes
 */
export type AgentMode = 'chat' | 'code' | 'computer_use'

/**
 * Session execution status
 */
export type SessionStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled'

/**
 * Agent session representing a complete task execution
 */
export interface AgentSession {
  /** Unique session identifier */
  id: string
  /** Agent operation mode */
  mode: AgentMode
  /** Current session status */
  status: SessionStatus
  /** The task/query being executed */
  task: string
  /** List of steps in the session */
  steps: AgentStep[]
  /** Model used for the session */
  model_used: string
  /** Session creation timestamp */
  created_at: number
  /** Session completion timestamp */
  completed_at?: number
  /** Error message if session failed */
  error?: string
  /** Maximum steps allowed */
  max_steps: number
  /** Current step count */
  step_count: number
  /** Final response content */
  final_response?: string
}

// ========== Annotation Types ==========

/**
 * Rating values for step annotations
 */
export type AnnotationRating = 'good' | 'bad' | 'neutral'

/**
 * Annotation for a specific step
 */
export interface StepAnnotation {
  /** Unique annotation identifier */
  id: string
  /** Session ID this annotation belongs to */
  session_id: string
  /** Step ID being annotated */
  step_id: string
  /** Rating value */
  rating: AnnotationRating
  /** Optional feedback text */
  feedback?: string
  /** Timestamp when annotation was created */
  created_at: number
}

// ========== API Types ==========

/**
 * Request to create a new agent session
 */
export interface CreateSessionRequest {
  /** The task/query to execute */
  task: string
  /** Agent mode (default: 'chat') */
  mode?: AgentMode
  /** Model to use (optional, uses router default) */
  model?: string
  /** Maximum steps allowed (default: 30) */
  max_steps?: number
  /** System prompt override */
  system_prompt?: string
  /** Enable web search tools */
  enable_web_search?: boolean
  /** Additional tools to enable */
  enabled_tools?: string[]
}

/**
 * Response from session creation
 */
export interface CreateSessionResponse {
  /** Created session ID */
  session_id: string
  /** Stream URL for SSE updates */
  stream_url: string
}

/**
 * SSE event types from the agent stream
 */
export type AgentEventType =
  | 'session_started'
  | 'step_started'
  | 'step_updated'
  | 'step_completed'
  | 'session_completed'
  | 'session_failed'
  | 'session_cancelled'

/**
 * SSE event data from agent stream
 */
export interface AgentStreamEvent {
  /** Event type */
  type: AgentEventType
  /** Session ID */
  session_id: string
  /** Step data (for step events) */
  step?: AgentStep
  /** Session data (for session events) */
  session?: Partial<AgentSession>
  /** Error message (for failure events) */
  error?: string
  /** Timestamp */
  timestamp: number
}

// ========== UI State Types ==========

/**
 * Visualization mode for the agent playground
 */
export type VisualizationMode = 'steps' | 'direct'

/**
 * Agent playground settings
 */
export interface AgentPlaygroundSettings {
  /** Current visualization mode */
  visualizationMode: VisualizationMode
  /** Agent operation mode */
  agentMode: AgentMode
  /** Selected model */
  model: string
  /** Maximum steps allowed */
  maxSteps: number
  /** Enable web search */
  enableWebSearch: boolean
  /** System prompt */
  systemPrompt: string
  /** Auto-scroll to latest step */
  autoScroll: boolean
}

/**
 * Default agent playground settings
 */
export const DEFAULT_AGENT_SETTINGS: AgentPlaygroundSettings = {
  visualizationMode: 'steps',
  agentMode: 'chat',
  model: '',
  maxSteps: 30,
  enableWebSearch: true,
  systemPrompt: '',
  autoScroll: true,
}

// ========== Step Visualization Types ==========

/**
 * Color theme for step types
 */
export interface StepTheme {
  /** Primary color */
  primary: string
  /** Background color */
  background: string
  /** Border color */
  border: string
  /** Icon name */
  icon: string
}

/**
 * Step type to theme mapping
 */
export const STEP_THEMES: Record<AgentStepType, StepTheme> = {
  thought: {
    primary: '#9333ea',    // Purple
    background: '#faf5ff',
    border: '#e9d5ff',
    icon: 'lightbulb',
  },
  action: {
    primary: '#2563eb',    // Blue
    background: '#eff6ff',
    border: '#bfdbfe',
    icon: 'wrench',
  },
  observation: {
    primary: '#16a34a',    // Green
    background: '#f0fdf4',
    border: '#bbf7d0',
    icon: 'eye',
  },
  final_answer: {
    primary: '#ea580c',    // Orange
    background: '#fff7ed',
    border: '#fed7aa',
    icon: 'check-circle',
  },
}

/**
 * Status to color mapping
 */
export const STATUS_COLORS: Record<StepStatus, string> = {
  pending: '#9ca3af',     // Gray
  running: '#2563eb',     // Blue
  completed: '#16a34a',   // Green
  failed: '#dc2626',      // Red
}

// ========== Helper Functions ==========

/**
 * Generate a unique step ID
 */
export function generateStepId(): string {
  return `step_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Generate a unique session ID
 */
export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Create a new agent step
 */
export function createStep(
  type: AgentStepType,
  index: number,
  data?: Partial<AgentStep>
): AgentStep {
  return {
    id: generateStepId(),
    index,
    timestamp: Date.now(),
    type,
    status: 'pending',
    ...data,
  }
}

/**
 * Create a new agent session
 */
export function createSession(
  task: string,
  mode: AgentMode = 'chat',
  model: string = ''
): AgentSession {
  return {
    id: generateSessionId(),
    mode,
    status: 'idle',
    task,
    steps: [],
    model_used: model,
    created_at: Date.now(),
    max_steps: 30,
    step_count: 0,
  }
}

/**
 * Check if a step is in a terminal state
 */
export function isStepTerminal(status: StepStatus): boolean {
  return status === 'completed' || status === 'failed'
}

/**
 * Check if a session is in a terminal state
 */
export function isSessionTerminal(status: SessionStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}

/**
 * Format step duration for display
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`
  }
  return `${(ms / 1000).toFixed(1)}s`
}

/**
 * Get step type display name
 */
export function getStepTypeName(type: AgentStepType): string {
  const names: Record<AgentStepType, string> = {
    thought: 'Thought',
    action: 'Action',
    observation: 'Observation',
    final_answer: 'Final Answer',
  }
  return names[type]
}
