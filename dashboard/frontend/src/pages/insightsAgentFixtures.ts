import type { InsightsRecord } from './insightsPageTypes'
import type { RunResponse } from './insightsRunTypes'

// Replay records as the live router recorded harness traffic (bodies and key
// hashes removed), and as current main records the same calls: with the
// agent ids, parents and kinds routerreplay/store/agent.go fills.

const base = {
  decision_tier: 0,
  decision_priority: 700,
  recipe: 'default',
  original_model: 'vllm-sr/norn-1',
  lifecycle_state: 'completed' as const,
  response_status: 200,
  terminal_reason: 'response_complete',
  streaming: true,
  currency: 'XXX',
}

export const CLAUDE_SESSION = 'key_redacted:49b948ac-48b9-4fd5-b4e8-ccb8f6fba483'
export const OPENCODE_SESSION = 'key_redacted:ses_e158e5346ffel2rgJH7CFjm2zB'
export const CODEX_SESSION = 'key_redacted:01a1ea71-6532-7fc1-9137-1fd7cf9e9851'

export const claudeSpawnCall: InsightsRecord = {
  ...base,
  id: '2ac8f36e420342320cf37693fc0f98b4',
  request_id: 'gate-8ce69e06e9d108e7',
  timestamp: '2026-11-17T16:57:32.958407Z',
  session_id: CLAUDE_SESSION,
  turn_index: 0,
  decision: 'agentic',
  selected_model: 'deepseek/deepseek-v4.1-flash',
  reasoning_mode: 'on',
  reasoning_effort: 'high',
  selection_method: 'static',
  signals: { conversation: ['has_tools'], decision: ['task:agentic', 'needs_tools'] },
  projections: ['effort_off'],
  tool_trace: {
    flow: 'User Query -> LLM Tool Call -> LLM Reasoning Complete',
    stage: 'LLM Reasoning Complete',
    tool_names: ['Agent'],
  },
  duration_ms: 1719,
  agent: { harness: 'claude_code' },
  prompt_tokens: 16738,
  completion_tokens: 299,
  total_tokens: 17037,
  actual_cost: 0.00386822,
  baseline_cost: 0.00866537,
  cost_savings: 0.00479715,
  baseline_model: 'zai/glm-5.3-flash',
  confidence_score: 0.9884917100836262,
  route_diagnostics: {
    session_policy_applied: true,
    session_action: 'establish_baseline',
    session_phase: 'user_turn',
    session_reason: 'new_session',
  },
}

/** The Explore subagent the spawn call started, as current main records it. */
export const claudeSubagentCall: InsightsRecord = {
  ...claudeSpawnCall,
  id: '659ab032af327c34ac59d20cf8e76e0e',
  request_id: 'gate-5b2c2b1d0a6e33f1',
  timestamp: '2026-11-17T16:57:39.160582Z',
  tool_trace: {
    flow: 'User Query -> LLM Tool Call -> Client Tool Result -> LLM Final Response',
    stage: 'LLM Final Response',
    tool_names: ['Read'],
  },
  duration_ms: 170,
  agent: { harness: 'claude_code', id: '3fa9c2d81e07' },
  prompt_tokens: 13629,
  completion_tokens: 3,
  actual_cost: 0.00008656,
  route_diagnostics: {
    session_policy_applied: true,
    session_action: 'stay',
    session_reason: 'session_scope_protect',
  },
}

export const codexCall: InsightsRecord = {
  ...base,
  id: '29b8dee429fdb6d032bb2ce776a9c09c',
  request_id: 'gate-2f1c09c4e1b7a3d0',
  timestamp: '2026-11-17T16:57:45.846672Z',
  session_id: CODEX_SESSION,
  turn_index: 1,
  decision: 'fast',
  selected_model: 'qwen/qwen3.8-27b',
  signals: { conversation: ['has_tools'], decision: ['task:chat'] },
  tool_trace: { flow: 'User Query -> LLM Final Response', stage: 'LLM Final Response' },
  duration_ms: 1395,
  agent: { harness: 'codex' },
  prompt_tokens: 10887,
  completion_tokens: 4,
  actual_cost: 0.0010927,
  route_diagnostics: {
    session_policy_applied: true,
    session_action: 'establish_baseline',
    session_reason: 'new_session',
  },
}

export const codexSubagentCall: InsightsRecord = {
  ...codexCall,
  id: 'c0d3c0d3c0d3c0d3c0d3c0d3c0d3c0d3',
  timestamp: '2026-11-17T16:57:47.112000Z',
  thread_id: '01a1ea71-6533-7aa0-8a11-55e0e5a1c9d2',
  agent: { harness: 'codex', id: '7b41e9a0c2f5', kind: 'collab_spawn' },
  tool_trace: { tool_names: ['shell', 'apply_patch'] },
}

export const openCodeCall: InsightsRecord = {
  ...base,
  id: 'ca5dd92369cd3cb672d69589d5397781',
  request_id: 'gate-91ce2ab0e1f04d77',
  timestamp: '2026-11-17T16:58:07.784714Z',
  session_id: OPENCODE_SESSION,
  turn_index: 0,
  decision: 'agentic',
  selected_model: 'deepseek/deepseek-v4.1-flash',
  signals: {
    conversation: ['has_tools', 'tool_loop', 'has_prior_answer'],
    decision: ['task:agentic', 'needs_tools'],
  },
  tool_trace: { tool_names: ['bash', 'task'] },
  duration_ms: 758,
  agent: { harness: 'opencode' },
  prompt_tokens: 10507,
  completion_tokens: 125,
  actual_cost: 0.00184864,
  route_diagnostics: {
    session_policy_applied: true,
    session_action: 'stay',
    session_reason: 'session_scope_protect',
  },
}

export const openCodeTitleCall: InsightsRecord = {
  ...base,
  id: '7e3d98e98d1beccb884a6f9ecd4cc001',
  request_id: 'gate-beb93dfbff83241e',
  timestamp: '2026-11-17T16:58:03.140786Z',
  request_kind: 'session_title',
  session_id: OPENCODE_SESSION,
  session_name: 'Subagent reports function name from calc.py',
  turn_index: 1,
  decision: 'housekeeping',
  decision_priority: 860,
  selected_model: 'deepseek/deepseek-v4.1-flash',
  signals: { agent: ['housekeeping'], decision: ['task:agentic', 'needs_tools'] },
  projections: ['effort_off'],
  duration_ms: 873,
  agent: { harness: 'opencode' },
  prompt_tokens: 580,
  completion_tokens: 153,
  actual_cost: 0.00078982,
  route_diagnostics: {
    session_policy_applied: true,
    session_action: 'establish_baseline',
    session_reason: 'new_session',
  },
}

/** A DeepSeek Harness call once the Router names it; the live router does not yet. */
export const deepSeekHarnessCall: InsightsRecord = {
  ...base,
  id: '0999a5bbaf8cbf97c5168d597a590d0d',
  timestamp: '2026-11-17T16:59:02.523455Z',
  session_id: 'cc-start-09a886187e53d62b',
  turn_index: 1,
  decision: 'fast',
  selected_model: 'qwen/qwen3.8-27b',
  signals: { conversation: ['has_tools'], decision: ['task:chat'] },
  duration_ms: 142,
  agent: { harness: 'deepseek_harness' },
  prompt_tokens: 5803,
  completion_tokens: 4,
  actual_cost: 0.0000355,
  route_diagnostics: { session_action: 'none', session_reason: 'missing_identity' },
}

/** A plain API call: no harness, a derived conversation session. */
export const apiClientCall: InsightsRecord = {
  ...deepSeekHarnessCall,
  id: 'b3193cafa59522a7313c9dd55090c4b0',
  session_id: 'cc-start-8d9e5baaa19fb2f5',
  agent: null,
  original_model: 'vllm-sr/auto',
  turn_index: 0,
}

export const HARNESS_FIXTURE_RECORDS: InsightsRecord[] = [
  deepSeekHarnessCall,
  openCodeCall,
  openCodeTitleCall,
  codexSubagentCall,
  codexCall,
  claudeSubagentCall,
  claudeSpawnCall,
  apiClientCall,
]

const cost = (total: number) => ({
  by_currency: [{ currency: 'XXX', total, router_call_spend: 0 }],
  unpriced_records: 0,
  unpriced_router_calls: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
})

/** The Claude Code run: the main agent and the Explore subagent it spawned. */
export const claudeRun: RunResponse = {
  object: 'router_replay.run',
  run: {
    session_id: CLAUDE_SESSION,
    session_name: 'Explain the retry loop in replay_store.go',
    harness: 'claude_code',
    started_at: '2026-11-17T16:57:32.958407Z',
    ended_at: '2026-11-17T16:57:41.387820Z',
    duration_ms: 8429,
    calls: 3,
    agents: [
      {
        label: 'main',
        calls: 2,
        first_call_at: '2026-11-17T16:57:32.958407Z',
        last_call_at: '2026-11-17T16:57:41.214820Z',
        models: ['deepseek/deepseek-v4.1-flash'],
        cost: { ...cost(0.00397414), prompt_tokens: 33980, completion_tokens: 305 },
      },
      {
        label: 'subagent 1',
        id: '3fa9c2d81e07',
        parent: 'main',
        calls: 1,
        first_call_at: '2026-11-17T16:57:39.160582Z',
        last_call_at: '2026-11-17T16:57:39.160582Z',
        spawned_after_record_id: claudeSpawnCall.id,
        models: ['deepseek/deepseek-v4.1-flash'],
        cost: { ...cost(0.00008656), prompt_tokens: 13629, completion_tokens: 3 },
      },
    ],
    models: [{ name: 'deepseek/deepseek-v4.1-flash', value: 3 }],
    outcome: {
      status: 'answered',
      final_answer_record_id: 'bf7f64ec7e0ca0a3c2ce66c879d33c49',
      tool_results: 2,
      tool_failures: 0,
    },
    cost: { ...cost(0.0040607), prompt_tokens: 47609, completion_tokens: 308 },
  },
  calls: [
    {
      record_id: claudeSpawnCall.id,
      agent: 'main',
      timestamp: claudeSpawnCall.timestamp,
      duration_ms: 1719,
      decision: 'agentic',
      model: 'deepseek/deepseek-v4.1-flash',
      session_action: 'establish_baseline',
      lifecycle_state: 'completed',
      response_status: 200,
      prompt_tokens: 16738,
      completion_tokens: 299,
      cost: 0.00386822,
      currency: 'XXX',
    },
    {
      record_id: claudeSubagentCall.id,
      agent: 'subagent 1',
      timestamp: claudeSubagentCall.timestamp,
      duration_ms: 170,
      decision: 'agentic',
      model: 'deepseek/deepseek-v4.1-flash',
      session_action: 'stay',
      lifecycle_state: 'completed',
      response_status: 200,
      prompt_tokens: 13629,
      completion_tokens: 3,
      cost: 0.00008656,
      currency: 'XXX',
      tool_results: 1,
    },
    {
      record_id: 'bf7f64ec7e0ca0a3c2ce66c879d33c49',
      agent: 'main',
      timestamp: '2026-11-17T16:57:41.214820Z',
      duration_ms: 173,
      decision: 'agentic',
      model: 'deepseek/deepseek-v4.1-flash',
      session_action: 'stay',
      lifecycle_state: 'completed',
      response_status: 200,
      prompt_tokens: 17242,
      completion_tokens: 6,
      cost: 0.00010592,
      currency: 'XXX',
      tool_results: 1,
      final_answer: true,
    },
  ],
}
