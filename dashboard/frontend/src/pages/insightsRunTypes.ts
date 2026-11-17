// The Replay run API: GET /replays/runs lists runs, GET /replays/run?session_id= reads one.

export interface RunCurrencyCost {
  currency: string
  total: number
  router_call_spend: number
}

export interface RunCost {
  by_currency?: RunCurrencyCost[]
  unpriced_records: number
  unpriced_router_calls: number
  prompt_tokens: number
  completion_tokens: number
}

export interface RunAgent {
  label: string
  id?: string
  parent?: string
  kind?: string
  calls: number
  first_call_at: string
  last_call_at: string
  spawned_after_record_id?: string
  // What the spawning tool call asked for, the parent's call that took the
  // agent's result, and how it came back: succeeded or failed.
  task?: string
  returned_to_record_id?: string
  return_status?: string
  // The tools the agent's first turn offered the model.
  offered_tools?: string[]
  models?: string[]
  cost: RunCost
}

export interface RunReportedOutcome {
  source: string
  target: string
  verdict: string
  reason?: string
  timestamp?: string
}

export interface RunOutcome {
  status: 'answered' | 'failed' | 'aborted' | 'in_progress' | 'unknown' | string
  final_answer_record_id?: string
  tool_results: number
  tool_failures: number
  reported?: { record_id: string; outcome: RunReportedOutcome }[]
}

export interface RunSummary {
  session_id: string
  // The name the harness gave the session: its title, or the user's rename.
  session_name?: string
  harness?: string
  prompt?: string
  started_at: string
  ended_at: string
  duration_ms: number
  calls: number
  agents: RunAgent[]
  models: { name: string; value: number }[]
  outcome: RunOutcome
  cost: RunCost
  events?: Record<string, number>
}

export interface RunRouterCall {
  model?: string
  kind: string
  calls: number
  cost?: number
  currency?: string
  unpriced_calls?: number
}

// A tool call of a call's turn: its name, its arguments when Replay captured
// them, and its result's status once the agent's next request carried it.
export interface RunToolCall {
  name: string
  id?: string
  summary?: string
  arguments?: string
  status?: 'succeeded' | 'failed' | 'denied' | string
  // gateway: from the end of the response that asked for the call to the
  // start of the next request that carried its result, shared by the calls
  // of one response; plugin: as a harness plugin reported it.
  timing_source?: 'gateway' | 'plugin' | string
  requested_at?: string
  returned_at?: string
  client_ms?: number
  started_at?: string
  ended_at?: string
  // plugin: when the harness put the call to approval, and the wait from
  // there to the start where it reports a start after the answer.
  approval_requested_at?: string
  approval_ms?: number
}

export interface RunCall {
  record_id: string
  request_id?: string
  agent: string
  timestamp: string
  duration_ms: number
  recipe?: string
  entrypoint?: string
  decision?: string
  signals?: string[]
  model?: string
  previous_model?: string
  effort?: string
  selection_method?: string
  session_action?: string
  reason?: string
  request_kind?: string
  lifecycle_state: string
  response_status?: number
  terminal_reason?: string
  prompt_tokens?: number
  completion_tokens?: number
  cost?: number
  currency?: string
  // catalog_list_price when the cost is the catalog's list price estimate.
  pricing_source?: string
  router_calls?: RunRouterCall[]
  looper_attempts?: number
  tool_results?: number
  tool_failures?: number
  final_answer?: boolean
  outcomes?: RunReportedOutcome[]
  markers?: string[]
  tool_calls?: RunToolCall[]
  offered_tools_added?: string[]
  offered_tools_removed?: string[]
  hops?: RunCall[]
}

export interface RunResponse {
  object: string
  run: RunSummary
  calls: RunCall[]
}

export interface RunListResponse {
  object: string
  count: number
  total: number
  limit: number
  offset: number
  has_more: boolean
  next_offset?: number
  data: RunSummary[]
}
