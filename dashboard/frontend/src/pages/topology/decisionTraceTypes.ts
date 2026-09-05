// topology/decisionTraceTypes.ts - Decision eval trace types
//
// Mirrors the router's decision.TraceNode / decision.DecisionTrace JSON shape
// exactly (src/semantic-router/pkg/decision/trace.go) so the recursive
// structure round-trips without a second schema.

export interface TraceNode {
  node_type: 'leaf' | 'AND' | 'OR' | 'NOT' | 'fallback'
  signal_type?: string
  signal_name?: string
  label?: string
  matched: boolean
  confidence: number
  children?: TraceNode[]
}

export interface DecisionTrace {
  decision_name: string
  matched: boolean
  confidence: number
  root_trace: TraceNode | null
}
