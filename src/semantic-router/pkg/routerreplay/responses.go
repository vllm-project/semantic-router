// Package-level response contracts are shared by runtime serialization and
// management OpenAPI generation.
package routerreplay

import "time"

type ListResponse struct {
	Object     string          `json:"object"`
	Count      int             `json:"count"`
	Total      int             `json:"total"`
	Limit      int             `json:"limit"`
	Offset     int             `json:"offset"`
	HasMore    bool            `json:"has_more"`
	NextOffset *int            `json:"next_offset,omitempty"`
	Data       []RoutingRecord `json:"data"`
}

type AggregateResponse struct {
	Object               string                `json:"object"`
	RecordCount          int                   `json:"record_count"`
	Lifecycle            LifecycleSummary      `json:"lifecycle"`
	Summary              AggregateCostSummary  `json:"summary"`
	ModelSelection       []AggregateValue      `json:"model_selection"`
	DecisionDistribution []AggregateValue      `json:"decision_distribution"`
	SignalDistribution   []AggregateValue      `json:"signal_distribution"`
	TokenVolume          AggregateTokenVolume  `json:"token_volume"`
	TokenBreakdown       AggregateTokenBuckets `json:"token_breakdown"`
	AvailableRecipes     []string              `json:"available_recipes"`
	AvailableDecisions   []string              `json:"available_decisions"`
	AvailableModels      []string              `json:"available_models"`
}

type LifecycleSummary struct {
	Completed  int `json:"completed"`
	Failed     int `json:"failed"`
	Aborted    int `json:"aborted"`
	InProgress int `json:"in_progress"`
	Unknown    int `json:"unknown"`
}

type AggregateCostSummary struct {
	TotalSaved          float64 `json:"total_saved"`
	BaselineSpend       float64 `json:"baseline_spend"`
	ActualSpend         float64 `json:"actual_spend"`
	Currency            string  `json:"currency,omitempty"`
	CostRecordCount     int     `json:"cost_record_count"`
	ExcludedRecordCount int     `json:"excluded_record_count"`
}

type AggregateValue struct {
	Name  string `json:"name"`
	Value int    `json:"value"`
}

type AggregateTokenVolume struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	TotalTokens         int `json:"total_tokens"`
	ExcludedRecordCount int `json:"excluded_record_count"`
}

type AggregateTokenBuckets struct {
	ByDecision      []AggregateTokenEntry `json:"by_decision"`
	BySelectedModel []AggregateTokenEntry `json:"by_selected_model"`
}

type AggregateTokenEntry struct {
	Name         string `json:"name"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	TotalTokens  int    `json:"total_tokens"`
}

type TrajectoryFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type TrajectoryToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function TrajectoryFunctionCall `json:"function"`
}

type TrajectoryMessage struct {
	ConversationID string               `json:"conversation_id,omitempty"`
	Role           string               `json:"role"`
	Content        string               `json:"content,omitempty"`
	ToolCalls      []TrajectoryToolCall `json:"tool_calls,omitempty"`
	ToolCallID     string               `json:"tool_call_id,omitempty"`
	ToolName       string               `json:"tool_name,omitempty"`
	TurnIndex      int                  `json:"turn_index"`
}

type TrajectoryResponse struct {
	Object      string              `json:"object"`
	SessionID   string              `json:"session_id"`
	Recipe      string              `json:"recipe"`
	RecordCount int                 `json:"record_count"`
	TurnCount   int                 `json:"turn_count"`
	Messages    []TrajectoryMessage `json:"messages"`
	Routes      []TrajectoryRoute   `json:"routes"`
}

type TrajectoryRoute struct {
	ConversationID       string    `json:"conversation_id,omitempty"`
	RecordID             string    `json:"record_id"`
	Timestamp            time.Time `json:"timestamp"`
	TurnIndex            int       `json:"turn_index"`
	Decision             string    `json:"decision,omitempty"`
	SelectedModel        string    `json:"selected_model,omitempty"`
	PreviousModel        string    `json:"previous_model,omitempty"`
	SelectionMethod      string    `json:"selection_method,omitempty"`
	SelectionReasoning   string    `json:"selection_reasoning,omitempty"`
	SessionPolicyApplied bool      `json:"session_policy_applied"`
	SessionAction        string    `json:"session_action,omitempty"`
	SessionReason        string    `json:"session_reason,omitempty"`
	LifecycleState       string    `json:"lifecycle_state"`
	ResponseStatus       int       `json:"response_status,omitempty"`
	DurationMS           int64     `json:"duration_ms"`
}
