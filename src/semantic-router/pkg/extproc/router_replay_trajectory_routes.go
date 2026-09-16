package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// Keep every HTTP routing result, including intermediate tool hops. Conversation
// snapshots may collapse within a turn; route decisions must not disappear with them.
type trajectoryRoute struct {
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

func filterTrajectoryRecordsByRecipe(records []routerreplay.RoutingRecord, recipe string) []routerreplay.RoutingRecord {
	matched := make([]routerreplay.RoutingRecord, 0, len(records))
	for _, record := range records {
		if record.Recipe == recipe {
			matched = append(matched, record)
		}
	}
	return matched
}

func buildTrajectoryRoutes(records []routerreplay.RoutingRecord) []trajectoryRoute {
	routes := make([]trajectoryRoute, 0, len(records))
	for _, record := range records {
		route := trajectoryRoute{
			RecordID: record.ID, Timestamp: record.Timestamp, TurnIndex: record.TurnIndex,
			ConversationID: record.ConversationID,
			Decision:       record.Decision, SelectedModel: record.SelectedModel,
			SelectionMethod: record.SelectionMethod, LifecycleState: record.LifecycleState,
			ResponseStatus: record.ResponseStatus, DurationMS: record.DurationMS,
		}
		if diagnostics := record.RouteDiagnostics; diagnostics != nil {
			route.PreviousModel = diagnostics.PreviousModel
			route.SelectionReasoning = diagnostics.SelectionReasoning
			route.SessionPolicyApplied = diagnostics.SessionPolicyApplied
			route.SessionAction = diagnostics.SessionAction
			route.SessionReason = diagnostics.SessionReason
		}
		routes = append(routes, route)
	}
	return routes
}
