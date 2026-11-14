package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// Keep every HTTP routing result, including intermediate tool hops. Conversation
// snapshots may collapse within a turn; route decisions must not disappear with them.
type trajectoryRoute = routerreplay.TrajectoryRoute

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
			Agent: agentOf(record),
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

// agentOf is the short hash of the agent identity the protection policy
// recorded for a call, "" when the harness sent none or named the session
// itself: Codex sends its root thread's id as both, and that thread is the
// main conversation.
func agentOf(record routerreplay.RoutingRecord) string {
	identity, _ := record.SessionPolicy["identity"].(map[string]interface{})
	agent := identityHash(identity["agent"])
	if agent == identityHash(identity["session"]) {
		return ""
	}
	return agent
}

func identityHash(part interface{}) string {
	fields, _ := part.(map[string]interface{})
	hash, _ := fields["hash"].(string)
	return hash
}
