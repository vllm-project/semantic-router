package extproc

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// safeErrorForLog returns diagnostic classification without calling Error().
// Error strings from request-time dependencies can contain request URLs,
// response bodies, prompts, or credentials and therefore are not log-safe.
func safeErrorForLog(err error) string {
	if err == nil {
		return "none"
	}
	if errors.Is(err, context.Canceled) {
		return "context_canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "context_deadline_exceeded"
	}
	var statusCarrier interface{ GRPCStatus() *status.Status }
	if errors.As(err, &statusCarrier) {
		grpcStatus := statusCarrier.GRPCStatus()
		if grpcStatus != nil && grpcStatus.Code() != codes.Unknown {
			return fmt.Sprintf("type=%T grpc_code=%s", err, grpcStatus.Code())
		}
	}
	return fmt.Sprintf("type=%T", err)
}

func safeRecoveredValueForLog(value interface{}) string {
	if value == nil {
		return "nil"
	}
	return fmt.Sprintf("type=%T", value)
}

// safeRouterReplayLogFields deliberately builds a small allowlist instead of
// redacting routerreplay.LogFields. Replay records retain request/response
// bodies, prompts, tool arguments and detection excerpts for the replay API;
// denylisting those values here would make newly added fields unsafe by
// default.
func safeRouterReplayLogFields(record routerreplay.RoutingRecord, event string) map[string]interface{} {
	fields := map[string]interface{}{
		"event":                      event,
		"replay_id":                  record.ID,
		"request_id":                 record.RequestID,
		"decision":                   record.Decision,
		"decision_tier":              record.DecisionTier,
		"decision_priority":          record.DecisionPriority,
		"category":                   record.Category,
		"original_model":             record.OriginalModel,
		"selected_model":             record.SelectedModel,
		"reasoning_mode":             record.ReasoningMode,
		"confidence_score":           record.ConfidenceScore,
		"selection_method":           record.SelectionMethod,
		"timestamp":                  record.Timestamp,
		"turn_index":                 record.TurnIndex,
		"from_cache":                 record.FromCache,
		"streaming":                  record.Streaming,
		"response_status":            record.ResponseStatus,
		"projection_count":           len(record.Projections),
		"session_policy_field_count": len(record.SessionPolicy),
		"pii_entity_count":           len(record.PIIEntities),
		"hallucination_span_count":   len(record.HallucinationSpans),
	}
	if record.SessionID != "" {
		fields["session_id"] = record.SessionID
	}
	if record.ConversationID != "" {
		fields["conversation_id"] = record.ConversationID
	}
	if record.PreviousResponseID != "" {
		fields["previous_response_id"] = record.PreviousResponseID
	}
	if record.ToolTrace != nil {
		fields["tool_trace_flow"] = record.ToolTrace.Flow
		fields["tool_trace_stage"] = record.ToolTrace.Stage
		fields["tool_name_count"] = len(record.ToolTrace.ToolNames)
		fields["tool_trace_step_count"] = len(record.ToolTrace.Steps)
		fields["tool_trace_steps_truncated"] = record.ToolTrace.StepsTruncated
		fields["tool_trace_dropped_step_count"] = record.ToolTrace.DroppedStepCount
	}
	return fields
}
