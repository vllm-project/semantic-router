package redaction

import (
	"bytes"
	"encoding/json"
	"strings"
)

const (
	RedactedToolResultStatusSucceeded = "succeeded"
	RedactedToolResultStatusFailed    = "failed"
	toolExecutionFailurePrefix        = "Tool execution failed:"
)

// RedactResponseBody removes captured prompt, response, and tool payloads from
// a replay API response while preserving its routing and execution metadata.
func RedactResponseBody(body []byte) ([]byte, bool, error) {
	if len(bytes.TrimSpace(body)) == 0 {
		return body, false, nil
	}

	var payload any
	if err := json.Unmarshal(body, &payload); err != nil {
		return body, false, err
	}
	changed := redactPayload(payload)
	if !changed {
		return body, false, nil
	}
	redactedBody, err := json.Marshal(payload)
	if err != nil {
		return body, false, err
	}
	return redactedBody, true, nil
}

func redactPayload(payload any) bool {
	switch typed := payload.(type) {
	case map[string]any:
		return redactObjectPayload(typed)
	case []any:
		return redactRecordItems(typed)
	default:
		return false
	}
}

func redactObjectPayload(payload map[string]any) bool {
	changed := redactRecordMap(payload)
	if messages, ok := payload["messages"].([]any); ok && redactTrajectoryMessages(messages) {
		changed = true
	}
	if data, ok := payload["data"].([]any); ok && redactRecordItems(data) {
		changed = true
	}
	return changed
}

func redactTrajectoryMessages(messages []any) bool {
	changed := false
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if ok && redactTrajectoryMessageMap(message) {
			changed = true
		}
	}
	return changed
}

func redactRecordItems(items []any) bool {
	changed := false
	for _, item := range items {
		record, ok := item.(map[string]any)
		if ok && redactRecordMap(record) {
			changed = true
		}
	}
	return changed
}

func redactTrajectoryMessageMap(message map[string]any) bool {
	changed := false
	if content, ok := message["content"].(string); ok && strings.TrimSpace(content) != "" {
		if role, _ := message["role"].(string); role == "tool" {
			message["status"] = ToolTraceResultStatus(content)
		}
		message["content"] = ""
		message["content_redacted"] = true
		changed = true
	}

	toolCalls, ok := message["tool_calls"].([]any)
	if !ok {
		return changed
	}
	for _, rawCall := range toolCalls {
		call, ok := rawCall.(map[string]any)
		if !ok {
			continue
		}
		function, ok := call["function"].(map[string]any)
		if !ok {
			continue
		}
		arguments, ok := function["arguments"].(string)
		if !ok || strings.TrimSpace(arguments) == "" {
			continue
		}
		function["arguments"] = ""
		call["content_redacted"] = true
		message["content_redacted"] = true
		changed = true
	}
	return changed
}

func redactRecordMap(record map[string]any) bool {
	changed := clearRecordPayloadFields(record)
	if redactHallucinationSpanDetails(record["hallucination_span_details"]) {
		changed = true
	}
	if redactOutcomes(record["outcomes"]) {
		changed = true
	}
	if clearObjectField(record, "session_policy") {
		changed = true
	}
	if redactNestedMap(record, "route_diagnostics", redactRouteDiagnosticsMap) {
		changed = true
	}
	if redactNestedMap(record, "learning", redactLearningDiagnosticsMap) {
		changed = true
	}
	if redactNestedMap(record, "tool_trace", redactToolTraceMap) {
		changed = true
	}
	return changed
}

func clearRecordPayloadFields(record map[string]any) bool {
	changed := false
	for _, field := range []string{"request_body", "response_body", "prompt", "tool_definitions"} {
		if clearStringField(record, field) {
			changed = true
		}
	}
	for _, field := range []string{"pii_entities", "hallucination_spans"} {
		if clearArrayField(record, field) {
			changed = true
		}
	}
	return changed
}

func redactHallucinationSpanDetails(raw any) bool {
	details, ok := raw.([]any)
	if !ok {
		return false
	}
	changed := false
	for _, rawDetail := range details {
		detail, ok := rawDetail.(map[string]any)
		if !ok {
			continue
		}
		detailChanged := clearStringField(detail, "text")
		if clearStringField(detail, "explanation") {
			detailChanged = true
		}
		if detailChanged {
			changed = true
		}
	}
	return changed
}

func redactOutcomes(raw any) bool {
	outcomes, ok := raw.([]any)
	if !ok {
		return false
	}
	changed := false
	for _, rawOutcome := range outcomes {
		outcome, ok := rawOutcome.(map[string]any)
		if !ok || !redactOutcome(outcome) {
			continue
		}
		outcome["content_redacted"] = true
		changed = true
	}
	return changed
}

func redactOutcome(outcome map[string]any) bool {
	changed := false
	for _, field := range []string{"target_ref", "reason"} {
		if clearStringField(outcome, field) {
			changed = true
		}
	}
	return clearObjectField(outcome, "metadata") || changed
}

func redactNestedMap(record map[string]any, field string, redact func(map[string]any) bool) bool {
	value, ok := record[field].(map[string]any)
	return ok && redact(value)
}

func redactRouteDiagnosticsMap(diagnostics map[string]any) bool {
	changed := false
	for _, field := range []string{
		"selection_reasoning",
		"session_reason",
		"hard_lock_reason",
		"decision_reason",
		"memory_reason",
		"memory_fallback_reason",
		"context_compression_skip_reason",
		"context_compression_fallback",
	} {
		if clearStringField(diagnostics, field) {
			changed = true
		}
	}
	for _, field := range []string{"annotations", "signal_errors"} {
		if clearObjectField(diagnostics, field) {
			changed = true
		}
	}
	return changed
}

func redactLearningDiagnosticsMap(learning map[string]any) bool {
	changed := false
	for _, componentName := range []string{"protection_preflight", "adaptation", "protection"} {
		component, ok := learning[componentName].(map[string]any)
		if !ok {
			continue
		}
		for _, field := range []string{
			"reason",
			"non_portable_context_reason",
			"hard_lock_reason",
			"decision_reason",
			"remaining_turn_prior_rejected",
		} {
			if clearStringField(component, field) {
				changed = true
			}
		}
	}
	return changed
}

func clearStringField(value map[string]any, field string) bool {
	raw, ok := value[field].(string)
	if !ok || strings.TrimSpace(raw) == "" {
		return false
	}
	value[field] = ""
	return true
}

func clearArrayField(value map[string]any, field string) bool {
	raw, ok := value[field].([]any)
	if !ok || len(raw) == 0 {
		return false
	}
	value[field] = []any{}
	return true
}

func clearObjectField(value map[string]any, field string) bool {
	raw, ok := value[field].(map[string]any)
	if !ok || len(raw) == 0 {
		return false
	}
	value[field] = map[string]any{}
	return true
}

func redactToolTraceMap(toolTrace map[string]any) bool {
	steps, ok := toolTrace["steps"].([]any)
	if !ok {
		return false
	}
	changed := false
	for _, rawStep := range steps {
		step, ok := rawStep.(map[string]any)
		if !ok {
			continue
		}
		if stepType, _ := step["type"].(string); strings.TrimSpace(stepType) == "client_tool_result" {
			step["status"] = ToolTraceResultStatus(step["text"])
			changed = true
		}
		for _, field := range []string{"text", "arguments", "raw_arguments", "output", "raw_output"} {
			value, ok := step[field].(string)
			if !ok || strings.TrimSpace(value) == "" {
				continue
			}
			step[field] = ""
			step["content_redacted"] = true
			changed = true
		}
	}
	return changed
}

// ToolTraceResultStatus preserves success/failure observability after a tool
// result payload is removed.
func ToolTraceResultStatus(rawText any) string {
	text, _ := rawText.(string)
	normalized := strings.TrimSpace(text)
	if normalized == "" {
		return RedactedToolResultStatusFailed
	}
	lower := strings.ToLower(normalized)
	if lower == "null" || lower == "undefined" ||
		strings.HasPrefix(lower, strings.ToLower(toolExecutionFailurePrefix)) {
		return RedactedToolResultStatusFailed
	}
	return RedactedToolResultStatusSucceeded
}
