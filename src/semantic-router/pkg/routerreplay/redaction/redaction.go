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
		return redactEach(typed, redactRecordMap)
	default:
		return false
	}
}

// redactObjectPayload redacts a record, a list's records or runs, a
// trajectory's messages and routes, and a run's summary and calls. A
// trajectory route repeats a record's route diagnostics and the agent hash
// from its session policy, and a run what its records hold, so they are
// redacted the same way.
func redactObjectPayload(payload map[string]any) bool {
	changed := redactRecordMap(payload)
	for _, list := range []struct {
		field  string
		redact func(map[string]any) bool
	}{
		{"messages", redactTrajectoryMessageMap},
		{"data", redactRecordOrRunMap},
		{"routes", redactRouteDiagnosticsMap},
		{"calls", redactRunCallMap},
	} {
		if items, ok := payload[list.field].([]any); ok && redactEach(items, list.redact) {
			changed = true
		}
	}
	if redactNestedMap(payload, "run", redactRecordOrRunMap) {
		changed = true
	}
	return changed
}

func redactRecordOrRunMap(item map[string]any) bool {
	changed := redactRecordMap(item)
	return redactRunSummaryMap(item) || changed
}

// redactRunSummaryMap clears a run's agent hashes and the text of the
// outcomes reported for it, keeping its tree, labels and counts. Its prompt
// and session name are cleared with a record's fields.
func redactRunSummaryMap(run map[string]any) bool {
	changed := false
	if agents, ok := run["agents"].([]any); ok && redactEach(agents, redactAgentIdentityMap) {
		changed = true
	}
	outcome, _ := run["outcome"].(map[string]any)
	if reported, ok := outcome["reported"].([]any); ok && redactEach(reported, func(item map[string]any) bool {
		return redactOutcomes([]any{item["outcome"]})
	}) {
		changed = true
	}
	return changed
}

// redactRunCallMap clears a run call's protection and Looper reason, its
// reported outcomes' text and its tool calls' arguments, which keep their
// names and statuses as a record's tool trace does, and its Looper hops' too.
func redactRunCallMap(call map[string]any) bool {
	changed := clearStringField(call, "reason")
	if redactOutcomes(call["outcomes"]) {
		changed = true
	}
	if tools, ok := call["tool_calls"].([]any); ok && redactEach(tools, func(tool map[string]any) bool {
		cleared := clearStringField(tool, "summary")
		return clearStringField(tool, "arguments") || cleared
	}) {
		changed = true
	}
	if hops, ok := call["hops"].([]any); ok && redactEach(hops, redactRunCallMap) {
		changed = true
	}
	return changed
}

// redactAgentIdentityMap clears the hashes of the agent ids a harness sent,
// and what its spawning tool call asked it for.
func redactAgentIdentityMap(agent map[string]any) bool {
	changed := clearStringField(agent, "id")
	changed = clearStringField(agent, "task") || changed
	return clearStringField(agent, "parent_id") || changed
}

func redactEach(items []any, redact func(map[string]any) bool) bool {
	changed := false
	for _, item := range items {
		object, ok := item.(map[string]any)
		if ok && redact(object) {
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
	if redactNestedMap(record, "agent", redactAgentIdentityMap) {
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
	if redactNestedMap(record, "review", redactReviewMap) {
		changed = true
	}
	return changed
}

// redactReviewMap keeps an approval review's status and verdict enums and
// the reviewed tool's name, and clears the reviewer's rationale and the
// planned action's command, directory and arguments.
func redactReviewMap(review map[string]any) bool {
	changed := false
	if assessment, ok := review["assessment"].(map[string]any); ok && clearStringField(assessment, "rationale") {
		changed = true
	}
	if action, ok := review["action"].(map[string]any); ok {
		for _, field := range []string{"summary", "cwd", "arguments"} {
			if clearStringField(action, field) {
				changed = true
			}
		}
	}
	if changed {
		review["content_redacted"] = true
	}
	return changed
}

func clearRecordPayloadFields(record map[string]any) bool {
	changed := false
	for _, field := range []string{"request_body", "response_body", "prompt", "prompt_original_sha256", "tool_definitions", "session_name"} {
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
	for _, field := range []string{"target_ref", "reason", "idempotency_key"} {
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
		"agent",
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
	for _, field := range []string{"annotations", "signal_errors", "looper"} {
		if clearObjectField(diagnostics, field) {
			changed = true
		}
	}
	if deleteField(diagnostics, "prepared_dispatch") {
		changed = true
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

func deleteField(value map[string]any, field string) bool {
	if _, ok := value[field]; !ok {
		return false
	}
	delete(value, field)
	return true
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
			if _, hasStatus := step["status"]; !hasStatus {
				step["status"] = ToolTraceResultStatus(step["text"])
			}
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
