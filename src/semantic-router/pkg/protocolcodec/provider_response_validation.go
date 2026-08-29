package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func validateResponsesOutputItemResource(body json.RawMessage, item responsesItemWire) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return invalidProviderResponse("invalid_response_item", "Responses output item is invalid")
	}
	required := []string{"type"}
	switch item.Type {
	case "message":
		required = append(required, "id", "role", "content", "status")
		if item.Role != "assistant" {
			return invalidProviderResponse("invalid_response_role", "Responses output messages must use the assistant role")
		}
	case "function_call":
		required = append(required, "call_id", "name", "arguments")
	case "reasoning":
		required = append(required, "id", "summary")
	default:
		return invalidProviderResponse("invalid_response_item", "Responses output item type is unsupported")
	}
	for _, field := range required {
		value, present := fields[field]
		if !present || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return invalidProviderResponse(
				"invalid_response_item",
				"Responses output item is missing required field "+field,
			)
		}
	}
	for _, value := range []struct {
		name  string
		value string
	}{
		{name: "id", value: item.ID},
		{name: "call_id", value: item.CallID},
		{name: "name", value: item.Name},
	} {
		if _, required := fields[value.name]; required && strings.TrimSpace(value.value) == "" {
			return invalidProviderResponse(
				"invalid_response_item",
				"Responses output item field "+value.name+" cannot be empty",
			)
		}
	}
	return validateResponsesOutputItemStatus(item)
}

// Provider response validation closes value-level protocol contracts before a
// provider envelope is reduced to the neutral response. Shape validation alone
// is insufficient: an unknown discriminator or a duplicate candidate index can
// otherwise be silently reinterpreted as a valid neutral response.

func validateChatResponseResource(wire chatResponseWire) error {
	if wire.Object != "" && wire.Object != "chat.completion" {
		return invalidProviderResponse("invalid_chat_response_object", "Chat response object must be chat.completion")
	}
	if wire.Error != nil && strings.TrimSpace(wire.Error.Message) == "" {
		return invalidProviderResponse("invalid_chat_response_error", "Chat response error requires a message")
	}
	return nil
}

func normalizedChatChoices(choices []chatChoiceWire) ([]chatChoiceWire, error) {
	normalized := append([]chatChoiceWire(nil), choices...)
	sort.Slice(normalized, func(left, right int) bool {
		return normalized[left].Index < normalized[right].Index
	})
	for position, choice := range normalized {
		if choice.Index != position {
			return nil, invalidProviderResponse(
				"invalid_chat_choice_index",
				fmt.Sprintf("Chat choice indexes must be unique and contiguous from zero; missing index %d", position),
			)
		}
		if choice.FinishReason != nil && !validChatFinishReason(*choice.FinishReason) {
			return nil, invalidProviderResponse("invalid_chat_finish_reason", "Chat finish reason is not recognized")
		}
	}
	return normalized, nil
}

func validChatFinishReason(reason string) bool {
	switch reason {
	case "stop", "length", "tool_calls", "content_filter", "function_call":
		return true
	default:
		return false
	}
}

func validateResponsesResponseResource(wire responsesResponseWire, allowNonterminal bool) error {
	if wire.Object != "" && wire.Object != "response" {
		return invalidProviderResponse("invalid_responses_object", "Responses object must be response")
	}
	if wire.Status != "" && !validResponsesStatus(wire.Status) {
		return invalidProviderResponse("invalid_responses_status", "Responses status is not recognized")
	}
	if !allowNonterminal && (wire.Status == "queued" || wire.Status == "in_progress") {
		return invalidProviderResponse("nonterminal_responses_resource", "Buffered inference returned a nonterminal Responses resource")
	}
	if wire.IncompleteDetails != nil && wire.IncompleteDetails.Reason != "max_output_tokens" &&
		wire.IncompleteDetails.Reason != "content_filter" {
		return invalidProviderResponse("invalid_responses_incomplete_reason", "Responses incomplete reason is not recognized")
	}
	if wire.Status == "incomplete" && wire.IncompleteDetails == nil {
		return invalidProviderResponse("responses_incomplete_details_required", "An incomplete Responses resource requires incomplete details")
	}
	if wire.Status != "" && wire.Status != "incomplete" && wire.IncompleteDetails != nil {
		return invalidProviderResponse("responses_incomplete_status", "Responses incomplete details require incomplete status")
	}
	if wire.Status == "failed" && wire.Error == nil {
		return invalidProviderResponse("responses_error_required", "A failed Responses resource requires an error")
	}
	if wire.Error != nil && wire.Status != "" && wire.Status != "failed" {
		return invalidProviderResponse("responses_error_status", "A Responses error is only valid with failed status")
	}
	if wire.Error != nil && (strings.TrimSpace(wire.Error.Code) == "" || strings.TrimSpace(wire.Error.Message) == "") {
		return invalidProviderResponse("invalid_responses_error", "A Responses error requires a code and message")
	}
	return nil
}

func validResponsesStatus(status string) bool {
	switch status {
	case "completed", "failed", "in_progress", "cancelled", "queued", "incomplete":
		return true
	default:
		return false
	}
}

func validateResponsesOutputItemStatus(item responsesItemWire) error {
	if item.Status == "" {
		return nil
	}
	switch item.Status {
	case "completed", "in_progress", "incomplete":
		return nil
	default:
		return invalidProviderResponse("invalid_responses_item_status", "Responses output item status is not recognized")
	}
}

func validateAnthropicResponseResource(wire anthropicResponseWire) error {
	if wire.Type != "" && wire.Type != "message" {
		return invalidProviderResponse("invalid_anthropic_response_type", "Anthropic response type must be message")
	}
	if wire.Role != "" && wire.Role != "assistant" {
		return invalidProviderResponse("invalid_anthropic_response_role", "Anthropic response role must be assistant")
	}
	if wire.StopReason != nil && !validAnthropicStopReason(*wire.StopReason) {
		return invalidProviderResponse("invalid_anthropic_stop_reason", "Anthropic stop reason is not recognized")
	}
	if wire.StopReason != nil && *wire.StopReason == "stop_sequence" &&
		(wire.StopSequence == nil || strings.TrimSpace(*wire.StopSequence) == "") {
		return invalidProviderResponse("anthropic_stop_sequence_required", "Anthropic stop_sequence reason requires the matched sequence")
	}
	if wire.StopSequence != nil && (wire.StopReason == nil || *wire.StopReason != "stop_sequence") {
		return invalidProviderResponse("anthropic_stop_sequence_reason", "Anthropic stop_sequence value requires stop_sequence reason")
	}
	if wire.Error != nil && (strings.TrimSpace(wire.Error.Type) == "" || strings.TrimSpace(wire.Error.Message) == "") {
		return invalidProviderResponse("invalid_anthropic_response_error", "Anthropic response error requires a type and message")
	}
	return nil
}

func validateTransportErrorDetails(errorType, message string) error {
	if strings.TrimSpace(errorType) == "" {
		return invalidProviderResponse(
			"upstream_error_type_required",
			"upstream transport error details require a non-empty type",
		)
	}
	if strings.TrimSpace(message) == "" {
		return invalidProviderResponse(
			"upstream_error_message_required",
			"upstream transport error details require a non-empty message",
		)
	}
	return nil
}

func validAnthropicStopReason(reason string) bool {
	switch reason {
	case "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal", "model_context_window_exceeded":
		return true
	default:
		return false
	}
}

func invalidProviderResponse(code, message string) error {
	return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, code, message, nil)
}
