package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"
)

func validateResponsesEventRequiredFields(wire responsesEventWire, body []byte) error {
	if err := validateResponsesEventIndexes(wire); err != nil {
		return err
	}
	if err := validateResponsesEventPayload(wire); err != nil {
		return err
	}
	return validateResponsesEventFieldPresence(wire.Type, body)
}

func validateResponsesEventPayload(wire responsesEventWire) error {
	switch wire.Type {
	case "response.created", "response.queued", "response.in_progress",
		"response.completed", "response.incomplete", "response.failed":
		return validateResponsesLifecycleEvent(wire)
	case "response.output_item.added", "response.output_item.done":
		return validateResponsesItemEvent(wire)
	case "response.content_part.added", "response.content_part.done":
		return validateResponsesContentPartEvent(wire)
	case "response.output_text.delta", "response.output_text.done",
		"response.refusal.delta", "response.refusal.done",
		"response.reasoning_text.delta", "response.reasoning_text.done":
		return validateResponsesContentEventTarget(wire)
	case "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done":
		return validateResponsesSummaryEventTarget(wire)
	case "response.function_call_arguments.delta", "response.function_call_arguments.done":
		return validateResponsesToolEventTarget(wire)
	case "response.output_text.annotation.added":
		return validateResponsesAnnotationEvent(wire)
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
		return validateResponsesSummaryPartEvent(wire)
	}
	return nil
}

func validateResponsesLifecycleEvent(wire responsesEventWire) error {
	if wire.Response == nil {
		return invalidProviderResponse("stream_response_required", "Responses lifecycle event requires a response resource")
	}
	return nil
}

func validateResponsesItemEvent(wire responsesEventWire) error {
	if wire.OutputIndex == nil || len(wire.Item) == 0 {
		return invalidProviderResponse("stream_item_required", "Responses output item event requires output_index and item")
	}
	return nil
}

func validateResponsesContentPartEvent(wire responsesEventWire) error {
	if wire.OutputIndex == nil || wire.ContentIndex == nil || strings.TrimSpace(wire.ItemID) == "" || wire.Part == nil {
		return invalidProviderResponse("stream_content_part_required", "Responses content part event is incomplete")
	}
	return nil
}

func validateResponsesContentEventTarget(wire responsesEventWire) error {
	if wire.OutputIndex == nil || wire.ContentIndex == nil || strings.TrimSpace(wire.ItemID) == "" {
		return invalidProviderResponse("stream_delta_target_required", "Responses content event requires output_index, content_index, and item_id")
	}
	return nil
}

func validateResponsesSummaryEventTarget(wire responsesEventWire) error {
	if wire.OutputIndex == nil || wire.SummaryIndex == nil || strings.TrimSpace(wire.ItemID) == "" {
		return invalidProviderResponse("stream_delta_target_required", "Responses reasoning summary event requires output_index, summary_index, and item_id")
	}
	return nil
}

func validateResponsesToolEventTarget(wire responsesEventWire) error {
	if wire.OutputIndex == nil || strings.TrimSpace(wire.ItemID) == "" {
		return invalidProviderResponse("stream_delta_target_required", "Responses delta event requires output_index and item_id")
	}
	return nil
}

func validateResponsesAnnotationEvent(wire responsesEventWire) error {
	if wire.OutputIndex == nil || wire.ContentIndex == nil || wire.AnnotationIndex == nil ||
		strings.TrimSpace(wire.ItemID) == "" || wire.Annotation == nil {
		return invalidProviderResponse("stream_annotation_required", "Responses annotation event is incomplete")
	}
	return nil
}

func validateResponsesSummaryPartEvent(wire responsesEventWire) error {
	if wire.OutputIndex == nil || wire.SummaryIndex == nil || strings.TrimSpace(wire.ItemID) == "" || wire.Part == nil {
		return invalidProviderResponse("stream_reasoning_part_required", "Responses reasoning summary part event is incomplete")
	}
	return nil
}

func validateAnthropicStreamEvent(wire anthropicEventWire, body []byte) error {
	if anthropicEventUsesIndex(wire.Type) && (wire.Index == nil || *wire.Index < 0) {
		return invalidProviderResponse("invalid_stream_item_index", "Anthropic content event requires a non-negative index")
	}
	if err := validateAnthropicEventPayload(wire); err != nil {
		return err
	}
	return validateAnthropicEventFieldPresence(wire.Type, body)
}

func validateAnthropicEventPayload(wire anthropicEventWire) error {
	switch wire.Type {
	case "message_start":
		return validateAnthropicMessageStart(wire)
	case "content_block_start":
		return requireAnthropicContentBlock(wire)
	case "content_block_delta":
		return requireAnthropicContentDelta(wire)
	case "message_delta":
		return validateAnthropicMessageDeltaEvent(wire)
	case "error":
		return validateAnthropicStreamError(wire)
	}
	return nil
}

func validateAnthropicMessageStart(wire anthropicEventWire) error {
	if wire.Message == nil {
		return invalidProviderResponse("stream_message_required", "Anthropic message_start requires a message")
	}
	if err := validateAnthropicResponseResource(*wire.Message); err != nil {
		return err
	}
	if wire.Message.StopReason != nil {
		return invalidProviderResponse("stream_start_stop_reason", "Anthropic message_start stop_reason must be null")
	}
	return nil
}

func requireAnthropicContentBlock(wire anthropicEventWire) error {
	if wire.ContentBlock == nil {
		return invalidProviderResponse("stream_content_block_required", "Anthropic content_block_start requires a content block")
	}
	return nil
}

func requireAnthropicContentDelta(wire anthropicEventWire) error {
	if wire.Delta == nil {
		return invalidProviderResponse("invalid_stream_delta", "Anthropic content_block_delta requires a delta")
	}
	return nil
}

func validateAnthropicMessageDeltaEvent(wire anthropicEventWire) error {
	if wire.Delta == nil {
		return invalidProviderResponse("stream_message_delta_required", "Anthropic message_delta requires a delta")
	}
	if wire.Usage == nil {
		return invalidProviderResponse("stream_message_usage_required", "Anthropic message_delta requires usage")
	}
	return validateAnthropicMessageDeltaStop(wire)
}

func validateAnthropicMessageDeltaStop(wire anthropicEventWire) error {
	if wire.Delta.StopReason != nil && !validAnthropicStopReason(*wire.Delta.StopReason) {
		return invalidProviderResponse("invalid_anthropic_stop_reason", "Anthropic stop reason is not recognized")
	}
	if wire.Delta.StopReason != nil && *wire.Delta.StopReason == "stop_sequence" &&
		(wire.Delta.StopSequence == nil || strings.TrimSpace(*wire.Delta.StopSequence) == "") {
		return invalidProviderResponse("anthropic_stop_sequence_required", "Anthropic stop_sequence reason requires the matched sequence")
	}
	if wire.Delta.StopSequence != nil &&
		(wire.Delta.StopReason == nil || *wire.Delta.StopReason != "stop_sequence") {
		return invalidProviderResponse("anthropic_stop_sequence_reason", "Anthropic stop_sequence value requires stop_sequence reason")
	}
	return nil
}

func validateAnthropicStreamError(wire anthropicEventWire) error {
	if wire.Error == nil || strings.TrimSpace(wire.Error.Type) == "" || strings.TrimSpace(wire.Error.Message) == "" {
		return invalidProviderResponse(
			"invalid_anthropic_stream_error",
			"Anthropic error event requires an error type and message",
		)
	}
	return nil
}

func validateResponsesEventFieldPresence(eventType string, body []byte) error {
	if eventType == "error" {
		fields, err := providerEventFields(body)
		if err != nil {
			return err
		}
		if err := requireProviderFields(fields, "code", "message", "param"); err != nil {
			return err
		}
		return requireProviderNonNullFields(fields, "message")
	}
	required := map[string][]string{
		"response.output_text.delta":             {"delta"},
		"response.output_text.done":              {"text"},
		"response.refusal.delta":                 {"delta"},
		"response.refusal.done":                  {"refusal"},
		"response.reasoning_text.delta":          {"delta"},
		"response.reasoning_text.done":           {"text"},
		"response.reasoning_summary_text.delta":  {"delta"},
		"response.reasoning_summary_text.done":   {"text"},
		"response.function_call_arguments.delta": {"delta"},
		"response.function_call_arguments.done":  {"name", "arguments"},
	}[eventType]
	if len(required) == 0 {
		return nil
	}
	fields, err := providerEventFields(body)
	if err != nil {
		return err
	}
	return requireProviderNonNullFields(fields, required...)
}

func validateAnthropicEventFieldPresence(eventType string, body []byte) error {
	if eventType != "message_delta" && eventType != "content_block_delta" {
		return nil
	}
	fields, err := providerEventFields(body)
	if err != nil {
		return err
	}
	delta, err := providerNestedEventFields(fields, "delta")
	if err != nil {
		return err
	}
	if eventType == "message_delta" {
		return requireProviderFields(delta, "stop_reason", "stop_sequence")
	}
	var deltaType string
	if err := json.Unmarshal(delta["type"], &deltaType); err != nil {
		return invalidProviderResponse("invalid_stream_delta", "Anthropic stream delta type is invalid")
	}
	required := map[string]string{
		"text_delta":       "text",
		"thinking_delta":   "thinking",
		"input_json_delta": "partial_json",
		"signature_delta":  "signature",
	}[deltaType]
	if required == "" {
		return nil
	}
	return requireProviderNonNullFields(delta, required)
}

func providerEventFields(body []byte) (map[string]json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, invalidProviderResponse("invalid_upstream_json", "Upstream stream event JSON is invalid")
	}
	return fields, nil
}

func providerNestedEventFields(fields map[string]json.RawMessage, name string) (map[string]json.RawMessage, error) {
	raw, present := fields[name]
	if !present || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, invalidProviderResponse("stream_required_field", "Upstream stream event is missing required field "+name)
	}
	var nested map[string]json.RawMessage
	if err := json.Unmarshal(raw, &nested); err != nil {
		return nil, invalidProviderResponse("invalid_upstream_json", "Upstream stream event field "+name+" is invalid")
	}
	return nested, nil
}

func requireProviderFields(fields map[string]json.RawMessage, required ...string) error {
	for _, name := range required {
		if _, present := fields[name]; !present {
			return invalidProviderResponse("stream_required_field", "Upstream stream event is missing required field "+name)
		}
	}
	return nil
}

func requireProviderNonNullFields(fields map[string]json.RawMessage, required ...string) error {
	if err := requireProviderFields(fields, required...); err != nil {
		return err
	}
	for _, name := range required {
		if bytes.Equal(bytes.TrimSpace(fields[name]), []byte("null")) {
			return invalidProviderResponse("stream_required_field", "Upstream stream event field "+name+" cannot be null")
		}
	}
	return nil
}

func validateResponsesEventIndexes(wire responsesEventWire) error {
	for _, field := range []struct {
		name  string
		value *int
	}{
		{name: "output_index", value: wire.OutputIndex},
		{name: "content_index", value: wire.ContentIndex},
		{name: "annotation_index", value: wire.AnnotationIndex},
		{name: "summary_index", value: wire.SummaryIndex},
	} {
		if field.value != nil && *field.value < 0 {
			return invalidProviderResponse("invalid_stream_item_index", "Responses "+field.name+" must be non-negative")
		}
	}
	return nil
}

func anthropicEventUsesIndex(eventType string) bool {
	switch eventType {
	case "content_block_start", "content_block_delta", "content_block_stop":
		return true
	default:
		return false
	}
}

func anthropicIndex(index int) *int {
	return &index
}

func anthropicEventIndex(wire anthropicEventWire) int {
	if wire.Index == nil {
		return 0
	}
	return *wire.Index
}
