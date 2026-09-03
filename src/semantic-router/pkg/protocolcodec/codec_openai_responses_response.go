package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type responsesResponseWire struct {
	ID                string              `json:"id"`
	Object            string              `json:"object,omitempty"`
	CreatedAt         int64               `json:"created_at"`
	Model             string              `json:"model"`
	Status            string              `json:"status,omitempty"`
	Output            json.RawMessage     `json:"output"`
	Usage             *responsesUsageWire `json:"usage,omitempty"`
	Error             *responsesErrorWire `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	PreviousResponseID   string            `json:"previous_response_id,omitempty"`
	Conversation         json.RawMessage   `json:"conversation,omitempty"`
	ConversationID       string            `json:"conversation_id,omitempty"`
	Metadata             map[string]string `json:"metadata"`
	Background           json.RawMessage   `json:"background,omitempty"`
	CompletedAt          json.RawMessage   `json:"completed_at,omitempty"`
	Instructions         json.RawMessage   `json:"instructions"`
	MaxOutputTokens      json.RawMessage   `json:"max_output_tokens,omitempty"`
	MaxToolCalls         json.RawMessage   `json:"max_tool_calls,omitempty"`
	Moderation           json.RawMessage   `json:"moderation,omitempty"`
	OutputText           json.RawMessage   `json:"output_text,omitempty"`
	ParallelToolCalls    json.RawMessage   `json:"parallel_tool_calls"`
	Prompt               json.RawMessage   `json:"prompt,omitempty"`
	PromptCacheKey       json.RawMessage   `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   json.RawMessage   `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention json.RawMessage   `json:"prompt_cache_retention,omitempty"`
	Reasoning            json.RawMessage   `json:"reasoning,omitempty"`
	SafetyIdentifier     json.RawMessage   `json:"safety_identifier,omitempty"`
	ServiceTier          json.RawMessage   `json:"service_tier,omitempty"`
	Store                json.RawMessage   `json:"store,omitempty"`
	Temperature          json.RawMessage   `json:"temperature"`
	Text                 json.RawMessage   `json:"text,omitempty"`
	ToolChoice           json.RawMessage   `json:"tool_choice"`
	Tools                json.RawMessage   `json:"tools"`
	TopLogprobs          json.RawMessage   `json:"top_logprobs,omitempty"`
	TopP                 json.RawMessage   `json:"top_p"`
	Truncation           json.RawMessage   `json:"truncation,omitempty"`
	User                 json.RawMessage   `json:"user,omitempty"`
}

func newResponsesResponseWire(id, model, status string, createdAt int64, previousResponseID string) responsesResponseWire {
	return responsesResponseWire{
		ID: id, Object: "response", CreatedAt: createdAt, Model: model, Status: status,
		Output:             json.RawMessage(`[]`),
		Metadata:           map[string]string{},
		Instructions:       json.RawMessage(`null`),
		ParallelToolCalls:  json.RawMessage(`true`),
		Temperature:        json.RawMessage(`null`),
		ToolChoice:         json.RawMessage(`"auto"`),
		Tools:              json.RawMessage(`[]`),
		TopP:               json.RawMessage(`null`),
		PreviousResponseID: previousResponseID,
	}
}

type responsesUsageWire struct {
	InputTokens         int64                        `json:"input_tokens"`
	OutputTokens        int64                        `json:"output_tokens"`
	TotalTokens         int64                        `json:"total_tokens"`
	ComputeUnits        json.RawMessage              `json:"compute_units,omitempty"`
	InputTokensDetails  *responsesInputUsageDetails  `json:"input_tokens_details"`
	OutputTokensDetails *responsesOutputUsageDetails `json:"output_tokens_details"`
}

type responsesInputUsageDetails struct {
	CachedTokens     int64 `json:"cached_tokens"`
	CacheWriteTokens int64 `json:"cache_write_tokens"`
}

type responsesOutputUsageDetails struct {
	ReasoningTokens int64 `json:"reasoning_tokens"`
}

type responsesErrorWire struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func responsesErrorCode(protocolError *llmprotocol.ProtocolError) string {
	if protocolError.Code != "" {
		return protocolError.Code
	}
	switch protocolError.Category {
	case llmprotocol.ErrorRateLimited:
		return "rate_limit_exceeded"
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorNotFound,
		llmprotocol.ErrorConflict, llmprotocol.ErrorUnsupportedFeature:
		return "invalid_prompt"
	default:
		return "server_error"
	}
}

func (OpenAIResponsesCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire responsesResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateResponsesResponseResource(wire, false); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	diagnostics := responsesResponseMetadataDiagnostics(wire, policy)
	response, err := decodeResponsesResponseResource(wire, policy, &diagnostics)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	return response, responseEnvelope(llmprotocol.OpenAIResponsesV1, body, response.Generation, wire.Status, policy), diagnostics, nil
}

func responsesResponseMetadataDiagnostics(wire responsesResponseWire, policy llmprotocol.Policy) llmprotocol.Diagnostics {
	var diagnostics llmprotocol.Diagnostics
	appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
		"background": len(wire.Background) > 0, "completed_at": len(wire.CompletedAt) > 0,
		"conversation": len(wire.Conversation) > 0, "conversation_id": wire.ConversationID != "",
		"instructions": len(wire.Instructions) > 0, "max_output_tokens": len(wire.MaxOutputTokens) > 0,
		"max_tool_calls": len(wire.MaxToolCalls) > 0, "metadata": len(wire.Metadata) > 0,
		"moderation": len(wire.Moderation) > 0, "output_text": len(wire.OutputText) > 0,
		"parallel_tool_calls":  len(wire.ParallelToolCalls) > 0,
		"previous_response_id": wire.PreviousResponseID != "", "reasoning": len(wire.Reasoning) > 0,
		"prompt": len(wire.Prompt) > 0, "prompt_cache_key": len(wire.PromptCacheKey) > 0,
		"prompt_cache_options":   len(wire.PromptCacheOptions) > 0,
		"prompt_cache_retention": len(wire.PromptCacheRetention) > 0,
		"safety_identifier":      len(wire.SafetyIdentifier) > 0,
		"service_tier":           len(wire.ServiceTier) > 0,
		"store":                  len(wire.Store) > 0, "temperature": len(wire.Temperature) > 0,
		"text": len(wire.Text) > 0, "tool_choice": len(wire.ToolChoice) > 0,
		"tools": len(wire.Tools) > 0, "top_logprobs": len(wire.TopLogprobs) > 0,
		"top_p":      len(wire.TopP) > 0,
		"truncation": len(wire.Truncation) > 0, "user": len(wire.User) > 0,
	}, "response request-echo metadata is not model output")
	return diagnostics
}

func decodeResponsesResponseResource(
	wire responsesResponseWire,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) (llmprotocol.Response, error) {
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.CreatedAt > 0 {
		response.CreatedAt = time.Unix(wire.CreatedAt, 0).UTC()
	}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: decodeProviderErrorCategory(wire.Error.Code), Code: wire.Error.Code, Message: wire.Error.Message}
		response.StopReason = llmprotocol.StopError
	}
	output, err := decodeResponsesOutput(wire.Output, policy, diagnostics)
	if err != nil {
		return llmprotocol.Response{}, err
	}
	response.Output = output
	if wire.Usage != nil {
		response.Usage = decodeResponsesUsage(*wire.Usage)
		appendProviderFieldOmissions(diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
			"usage.compute_units": len(wire.Usage.ComputeUnits) > 0,
		}, "compute-unit accounting has no protocol-neutral token representation")
	}
	if response.Error == nil {
		response.StopReason = llmprotocol.StopEndTurn
		if wire.Status == "completed" && responsesOutputHasToolCall(output) {
			response.StopReason = llmprotocol.StopToolCall
		}
	}
	response.SourceStopReason = wire.Status
	response.StopReason = decodeResponsesStopReason(wire, response.StopReason)
	return response, nil
}

func responsesOutputHasToolCall(output []llmprotocol.OutputItem) bool {
	for _, item := range output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil {
				return true
			}
		}
	}
	return false
}

func decodeResponsesStopReason(wire responsesResponseWire, fallback llmprotocol.StopReason) llmprotocol.StopReason {
	if wire.Status == "incomplete" && wire.IncompleteDetails != nil {
		switch wire.IncompleteDetails.Reason {
		case "max_output_tokens":
			return llmprotocol.StopMaxTokens
		case "content_filter":
			return llmprotocol.StopContentFilter
		default:
			return llmprotocol.StopUnknown
		}
	}
	switch wire.Status {
	case "failed":
		return llmprotocol.StopError
	case "cancelled":
		return llmprotocol.StopCanceled
	default:
		return fallback
	}
}

func decodeResponsesOutput(
	raw json.RawMessage,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) ([]llmprotocol.OutputItem, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var itemBodies []json.RawMessage
	if err := decodeProviderValue(raw, &itemBodies, policy); err != nil {
		return nil, err
	}
	output := make([]llmprotocol.OutputItem, 0, len(itemBodies))
	for index, itemBody := range itemBodies {
		item, err := decodeResponsesItemWire(itemBody, policy, true)
		if err != nil {
			return nil, err
		}
		if err := validateResponsesOutputItemResource(itemBody, item, policy.Limits); err != nil {
			return nil, err
		}
		decoded, err := decodeResponsesOutputItem(item, index, policy, diagnostics)
		if err != nil {
			return nil, err
		}
		output = append(output, decoded)
	}
	return output, nil
}

func decodeResponsesOutputItem(item responsesItemWire, index int, policy llmprotocol.Policy, diagnostics *llmprotocol.Diagnostics) (llmprotocol.OutputItem, error) {
	if err := validateResponsesOutputItemStatus(item); err != nil {
		return llmprotocol.OutputItem{}, err
	}
	appendProviderFieldOmissions(diagnostics, policy, llmprotocol.OpenAIResponsesV1, map[string]bool{
		"output.caller":            len(item.Caller) > 0,
		"output.encrypted_content": len(item.EncryptedContent) > 0,
		"output.namespace":         item.Namespace != "",
		"output.phase":             len(item.Phase) > 0,
		"output.status":            item.Status != "" && item.Type != "image_generation_call",
	}, "response item metadata has no protocol-neutral representation")
	id := item.ID
	if id == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		id = llmprotocol.StableID("response-output", fmt.Sprint(index), item.Type, item.CallID)
	}
	output := llmprotocol.OutputItem{ID: id, Role: llmprotocol.RoleAssistant}
	switch item.Type {
	case "message":
		return decodeResponsesMessageOutput(output, item, policy, diagnostics)
	case "function_call":
		output.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments}}}
	case "reasoning":
		return decodeResponsesReasoningOutput(output, item, policy)
	case "image_generation_call":
		output.Content = []llmprotocol.Content{{
			Kind:           llmprotocol.ContentGeneratedImage,
			GeneratedImage: decodeResponsesGeneratedImage(item),
		}}
	default:
		return llmprotocol.OutputItem{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_item", "Responses output item is unsupported", nil)
	}
	return output, nil
}

func decodeResponsesGeneratedImage(item responsesItemWire) *llmprotocol.GeneratedImage {
	image := &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationStatus(item.Status)}
	if item.Result != nil {
		result := *item.Result
		image.Result = &result
	}
	return image
}

func decodeResponsesMessageOutput(
	output llmprotocol.OutputItem,
	item responsesItemWire,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) (llmprotocol.OutputItem, error) {
	role, err := canonicalRole(item.Role)
	if err != nil {
		return llmprotocol.OutputItem{}, invalidProviderResponse("invalid_response_role", "Responses output messages must use the assistant role")
	}
	if role != llmprotocol.RoleAssistant {
		return llmprotocol.OutputItem{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_response_role",
			"Responses output messages must use the assistant role",
			nil,
		)
	}
	content, err := decodeResponsesContent(item.Content, policy, responsesProviderOutputContent)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	if responsesContentFieldPresent(item.Content, "logprobs") {
		appendProviderFieldOmission(diagnostics, policy, llmprotocol.OpenAIResponsesV1, "output.content.logprobs", "token log probabilities have no protocol-neutral representation")
	}
	output.Role, output.Content = role, content
	return output, nil
}

func decodeResponsesReasoningOutput(
	output llmprotocol.OutputItem,
	item responsesItemWire,
	policy llmprotocol.Policy,
) (llmprotocol.OutputItem, error) {
	content, err := decodeResponsesReasoning(item.Summary, policy, true)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	reasoning, err := decodeResponsesContent(item.Content, policy, responsesProviderReasoningContent)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	output.Content = append(content, reasoning...)
	return output, nil
}

func responsesContentFieldPresent(raw json.RawMessage, field string) bool {
	var parts []map[string]json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return false
	}
	for _, part := range parts {
		if value, present := part[field]; present && len(value) > 0 {
			return true
		}
	}
	return false
}

func decodeResponsesUsage(wire responsesUsageWire) llmprotocol.Usage {
	cached, cacheWrite := int64(0), int64(0)
	if wire.InputTokensDetails != nil {
		cached = wire.InputTokensDetails.CachedTokens
		cacheWrite = wire.InputTokensDetails.CacheWriteTokens
	}
	reasoning := int64(0)
	if wire.OutputTokensDetails != nil {
		reasoning = wire.OutputTokensDetails.ReasoningTokens
	}
	uncached := int64(-1)
	if cached >= 0 && cacheWrite >= 0 && wire.InputTokens >= cached && cacheWrite <= wire.InputTokens-cached {
		uncached = wire.InputTokens - cached - cacheWrite
	}
	other := int64(-1)
	if reasoning >= 0 && wire.OutputTokens >= reasoning {
		other = wire.OutputTokens - reasoning
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(uncached), InputCacheRead: authoritative(cached), InputCacheWrite: authoritative(cacheWrite),
		OutputReasoning: authoritative(reasoning), OutputOther: authoritative(other),
		InputTotal: authoritative(wire.InputTokens), OutputTotal: authoritative(wire.OutputTokens), Total: authoritative(wire.TotalTokens),
	}
}

func (OpenAIResponsesCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if envelope.CanReplay(llmprotocol.OpenAIResponsesV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	if response.Error != nil {
		body, err := encodeResponsesErrorResource(response, envelope)
		return body, nil, err
	}
	var diagnostics llmprotocol.Diagnostics
	if len(response.Alternatives) > 0 {
		if err := appendLossy(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIResponsesV1, "response.alternatives", "Responses has one primary output sequence"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire, err := encodeResponsesSuccessResource(response, envelope)
	if err != nil {
		return nil, diagnostics, err
	}
	wire.OutputText = encodeResponsesOutputText(response.Output)
	wire.Usage = encodeResponsesUsage(response.Usage)
	if err := applyResponsesStopReason(&wire, response.StopReason, envelope.Format, policy, &diagnostics); err != nil {
		return nil, diagnostics, err
	}
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func encodeResponsesErrorResource(response llmprotocol.Response, envelope llmprotocol.Envelope) ([]byte, error) {
	wire := newResponsesResponseWire(
		response.ID,
		response.Model,
		"failed",
		responsesCreatedAt(response.CreatedAt),
		envelope.ResponseRender.PreviousResponseID,
	)
	wire.Error = &responsesErrorWire{Code: responsesErrorCode(response.Error), Message: response.Error.Message}
	if response.Usage.State == llmprotocol.UsageAvailable {
		wire.Usage = encodeResponsesUsage(response.Usage)
	}
	return marshalWire(wire)
}

func encodeResponsesSuccessResource(response llmprotocol.Response, envelope llmprotocol.Envelope) (responsesResponseWire, error) {
	wire := newResponsesResponseWire(
		response.ID,
		response.Model,
		"completed",
		responsesCreatedAt(response.CreatedAt),
		envelope.ResponseRender.PreviousResponseID,
	)
	items := make([]responsesItemWire, 0, len(response.Output))
	for _, item := range response.Output {
		encoded, err := encodeResponsesOutputItem(item)
		if err != nil {
			return responsesResponseWire{}, err
		}
		items = append(items, encoded...)
	}
	if len(items) == 0 {
		return wire, nil
	}
	encoded, err := json.Marshal(items)
	wire.Output = encoded
	return wire, err
}

func responsesCreatedAt(createdAt time.Time) int64 {
	if createdAt.IsZero() {
		return 0
	}
	return createdAt.Unix()
}

func applyResponsesStopReason(
	wire *responsesResponseWire,
	stopReason llmprotocol.StopReason,
	source llmprotocol.WireFormat,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) error {
	switch stopReason {
	case llmprotocol.StopMaxTokens, llmprotocol.StopContentFilter:
		wire.Status = "incomplete"
		reason := "max_output_tokens"
		if stopReason == llmprotocol.StopContentFilter {
			reason = "content_filter"
		}
		wire.IncompleteDetails = &struct {
			Reason string `json:"reason"`
		}{Reason: reason}
		return nil
	case llmprotocol.StopPaused, llmprotocol.StopContextWindow, llmprotocol.StopCanceled, llmprotocol.StopUnknown:
		return appendLossy(diagnostics, policy, source, llmprotocol.OpenAIResponsesV1, "response.stop_reason", "Responses cannot represent the source terminal reason")
	default:
		return nil
	}
}

func encodeResponsesOutputText(items []llmprotocol.OutputItem) json.RawMessage {
	var text strings.Builder
	for _, item := range items {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentText {
				text.WriteString(content.Text)
			}
		}
	}
	if text.Len() == 0 {
		return nil
	}
	encoded, _ := json.Marshal(text.String())
	return encoded
}

func encodeResponsesOutputItem(item llmprotocol.OutputItem) ([]responsesItemWire, error) {
	message := llmprotocol.Message(item)
	return encodeResponsesMessage(message, "output")
}

func encodeResponsesUsage(usage llmprotocol.Usage) *responsesUsageWire {
	if usage.State == llmprotocol.UsageUnavailable || usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil && usage.Total.Value == nil {
		return nil
	}
	wire := &responsesUsageWire{
		InputTokens:  tokenValue(usage.InputTotal),
		OutputTokens: tokenValue(usage.OutputTotal),
		TotalTokens:  tokenValue(usage.Total),
		InputTokensDetails: &responsesInputUsageDetails{
			CachedTokens: tokenValue(usage.InputCacheRead), CacheWriteTokens: tokenValue(usage.InputCacheWrite),
		},
		OutputTokensDetails: &responsesOutputUsageDetails{ReasoningTokens: tokenValue(usage.OutputReasoning)},
	}
	if wire.TotalTokens == 0 {
		wire.TotalTokens = wire.InputTokens + wire.OutputTokens
	}
	return wire
}

func (OpenAIResponsesCodec) DecodeTransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	return decodeOpenAITransportError(body, policy)
}

func (OpenAIResponsesCodec) EncodeTransportError(transportError llmprotocol.TransportError) []byte {
	return encodeOpenAITransportError(transportError)
}
