package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type anthropicResponseWire struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Role         string              `json:"role"`
	Model        string              `json:"model"`
	Content      json.RawMessage     `json:"content"`
	StopReason   *string             `json:"stop_reason"`
	StopSequence *string             `json:"stop_sequence"`
	Usage        *anthropicUsageWire `json:"usage"`
	Error        *anthropicErrorWire `json:"error,omitempty"`
	Container    json.RawMessage     `json:"container"`
	StopDetails  json.RawMessage     `json:"stop_details"`
	// ContextManagement reports the context edits the upstream applied to the
	// billed prompt. It only appears when the request carried the directive, so
	// unlike container and stop_details it is omitted rather than null-valued.
	ContextManagement json.RawMessage `json:"context_management,omitempty"`
}

type anthropicUsageWire struct {
	InputTokens              int64                           `json:"input_tokens"`
	OutputTokens             int64                           `json:"output_tokens"`
	CacheCreationInputTokens int64                           `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64                           `json:"cache_read_input_tokens"`
	CacheCreation            anthropicCacheCreationUsageWire `json:"cache_creation"`
	InferenceGeo             string                          `json:"inference_geo"`
	OutputTokensDetails      anthropicOutputUsageDetailsWire `json:"output_tokens_details"`
	ServerToolUse            anthropicServerToolUsageWire    `json:"server_tool_use"`
	ServiceTier              string                          `json:"service_tier"`
}

type anthropicCacheCreationUsageWire struct {
	Ephemeral1hInputTokens int64 `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens int64 `json:"ephemeral_5m_input_tokens"`
}

type anthropicOutputUsageDetailsWire struct {
	ThinkingTokens int64 `json:"thinking_tokens"`
}

type anthropicServerToolUsageWire struct {
	WebFetchRequests  int64 `json:"web_fetch_requests"`
	WebSearchRequests int64 `json:"web_search_requests"`
}

type anthropicErrorWire struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicTransportErrorWire struct {
	Type      string              `json:"type"`
	Error     *anthropicErrorWire `json:"error"`
	RequestID string              `json:"request_id,omitempty"`
}

func (AnthropicMessagesCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire anthropicResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateAnthropicResponseResource(wire); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	diagnostics := anthropicResponseMetadataDiagnostics(wire, policy)
	response, err := decodeAnthropicResponseResource(wire, policy)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	appendAnthropicResponseUsage(&response, wire.Usage, policy, &diagnostics)
	return response, responseEnvelope(llmprotocol.AnthropicMessagesV1, body, response.Generation, response.SourceStopReason, policy), diagnostics, nil
}

func anthropicResponseMetadataDiagnostics(wire anthropicResponseWire, policy llmprotocol.Policy) llmprotocol.Diagnostics {
	var diagnostics llmprotocol.Diagnostics
	if len(wire.Container) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Container), []byte("null")) {
		appendProviderFieldOmission(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, "container", "container execution metadata is not model output")
	}
	if len(wire.StopDetails) > 0 && !bytes.Equal(bytes.TrimSpace(wire.StopDetails), []byte("null")) {
		appendProviderFieldOmission(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, "stop_details", "structured refusal detail has no neutral representation")
	}
	if len(wire.ContextManagement) > 0 && !bytes.Equal(bytes.TrimSpace(wire.ContextManagement), []byte("null")) {
		appendProviderFieldOmission(&diagnostics, policy, llmprotocol.AnthropicMessagesV1, "context_management", "applied context edits describe upstream prompt trimming and have no neutral representation")
	}
	return diagnostics
}

func decodeAnthropicResponseResource(wire anthropicResponseWire, policy llmprotocol.Policy) (llmprotocol.Response, error) {
	response := llmprotocol.Response{Generation: 1, ID: wire.ID, Model: wire.Model, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{Category: decodeProviderErrorCategory(wire.Error.Type), Code: wire.Error.Type, Message: wire.Error.Message}
		response.StopReason = llmprotocol.StopError
	}
	contents, err := decodeAnthropicResponseContent(wire.Content, policy)
	if err != nil && wire.Error == nil {
		return llmprotocol.Response{}, err
	}
	if len(contents) > 0 {
		response.Output = []llmprotocol.OutputItem{{ID: llmprotocol.StableID("anthropic-response", wire.ID), Role: llmprotocol.RoleAssistant, Content: contents}}
	}
	if wire.StopReason != nil && response.Error == nil {
		response.SourceStopReason = *wire.StopReason
		response.StopReason = decodeAnthropicStop(*wire.StopReason)
		if response.StopReason == llmprotocol.StopSequence && wire.StopSequence != nil {
			response.MatchedStopSequence = *wire.StopSequence
		}
	}
	return response, nil
}

func appendAnthropicResponseUsage(
	response *llmprotocol.Response,
	usage *anthropicUsageWire,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) {
	if usage == nil {
		return
	}
	response.Usage = decodeAnthropicUsage(*usage)
	appendProviderFieldOmissions(diagnostics, policy, llmprotocol.AnthropicMessagesV1, map[string]bool{
		"usage.cache_creation": usage.CacheCreation.Ephemeral1hInputTokens != 0 ||
			usage.CacheCreation.Ephemeral5mInputTokens != 0,
		"usage.inference_geo": usage.InferenceGeo != "",
		"usage.server_tool_use": usage.ServerToolUse.WebFetchRequests != 0 ||
			usage.ServerToolUse.WebSearchRequests != 0,
		"usage.service_tier": usage.ServiceTier != "",
	}, "provider usage metadata has no neutral accounting bucket")
}

func decodeAnthropicUsage(wire anthropicUsageWire) llmprotocol.Usage {
	inputTotal := wire.InputTokens + wire.CacheReadInputTokens + wire.CacheCreationInputTokens
	reasoning := int64(0)
	reasoning = wire.OutputTokensDetails.ThinkingTokens
	other := wire.OutputTokens - reasoning
	if other < 0 {
		other = 0
	}
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(wire.InputTokens), InputCacheRead: authoritative(wire.CacheReadInputTokens), InputCacheWrite: authoritative(wire.CacheCreationInputTokens),
		OutputReasoning: authoritative(reasoning), OutputOther: authoritative(other),
		InputTotal: authoritative(inputTotal), OutputTotal: authoritative(wire.OutputTokens), Total: llmprotocol.TokenCount{Value: llmprotocol.Int64(inputTotal + wire.OutputTokens), Provenance: llmprotocol.UsageDerived},
	}
}

func (AnthropicMessagesCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if response.Error != nil {
		var diagnostics llmprotocol.Diagnostics
		if response.Usage.State == llmprotocol.UsageAvailable {
			appendAccountingOmission(&diagnostics, policy, envelope.Format, llmprotocol.AnthropicMessagesV1, "usage", "Messages error envelopes cannot carry token usage")
		}
		return encodeAnthropicError(response.Error, response.ProviderRequestID), diagnostics, nil
	}
	if envelope.CanReplay(llmprotocol.AnthropicMessagesV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if usageUnavailable(response.Usage) {
		if err := appendLossy(
			&diagnostics, policy, envelope.Format, llmprotocol.AnthropicMessagesV1,
			"usage", "Messages requires usage; emitted an explicit zero-valued usage object",
		); err != nil {
			return nil, diagnostics, err
		}
	}
	if len(response.Alternatives) > 0 {
		if err := appendLossy(&diagnostics, policy, envelope.Format, llmprotocol.AnthropicMessagesV1, "response.alternatives", "Messages has one output sequence"); err != nil {
			return nil, diagnostics, err
		}
	}
	contents := make([]llmprotocol.Content, 0)
	for _, item := range response.Output {
		contents = append(contents, item.Content...)
	}
	contentDiagnostics, err := anthropicContentDiagnostics(contents, envelope.Format, policy)
	diagnostics = appendDiagnostics(diagnostics, contentDiagnostics, policy.Limits.Diagnostics)
	if err != nil {
		return nil, diagnostics, err
	}
	content, err := encodeAnthropicContent(contents)
	if err != nil {
		return nil, diagnostics, err
	}
	stop := encodeAnthropicStop(response.StopReason)
	wire := anthropicResponseWire{ID: response.ID, Type: "message", Role: "assistant", Model: response.Model, Content: content, StopReason: &stop, Usage: encodeAnthropicUsage(response.Usage)}
	if response.StopReason == llmprotocol.StopSequence {
		wire.StopSequence = &response.MatchedStopSequence
	}
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func encodeAnthropicUsage(usage llmprotocol.Usage) *anthropicUsageWire {
	wire := newAnthropicUsageWire()
	if usageUnavailable(usage) {
		return wire
	}
	inputTokens := tokenValue(usage.InputUncached)
	if usage.InputUncached.Value == nil {
		inputTokens = tokenValue(usage.InputTotal)
	}
	cacheWrite := tokenValue(usage.InputCacheWrite)
	*wire = anthropicUsageWire{
		InputTokens: inputTokens, OutputTokens: tokenValue(usage.OutputTotal),
		CacheCreationInputTokens: cacheWrite, CacheReadInputTokens: tokenValue(usage.InputCacheRead),
		CacheCreation:       anthropicCacheCreationUsageWire{Ephemeral5mInputTokens: cacheWrite},
		InferenceGeo:        "global",
		OutputTokensDetails: anthropicOutputUsageDetailsWire{ThinkingTokens: tokenValue(usage.OutputReasoning)},
		ServerToolUse:       anthropicServerToolUsageWire{},
		ServiceTier:         "standard",
	}
	return wire
}

func newAnthropicUsageWire() *anthropicUsageWire {
	return &anthropicUsageWire{
		InferenceGeo: "global",
		ServiceTier:  "standard",
	}
}

func usageUnavailable(usage llmprotocol.Usage) bool {
	return usage.State == llmprotocol.UsageUnavailable ||
		usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil
}

func decodeAnthropicStop(reason string) llmprotocol.StopReason {
	switch reason {
	case "end_turn":
		return llmprotocol.StopEndTurn
	case "max_tokens":
		return llmprotocol.StopMaxTokens
	case "stop_sequence":
		return llmprotocol.StopSequence
	case "tool_use":
		return llmprotocol.StopToolCall
	case "refusal":
		return llmprotocol.StopContentFilter
	case "pause_turn":
		return llmprotocol.StopPaused
	case "model_context_window_exceeded":
		return llmprotocol.StopContextWindow
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeAnthropicStop(reason llmprotocol.StopReason) string {
	switch reason {
	case llmprotocol.StopMaxTokens:
		return "max_tokens"
	case llmprotocol.StopSequence:
		return "stop_sequence"
	case llmprotocol.StopToolCall:
		return "tool_use"
	case llmprotocol.StopContentFilter:
		return "refusal"
	case llmprotocol.StopPaused:
		return "pause_turn"
	case llmprotocol.StopContextWindow:
		return "model_context_window_exceeded"
	default:
		return "end_turn"
	}
}

func encodeAnthropicError(protocolError *llmprotocol.ProtocolError, requestID string) []byte {
	return AnthropicMessagesCodec{}.EncodeTransportError(llmprotocol.TransportError{
		Error: protocolError, ProviderRequestID: requestID,
	})
}

func (AnthropicMessagesCodec) DecodeTransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire anthropicTransportErrorWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	if wire.Type != "error" {
		return llmprotocol.TransportError{}, nil, invalidProviderResponse(
			"invalid_upstream_error_envelope",
			"Anthropic transport error type must be error",
		)
	}
	if wire.Error == nil {
		return llmprotocol.TransportError{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_error_required",
			"upstream transport error body is missing error details",
			nil,
		)
	}
	if err := validateTransportErrorDetails(wire.Error.Type, wire.Error.Message); err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	return llmprotocol.TransportError{
		Error: &llmprotocol.ProtocolError{
			Category: decodeProviderErrorCategory(wire.Error.Type),
			Code:     wire.Error.Type, Message: wire.Error.Message,
		},
		ProviderRequestID: wire.RequestID,
	}, nil, nil
}

func (AnthropicMessagesCodec) EncodeTransportError(transportError llmprotocol.TransportError) []byte {
	protocolError := transportError.Error
	if protocolError == nil {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "internal", "request failed", nil)
	}
	wire := anthropicTransportErrorWire{
		Type: "error", RequestID: transportError.ProviderRequestID,
		Error: &anthropicErrorWire{Type: canonicalAnthropicErrorType(protocolError), Message: protocolError.Message},
	}
	body, _ := json.Marshal(wire)
	return body
}

func canonicalAnthropicErrorType(protocolError *llmprotocol.ProtocolError) string {
	if protocolError == nil {
		return "api_error"
	}
	if canonicalAnthropicErrorTypeMatchesCategory(protocolError.Code, protocolError.Category) {
		return protocolError.Code
	}
	switch protocolError.Category {
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorUnsupportedFeature:
		return "invalid_request_error"
	case llmprotocol.ErrorAuthentication:
		return "authentication_error"
	case llmprotocol.ErrorPermission:
		return "permission_error"
	case llmprotocol.ErrorNotFound:
		return "not_found_error"
	case llmprotocol.ErrorConflict:
		return "conflict_error"
	case llmprotocol.ErrorRateLimited:
		return "rate_limit_error"
	case llmprotocol.ErrorUpstreamTimeout:
		return "timeout_error"
	default:
		return "api_error"
	}
}

func canonicalAnthropicErrorTypeMatchesCategory(code string, category llmprotocol.ErrorCategory) bool {
	categories := map[string][]llmprotocol.ErrorCategory{
		"invalid_request_error": {llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorUnsupportedFeature},
		"request_too_large":     {llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorUnsupportedFeature},
		"authentication_error":  {llmprotocol.ErrorAuthentication},
		"billing_error":         {llmprotocol.ErrorUpstreamUnavailable},
		"permission_error":      {llmprotocol.ErrorPermission},
		"not_found_error":       {llmprotocol.ErrorNotFound},
		"conflict_error":        {llmprotocol.ErrorConflict},
		"rate_limit_error":      {llmprotocol.ErrorRateLimited},
		"timeout_error":         {llmprotocol.ErrorUpstreamTimeout},
		"api_error":             {llmprotocol.ErrorUpstreamUnavailable, llmprotocol.ErrorInternal},
		"overloaded_error":      {llmprotocol.ErrorUpstreamUnavailable, llmprotocol.ErrorInternal},
	}
	for _, allowed := range categories[code] {
		if allowed == category {
			return true
		}
	}
	return false
}
