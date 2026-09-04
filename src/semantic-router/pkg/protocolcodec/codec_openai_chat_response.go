package protocolcodec

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (OpenAIChatCodec) DecodeResponse(body []byte, policy llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire chatResponseWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateChatResponseResource(wire); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	if err := validateChatExecutionMetadata(wire.SystemFingerprint); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	response := decodeChatResponseEnvelope(wire)
	var diagnostics llmprotocol.Diagnostics
	appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.OpenAIChatV1, map[string]bool{
		"choices.message.tool_calls.function.TokenizedArguments": chatChoicesHaveTokenizedArguments(wire.Choices),
		"kv_transfer": wire.hasLegacyKVTransferMetadata(),
		"metadata":    len(wire.Metadata) > 0,
		"moderation":  len(wire.Moderation) > 0,
	}, "response request metadata is not model output")
	if err := decodeChatChoices(wire, &response, policy); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, diagnostics, err
	}
	decodeChatResponseUsage(wire.Usage, &response, &diagnostics, policy)
	envelope := responseEnvelope(llmprotocol.OpenAIChatV1, body, response.Generation, response.SourceStopReason, policy)
	return response, envelope, diagnostics, nil
}

func chatChoicesHaveTokenizedArguments(choices []chatChoiceWire) bool {
	for _, choice := range choices {
		for _, call := range choice.Message.ToolCalls {
			if call.Function.TokenizedArguments != nil {
				return true
			}
		}
	}
	return false
}

func decodeChatResponseUsage(
	wire *chatUsageWire,
	response *llmprotocol.Response,
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
) {
	if wire == nil {
		return
	}
	response.Usage = decodeChatUsage(*wire)
	appendProviderFieldOmissions(diagnostics, policy, llmprotocol.OpenAIChatV1, map[string]bool{
		"usage.compute_units":                                        len(wire.ComputeUnits) > 0,
		"usage.prompt_tokens_details.audio_tokens":                   wire.PromptTokensDetails != nil && wire.PromptTokensDetails.AudioTokens != 0,
		"usage.prompt_tokens_details.image_tokens":                   wire.PromptTokensDetails != nil && wire.PromptTokensDetails.ImageTokens != 0,
		"usage.prompt_tokens_details.text_tokens":                    wire.PromptTokensDetails != nil && wire.PromptTokensDetails.TextTokens != 0,
		"usage.completion_tokens_details.accepted_prediction_tokens": wire.CompletionTokensDetails != nil && wire.CompletionTokensDetails.AcceptedPredictionTokens != 0,
		"usage.completion_tokens_details.audio_tokens":               wire.CompletionTokensDetails != nil && wire.CompletionTokensDetails.AudioTokens != 0,
		"usage.completion_tokens_details.rejected_prediction_tokens": wire.CompletionTokensDetails != nil && wire.CompletionTokensDetails.RejectedPredictionTokens != 0,
		"usage.completion_tokens_details.text_tokens":                wire.CompletionTokensDetails != nil && wire.CompletionTokensDetails.TextTokens != 0,
	}, "provider accounting detail has no separate protocol-neutral bucket")
}

func decodeChatResponseEnvelope(wire chatResponseWire) llmprotocol.Response {
	response := llmprotocol.Response{
		Generation: 1, ID: wire.ID, Model: wire.Model,
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	}
	if wire.Created > 0 {
		response.CreatedAt = time.Unix(wire.Created, 0).UTC()
	}
	if wire.Error != nil {
		response.Error = &llmprotocol.ProtocolError{
			Category: decodeProviderErrorCategory(wire.Error.Type, wire.Error.Code),
			Code:     wire.Error.Code, Message: wire.Error.Message, Parameter: wire.Error.Param,
		}
		response.StopReason = llmprotocol.StopError
	}
	return response
}

func decodeChatChoices(wire chatResponseWire, response *llmprotocol.Response, policy llmprotocol.Policy) error {
	choices, err := normalizedChatChoices(wire.Choices)
	if err != nil {
		return err
	}
	for choiceIndex, choice := range choices {
		item, err := decodeChatChoiceItem(choice, wire.ID, policy)
		if err != nil {
			return err
		}
		if choiceIndex == 0 {
			response.Output = append(response.Output, item)
			response.Evidence.TokenLogprobs = decodeChatTokenLogprobs(choice.Logprobs)
			if choice.FinishReason != nil {
				response.SourceStopReason = *choice.FinishReason
				response.StopReason = decodeChatStop(*choice.FinishReason)
			}
			continue
		}
		response.Alternatives = append(response.Alternatives, []llmprotocol.OutputItem{item})
	}
	return nil
}

func decodeChatChoiceItem(choice chatChoiceWire, responseID string, policy llmprotocol.Policy) (llmprotocol.OutputItem, error) {
	if err := validateChatChoiceExtensions(choice); err != nil {
		return llmprotocol.OutputItem{}, err
	}
	message, err := decodeChatResponseMessage(choice.Message, choice.Index, policy)
	if err != nil {
		return llmprotocol.OutputItem{}, err
	}
	item := llmprotocol.OutputItem(message)
	if item.ID == "" && policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		item.ID = llmprotocol.StableID("chat-response", responseID, fmt.Sprint(choice.Index))
	}
	return item, nil
}

func decodeChatTokenLogprobs(wire *chatLogprobsWire) []llmprotocol.TokenLogprob {
	if wire == nil {
		return nil
	}
	tokens := make([]llmprotocol.TokenLogprob, 0, len(wire.Content))
	for _, token := range wire.Content {
		decoded := llmprotocol.TokenLogprob{Token: token.Token, Logprob: token.Logprob}
		for _, alternative := range token.TopLogprobs {
			decoded.Alternatives = append(decoded.Alternatives, llmprotocol.TokenLogprobAlternative{
				Token: alternative.Token, Logprob: alternative.Logprob,
			})
		}
		tokens = append(tokens, decoded)
	}
	return tokens
}

func decodeChatUsage(wire chatUsageWire) llmprotocol.Usage {
	usage := llmprotocol.Usage{
		State:           llmprotocol.UsageAvailable,
		InputUncached:   unknownCount(),
		InputCacheRead:  unknownCount(),
		InputCacheWrite: unknownCount(),
		OutputReasoning: unknownCount(),
		OutputOther:     unknownCount(),
		InputTotal:      authoritative(wire.PromptTokens),
		OutputTotal:     authoritative(wire.CompletionTokens),
		Total:           authoritative(wire.TotalTokens),
	}
	if wire.PromptTokensDetails != nil {
		cached, cacheWrite := wire.PromptTokensDetails.CachedTokens, wire.PromptTokensDetails.CacheWriteTokens
		uncached := int64(-1)
		if cached >= 0 && cacheWrite >= 0 && wire.PromptTokens >= cached && cacheWrite <= wire.PromptTokens-cached {
			uncached = wire.PromptTokens - cached - cacheWrite
		}
		usage.InputCacheRead = authoritative(cached)
		usage.InputCacheWrite = authoritative(cacheWrite)
		usage.InputUncached = llmprotocol.TokenCount{
			Value: llmprotocol.Int64(uncached), Provenance: llmprotocol.UsageDerived,
		}
	}
	if wire.CompletionTokensDetails != nil {
		reasoning := wire.CompletionTokensDetails.ReasoningTokens
		other := wire.CompletionTokens - reasoning
		if reasoning < 0 || wire.CompletionTokens < reasoning {
			other = -1
		}
		usage.OutputReasoning = authoritative(reasoning)
		usage.OutputOther = llmprotocol.TokenCount{
			Value: llmprotocol.Int64(other), Provenance: llmprotocol.UsageDerived,
		}
	}
	return usage
}

func (OpenAIChatCodec) EncodeResponse(response llmprotocol.Response, envelope llmprotocol.Envelope, policy llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if response.Error != nil {
		var diagnostics llmprotocol.Diagnostics
		if response.Usage.State == llmprotocol.UsageAvailable {
			appendAccountingOmission(&diagnostics, policy, envelope.Format, llmprotocol.OpenAIChatV1, "usage", "Chat Completions error envelopes cannot carry token usage")
		}
		return OpenAIChatCodec{}.EncodeTransportError(llmprotocol.TransportError{Error: response.Error}), diagnostics, nil
	}
	if envelope.CanReplay(llmprotocol.OpenAIChatV1, response.Generation, policy, true) {
		return append([]byte(nil), envelope.Response...), nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	if err := appendChatStopReasonDiagnostic(&diagnostics, response.StopReason, envelope.Format, policy); err != nil {
		return nil, diagnostics, err
	}
	wire := chatResponseWire{ID: response.ID, Object: "chat.completion", Model: response.Model}
	if !response.CreatedAt.IsZero() {
		wire.Created = response.CreatedAt.Unix()
	}
	choices, err := encodeChatChoices(response)
	if err != nil {
		return nil, diagnostics, err
	}
	wire.Choices = choices
	wire.Usage = encodeChatUsage(response.Usage)
	body, err := marshalWire(wire)
	return body, diagnostics, err
}

func appendChatStopReasonDiagnostic(
	diagnostics *llmprotocol.Diagnostics,
	stopReason llmprotocol.StopReason,
	source llmprotocol.WireFormat,
	policy llmprotocol.Policy,
) error {
	switch stopReason {
	case llmprotocol.StopPaused, llmprotocol.StopContextWindow, llmprotocol.StopCanceled, llmprotocol.StopUnknown:
		return appendLossy(
			diagnostics, policy, source, llmprotocol.OpenAIChatV1,
			"response.stop_reason", "Chat Completions cannot represent the source terminal reason",
		)
	default:
		return nil
	}
}

func encodeChatChoices(response llmprotocol.Response) ([]chatChoiceWire, error) {
	primary, err := encodeChatOutput(response.Output, 0, response.StopReason)
	if err != nil {
		return nil, err
	}
	choices := []chatChoiceWire{primary}
	for index, alternative := range response.Alternatives {
		choice, err := encodeChatOutput(alternative, index+1, response.StopReason)
		if err != nil {
			return nil, err
		}
		choices = append(choices, choice)
	}
	return choices, nil
}

func encodeChatOutput(items []llmprotocol.OutputItem, index int, stop llmprotocol.StopReason) (chatChoiceWire, error) {
	combined := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
	for _, item := range items {
		if item.Role != "" {
			combined.Role = item.Role
		}
		combined.Content = append(combined.Content, item.Content...)
	}
	message, err := encodeChatMessage(combined)
	if err != nil {
		return chatChoiceWire{}, err
	}
	reason := encodeChatStop(stop)
	return chatChoiceWire{Index: index, Message: message, FinishReason: &reason}, nil
}

func encodeChatUsage(usage llmprotocol.Usage) *chatUsageWire {
	if usage.State == llmprotocol.UsageUnavailable || usage.InputTotal.Value == nil && usage.OutputTotal.Value == nil && usage.Total.Value == nil {
		return nil
	}
	prompt := tokenValue(usage.InputTotal)
	completion := tokenValue(usage.OutputTotal)
	total := tokenValue(usage.Total)
	if total == 0 {
		total = prompt + completion
	}
	wire := &chatUsageWire{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: total}
	if usage.InputCacheRead.Value != nil || usage.InputCacheWrite.Value != nil {
		wire.PromptTokensDetails = &chatPromptTokensDetailsWire{
			CachedTokens: tokenValue(usage.InputCacheRead), CacheWriteTokens: tokenValue(usage.InputCacheWrite),
		}
	}
	if usage.OutputReasoning.Value != nil {
		wire.CompletionTokensDetails = &chatCompletionTokensDetailsWire{ReasoningTokens: tokenValue(usage.OutputReasoning)}
	}
	return wire
}

func decodeChatStop(reason string) llmprotocol.StopReason {
	switch reason {
	case "stop":
		return llmprotocol.StopEndTurn
	case "length":
		return llmprotocol.StopMaxTokens
	case "tool_calls", "function_call":
		return llmprotocol.StopToolCall
	case "content_filter":
		return llmprotocol.StopContentFilter
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeChatStop(reason llmprotocol.StopReason) string {
	switch reason {
	case llmprotocol.StopMaxTokens:
		return "length"
	case llmprotocol.StopToolCall:
		return "tool_calls"
	case llmprotocol.StopContentFilter:
		return "content_filter"
	default:
		return "stop"
	}
}

func (OpenAIChatCodec) DecodeTransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	return decodeOpenAITransportError(body, policy)
}

func (OpenAIChatCodec) EncodeTransportError(transportError llmprotocol.TransportError) []byte {
	return encodeOpenAITransportError(transportError)
}
