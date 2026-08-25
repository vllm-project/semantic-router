package protocolcodec

import (
	"bytes"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type chatStreamDecoder struct {
	streamState
	framer sseFramer
}
type chatStreamEncoder struct {
	streamState
	itemStarted bool
	toolIndexes map[string]int
}

func (OpenAIChatCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &chatStreamDecoder{streamState: streamState{context: context, policy: policy}, framer: newSSEFramer(policy.Limits.SSEFrameBytes)}
}

func (OpenAIChatCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &chatStreamEncoder{streamState: streamState{context: context, policy: policy}, toolIndexes: make(map[string]int)}
}

type chatChunkWire struct {
	ID                string                    `json:"id"`
	Object            string                    `json:"object,omitempty"`
	Created           int64                     `json:"created,omitempty"`
	Model             string                    `json:"model,omitempty"`
	Choices           []chatChunkChoiceWire     `json:"choices,omitempty"`
	Usage             *chatUsageWire            `json:"usage,omitempty"`
	Error             *chatErrorWire            `json:"error,omitempty"`
	ServiceTier       *chatServiceTierWire      `json:"service_tier,omitempty"`
	SystemFingerprint *string                   `json:"system_fingerprint,omitempty"`
	PromptLogprobs    *chatNullOnlyWire         `json:"prompt_logprobs,omitempty"`
	PromptTokenIDs    []int64                   `json:"prompt_token_ids,omitempty"`
	KVTransferParams  *chatKVTransferParamsWire `json:"kv_transfer_params,omitempty"`
}

type chatChunkChoiceWire struct {
	Index        int                 `json:"index"`
	Delta        chatChunkDeltaWire  `json:"delta"`
	FinishReason *string             `json:"finish_reason"`
	Logprobs     *chatLogprobsWire   `json:"logprobs,omitempty"`
	StopReason   *chatStopReasonWire `json:"stop_reason,omitempty"`
	TokenIDs     []int64             `json:"token_ids,omitempty"`
}

type chatChunkDeltaWire struct {
	Role               string                  `json:"role,omitempty"`
	Content            *string                 `json:"content,omitempty"`
	Reasoning          *string                 `json:"reasoning_content,omitempty"`
	AlternateReasoning *string                 `json:"reasoning,omitempty"`
	Refusal            *string                 `json:"refusal,omitempty"`
	Audio              *chatAudioOutputWire    `json:"audio,omitempty"`
	LegacyFunctionCall *chatLegacyCallWire     `json:"function_call,omitempty"`
	ToolCalls          []chatChunkToolCallWire `json:"tool_calls,omitempty"`
	Annotations        []chatAnnotationWire    `json:"annotations,omitempty"`
}

type chatChunkToolCallWire struct {
	Index    int              `json:"index"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function chatFunctionWire `json:"function"`
}

func (decoder *chatStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := decoder.framer.Push(chunk)
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := decoder.pushFrame(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, decoder.policy.Limits.Diagnostics)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func (decoder *chatStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := parseSSEFrame(frame, decoder.policy.Limits.SSEFrameBytes)
	if err != nil || len(parsed.Data) == 0 {
		return nil, nil, err
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
		return []llmprotocol.Event{event}, nil, err
	}
	var chunk chatChunkWire
	if err := decodeProviderWire(parsed.Data, &chunk, decoder.policy); err != nil {
		return nil, nil, err
	}
	if err := validateChatExecutionMetadata(chunk.SystemFingerprint); err != nil {
		return nil, nil, err
	}
	for _, choice := range chunk.Choices {
		if err := validateChatChunkChoiceExtensions(choice); err != nil {
			return nil, nil, err
		}
	}
	if chunk.Error != nil {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: &llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable, Code: chunk.Error.Code, Message: chunk.Error.Message}, StopReason: llmprotocol.StopError})
		return []llmprotocol.Event{event}, nil, err
	}
	events := make([]llmprotocol.Event, 0, 8)
	if !decoder.started {
		event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseStarted, ResponseID: chunk.ID, Model: chunk.Model})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, event)
	}
	for _, choice := range chunk.Choices {
		choiceEvents, choiceErr := decoder.decodeChoice(choice)
		if choiceErr != nil {
			return nil, nil, choiceErr
		}
		events = append(events, choiceEvents...)
	}
	if chunk.Usage != nil {
		usage := decodeChatUsage(*chunk.Usage)
		event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &usage})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, event)
	}
	return events, nil, nil
}

func (decoder *chatStreamDecoder) decodeChoice(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	if choice.Index != 0 {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "stream_multiple_choices", "streaming multiple choices is unsupported", nil)
	}
	events := make([]llmprotocol.Event, 0, 8)
	needsItem := choice.Delta.Role != "" || choice.Delta.Content != nil || len(choice.Delta.Annotations) > 0 ||
		choice.Delta.Reasoning != nil || choice.Delta.AlternateReasoning != nil || choice.Delta.Refusal != nil
	if needsItem && !decoder.items[choice.Index] {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: choice.Index, Role: llmprotocol.RoleAssistant})
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	if choice.Delta.Content != nil {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index, Delta: *choice.Delta.Content})
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	annotationEvents, err := decoder.decodeAnnotations(choice)
	if err != nil {
		return nil, err
	}
	events = append(events, annotationEvents...)
	reasoning := choice.Delta.Reasoning
	if reasoning == nil {
		reasoning = choice.Delta.AlternateReasoning
	}
	if reasoning != nil {
		event, eventErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, ItemIndex: choice.Index, Delta: *reasoning})
		if eventErr != nil {
			return nil, eventErr
		}
		events = append(events, event)
	}
	if choice.Delta.Refusal != nil {
		content := llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: *choice.Delta.Refusal}
		event, eventErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index, Delta: *choice.Delta.Refusal, Content: &content})
		if eventErr != nil {
			return nil, eventErr
		}
		events = append(events, event)
	}
	toolEvents, err := decoder.decodeToolCalls(choice.Delta.ToolCalls)
	if err != nil {
		return nil, err
	}
	events = append(events, toolEvents...)
	completed, err := decoder.completeChoice(choice.FinishReason)
	return append(events, completed...), err
}

func (decoder *chatStreamDecoder) decodeAnnotations(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	if len(choice.Delta.Annotations) == 0 {
		return nil, nil
	}
	citations, err := decodeChatAnnotations(choice.Delta.Annotations)
	if err != nil {
		return nil, err
	}
	content := llmprotocol.Content{Kind: llmprotocol.ContentText, Citations: citations}
	event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index, Content: &content})
	return []llmprotocol.Event{event}, err
}

func (decoder *chatStreamDecoder) decodeToolCalls(calls []chatChunkToolCallWire) ([]llmprotocol.Event, error) {
	events := make([]llmprotocol.Event, 0, len(calls)*2)
	for _, call := range calls {
		itemIndex := call.Index + 1
		if !decoder.items[itemIndex] {
			started, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: itemIndex, Role: llmprotocol.RoleAssistant, ToolCall: &llmprotocol.ToolCall{ID: call.ID, Name: call.Function.Name}})
			if err != nil {
				return nil, err
			}
			events = append(events, started)
		}
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventToolCallDelta, ItemIndex: itemIndex, ToolCall: &llmprotocol.ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: call.Function.Arguments}})
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	return events, nil
}

func (decoder *chatStreamDecoder) completeChoice(reason *string) ([]llmprotocol.Event, error) {
	if reason == nil {
		return nil, nil
	}
	decoder.stop = decodeChatStop(*reason)
	active := make([]int, 0, len(decoder.items))
	for itemIndex := range decoder.items {
		if !decoder.completedItems[itemIndex] {
			active = append(active, itemIndex)
		}
	}
	sort.Ints(active)
	events := make([]llmprotocol.Event, 0, len(active))
	for _, itemIndex := range active {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: itemIndex, StopReason: decoder.stop})
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	return events, nil
}

func (decoder *chatStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *chatStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, err := encoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	event = normalized
	chunk := chatChunkWire{ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model}
	choice := chatChunkChoiceWire{Index: 0}
	emitChunk := true
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		choice.Delta.Role = "assistant"
		encoder.itemStarted = true
	case llmprotocol.EventOutputItemStarted:
		if encoder.itemStarted {
			return nil, nil, nil
		}
		choice.Delta.Role = "assistant"
		encoder.itemStarted = true
	case llmprotocol.EventOutputTextDelta:
		if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
			choice.Delta.Refusal = &event.Delta
		} else {
			choice.Delta.Content = &event.Delta
		}
		if event.Content != nil && len(event.Content.Citations) > 0 {
			choice.Delta.Annotations = encodeChatAnnotations(event.Content.Citations)
		}
	case llmprotocol.EventReasoningDelta:
		if event.Content != nil && event.Content.Signature != "" {
			var diagnostics llmprotocol.Diagnostics
			if lossyErr := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "reasoning.signature", "Chat Completions cannot represent a signed reasoning delta"); lossyErr != nil {
				return nil, diagnostics, lossyErr
			}
			choice.Delta.Reasoning = &event.Delta
			frame, encodeErr := encodeSSE("", chatChunkWire{ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model, Choices: []chatChunkChoiceWire{choice}})
			return [][]byte{frame}, diagnostics, encodeErr
		}
		choice.Delta.Reasoning = &event.Delta
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
		}
		index, found := encoder.toolIndexes[event.ToolCall.ID]
		if !found {
			index = len(encoder.toolIndexes)
			encoder.toolIndexes[event.ToolCall.ID] = index
		}
		choice.Delta.ToolCalls = []chatChunkToolCallWire{{Index: index, ID: event.ToolCall.ID, Type: "function", Function: chatFunctionWire{Name: event.ToolCall.Name, Arguments: event.ToolCall.Arguments}}}
	case llmprotocol.EventOutputItemCompleted:
		// Chat exposes one choice lifecycle even when the neutral response has
		// several ordered content items (for example reasoning followed by
		// text). Emit the choice finish only at the semantic response terminal;
		// completing an intermediate item must not terminate the public stream.
		return nil, nil, nil
	case llmprotocol.EventUsageUpdated:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "usage event is invalid", nil)
		}
		chunk.Usage = encodeChatUsage(*event.Usage)
		emitChunk = true
	case llmprotocol.EventResponseCompleted:
		return encoder.encodeCompletion(event)
	case llmprotocol.EventResponseFailed:
		encoder.terminal = true
		if event.Error == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
		}
		chunk.Error = &chatErrorWire{Message: event.Error.Message, Type: string(event.Error.Category), Code: event.Error.Code}
	case llmprotocol.EventProviderOpaque:
		if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
		}
		return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
	default:
		emitChunk = false
	}
	if !emitChunk {
		return nil, nil, nil
	}
	if event.Type != llmprotocol.EventUsageUpdated && event.Type != llmprotocol.EventResponseFailed {
		chunk.Choices = []chatChunkChoiceWire{choice}
	}
	frame, err := encodeSSE("", chunk)
	return [][]byte{frame}, nil, err
}

func (encoder *chatStreamEncoder) encodeCompletion(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	encoder.terminal = true
	reason := encodeChatStop(event.StopReason)
	finishFrame, err := encodeSSE("", chatChunkWire{
		ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model,
		Choices: []chatChunkChoiceWire{{Index: 0, Delta: chatChunkDeltaWire{}, FinishReason: &reason}},
	})
	if err != nil {
		return nil, nil, err
	}
	if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
		usageChunk := chatChunkWire{
			ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model,
			Usage: encodeChatUsage(*event.Usage),
		}
		usageFrame, err := encodeSSE("", usageChunk)
		if err != nil {
			return nil, nil, err
		}
		return [][]byte{finishFrame, usageFrame, []byte("data: [DONE]\n\n")}, nil, nil
	}
	return [][]byte{finishFrame, []byte("data: [DONE]\n\n")}, nil, nil
}

func (encoder *chatStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	if reason != nil {
		body := OpenAIChatCodec{}.EncodeError(llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", "stream ended before completion", reason))
		return [][]byte{append([]byte("data: "), append(body, []byte("\n\n")...)...)}, nil, nil
	}
	return [][]byte{[]byte("data: [DONE]\n\n")}, nil, nil
}
