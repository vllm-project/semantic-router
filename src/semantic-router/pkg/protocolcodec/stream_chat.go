package protocolcodec

import (
	"bytes"
	"encoding/json"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type chatStreamDecoder struct {
	streamState
	framer             sseFramer
	contentIndexes     map[chatContentKey]int
	nextContentIndexes map[int]int
}

type chatContentKey struct {
	item int
	kind llmprotocol.ContentKind
}
type chatStreamEncoder struct {
	streamState
	itemStarted bool
	toolIndexes map[string]int
}

func (OpenAIChatCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &chatStreamDecoder{
		streamState:        streamState{context: context, policy: policy},
		framer:             newSSEFramer(policy.Limits.SSEFrameBytes),
		contentIndexes:     make(map[chatContentKey]int),
		nextContentIndexes: make(map[int]int),
	}
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
	Moderation        json.RawMessage           `json:"moderation,omitempty"`
	Obfuscation       string                    `json:"obfuscation,omitempty"`
	Error             *chatErrorWire            `json:"error,omitempty"`
	ServiceTier       *chatServiceTierWire      `json:"service_tier,omitempty"`
	SystemFingerprint *string                   `json:"system_fingerprint,omitempty"`
	PromptLogprobs    *chatNullOnlyWire         `json:"prompt_logprobs,omitempty"`
	PromptTokenIDs    []int64                   `json:"prompt_token_ids,omitempty"`
	PromptText        *chatNullOnlyWire         `json:"prompt_text,omitempty"`
	KVTransferParams  *chatKVTransferParamsWire `json:"kv_transfer_params,omitempty"`
	ECTransferParams  *chatNullOnlyWire         `json:"ec_transfer_params,omitempty"`
	Metrics           *chatNullOnlyWire         `json:"metrics,omitempty"`
	DoRemoteDecode    *bool                     `json:"do_remote_decode,omitempty"`
	DoRemotePrefill   *bool                     `json:"do_remote_prefill,omitempty"`
	RemoteBlockIDs    []int64                   `json:"remote_block_ids,omitempty"`
	RemoteEngineID    *string                   `json:"remote_engine_id,omitempty"`
	RemoteHost        *string                   `json:"remote_host,omitempty"`
	RemotePort        *int64                    `json:"remote_port,omitempty"`
}

func (wire chatChunkWire) hasLegacyKVTransferMetadata() bool {
	return wire.DoRemoteDecode != nil || wire.DoRemotePrefill != nil || wire.RemoteBlockIDs != nil ||
		wire.RemoteEngineID != nil || wire.RemoteHost != nil || wire.RemotePort != nil
}

func (wire chatChunkWire) hasTokenizedToolArguments() bool {
	for _, choice := range wire.Choices {
		for _, call := range choice.Delta.ToolCalls {
			if call.Function.TokenizedArguments != nil {
				return true
			}
		}
	}
	return false
}

type chatChunkChoiceWire struct {
	Index         int                 `json:"index"`
	Delta         chatChunkDeltaWire  `json:"delta"`
	FinishReason  *string             `json:"finish_reason"`
	Logprobs      *chatLogprobsWire   `json:"logprobs,omitempty"`
	StopReason    *chatStopReasonWire `json:"stop_reason,omitempty"`
	TokenIDs      []int64             `json:"token_ids,omitempty"`
	RoutedExperts *chatNullOnlyWire   `json:"routed_experts,omitempty"`
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
	Index    int                  `json:"index"`
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type,omitempty"`
	Function chatFunctionCallWire `json:"function"`
}

func (decoder *chatStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if err := decoder.observeProviderStreamBytes(chunk); err != nil {
		return nil, nil, err
	}
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
	parsed, err := decoder.parseProviderSSEFrame(frame)
	if err != nil || !parsed.HasData {
		return nil, nil, err
	}
	if decoder.terminal {
		return nil, nil, invalidProviderResponse("stream_event_after_terminal", "Chat stream emitted data after its terminal sentinel")
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
		return []llmprotocol.Event{event}, nil, err
	}
	var chunk chatChunkWire
	if err := decodeProviderWire(parsed.Data, &chunk, decoder.policy); err != nil {
		return nil, nil, err
	}
	if err := validateChatStreamChunk(chunk); err != nil {
		return nil, nil, err
	}
	if err := decoder.observeProviderIdentity(chunk.ID, chunk.Model); err != nil {
		return nil, nil, err
	}
	if chunk.Error != nil {
		event, err := decoder.next(chatStreamFailureEvent(chunk.Error))
		return []llmprotocol.Event{event}, nil, err
	}
	events, diagnostics, err := decoder.decodeChunkEvents(chunk)
	if chunk.hasTokenizedToolArguments() {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.OpenAIChatV1,
			"choices.delta.tool_calls.function.TokenizedArguments",
			"provider tokenization metadata is not model output",
		)
	}
	if chunk.hasLegacyKVTransferMetadata() {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.OpenAIChatV1,
			"stream.kv_transfer", "provider KV-transfer metadata is not model output",
		)
	}
	if len(chunk.Moderation) > 0 && !bytes.Equal(bytes.TrimSpace(chunk.Moderation), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.OpenAIChatV1,
			"stream.moderation", "moderation metadata has no protocol-neutral representation",
		)
	}
	return events, diagnostics, err
}

func validateChatStreamChunk(chunk chatChunkWire) error {
	if chunk.Object != "" && chunk.Object != "chat.completion.chunk" {
		return invalidProviderResponse("invalid_chat_stream_object", "Chat stream object must be chat.completion.chunk")
	}
	if err := validateChatExecutionMetadata(chunk.SystemFingerprint); err != nil {
		return err
	}
	if len(chunk.Choices) > 1 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"stream_multiple_choices",
			"streaming multiple choices is unsupported",
			nil,
		)
	}
	for _, choice := range chunk.Choices {
		if choice.FinishReason != nil && !validChatFinishReason(*choice.FinishReason) {
			return invalidProviderResponse("invalid_chat_finish_reason", "Chat finish reason is not recognized")
		}
		if err := validateChatChunkChoiceExtensions(choice); err != nil {
			return err
		}
		seenToolIndexes := make(map[int]struct{}, len(choice.Delta.ToolCalls))
		for _, call := range choice.Delta.ToolCalls {
			if call.Index < 0 {
				return invalidProviderResponse("invalid_stream_tool_index", "Chat stream tool index must be non-negative")
			}
			if _, duplicate := seenToolIndexes[call.Index]; duplicate {
				return invalidProviderResponse("duplicate_stream_tool_index", "Chat stream tool indexes must be unique within a chunk")
			}
			seenToolIndexes[call.Index] = struct{}{}
		}
	}
	return nil
}

func chatStreamFailureEvent(wire *chatErrorWire) llmprotocol.Event {
	return llmprotocol.Event{
		Type: llmprotocol.EventResponseFailed,
		Error: &llmprotocol.ProtocolError{
			Category: decodeProviderErrorCategory(wire.Type, wire.Code),
			Code:     wire.Code, Message: wire.Message, Parameter: wire.Param,
		},
		StopReason: llmprotocol.StopError,
		Failure:    llmprotocol.FailureTransport,
	}
}

func (decoder *chatStreamDecoder) decodeChunkEvents(chunk chatChunkWire) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
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
	events, err := decoder.decodeChoiceTextEvents(choice)
	if err != nil {
		return nil, err
	}
	toolEvents, err := decoder.decodeToolCalls(choice.Delta.ToolCalls)
	if err != nil {
		return nil, err
	}
	events = append(events, toolEvents...)
	completed, err := decoder.completeChoice(choice.FinishReason)
	return append(events, completed...), err
}

func (decoder *chatStreamDecoder) decodeChoiceTextEvents(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	events := make([]llmprotocol.Event, 0, 5)
	if chatChoiceNeedsItem(choice) && !decoder.items[choice.Index] {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: choice.Index, Role: llmprotocol.RoleAssistant})
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	for _, create := range decoder.chatChoiceEventFactories(choice) {
		event, err := create()
		if err != nil {
			return nil, err
		}
		if event != nil {
			events = append(events, event...)
		}
	}
	return events, nil
}

func chatChoiceNeedsItem(choice chatChunkChoiceWire) bool {
	return choice.Delta.Content != nil || len(choice.Delta.Annotations) > 0 ||
		choice.Delta.Reasoning != nil || choice.Delta.AlternateReasoning != nil || choice.Delta.Refusal != nil
}

type chatEventFactory func() ([]llmprotocol.Event, error)

func (decoder *chatStreamDecoder) chatChoiceEventFactories(choice chatChunkChoiceWire) []chatEventFactory {
	return []chatEventFactory{
		func() ([]llmprotocol.Event, error) { return decoder.decodeContentDelta(choice) },
		func() ([]llmprotocol.Event, error) { return decoder.decodeAnnotations(choice) },
		func() ([]llmprotocol.Event, error) { return decoder.decodeReasoningDelta(choice) },
		func() ([]llmprotocol.Event, error) { return decoder.decodeRefusalDelta(choice) },
	}
}

func (decoder *chatStreamDecoder) decodeContentDelta(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	if choice.Delta.Content == nil {
		return nil, nil
	}
	content := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: *choice.Delta.Content}
	event, err := decoder.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index,
		ContentIndex: decoder.chatContentIndex(choice.Index, llmprotocol.ContentText),
		Delta:        *choice.Delta.Content, Content: &content,
	})
	return []llmprotocol.Event{event}, err
}

func (decoder *chatStreamDecoder) decodeReasoningDelta(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	reasoning := choice.Delta.Reasoning
	if reasoning == nil {
		reasoning = choice.Delta.AlternateReasoning
	}
	if reasoning == nil {
		return nil, nil
	}
	content := llmprotocol.Content{
		Kind: llmprotocol.ContentReasoning, Text: *reasoning, Reasoning: llmprotocol.ReasoningScopeText,
	}
	event, err := decoder.next(llmprotocol.Event{
		Type: llmprotocol.EventReasoningDelta, ItemIndex: choice.Index,
		ContentIndex: decoder.chatContentIndex(choice.Index, llmprotocol.ContentReasoning),
		Delta:        *reasoning, Content: &content,
	})
	return []llmprotocol.Event{event}, err
}

func (decoder *chatStreamDecoder) decodeRefusalDelta(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	if choice.Delta.Refusal == nil {
		return nil, nil
	}
	content := llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: *choice.Delta.Refusal}
	event, err := decoder.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index,
		ContentIndex: decoder.chatContentIndex(choice.Index, llmprotocol.ContentRefusal),
		Delta:        *choice.Delta.Refusal, Content: &content,
	})
	return []llmprotocol.Event{event}, err
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
	event, err := decoder.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index,
		ContentIndex: decoder.chatContentIndex(choice.Index, llmprotocol.ContentText), Content: &content,
	})
	return []llmprotocol.Event{event}, err
}

func (decoder *chatStreamDecoder) chatContentIndex(itemIndex int, kind llmprotocol.ContentKind) int {
	lookup := chatContentKey{item: itemIndex, kind: kind}
	if index, found := decoder.contentIndexes[lookup]; found {
		return index
	}
	index := decoder.nextContentIndexes[itemIndex]
	decoder.nextContentIndexes[itemIndex] = index + 1
	decoder.contentIndexes[lookup] = index
	return index
}

func (decoder *chatStreamDecoder) decodeToolCalls(calls []chatChunkToolCallWire) ([]llmprotocol.Event, error) {
	events := make([]llmprotocol.Event, 0, len(calls)*2)
	for _, call := range calls {
		if call.Type != "" && call.Type != "function" {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorUnsupportedFeature,
				"unsupported_tool_call",
				"only function tool calls enter the model protocol",
				nil,
			)
		}
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
	if len(decoder.items) == 0 {
		if decoder.stop == llmprotocol.StopToolCall {
			return nil, invalidProviderResponse("stream_tool_output_missing", "Chat stream ended with tool_calls but emitted no tool call")
		}
		started, err := decoder.next(llmprotocol.Event{
			Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0,
			Role: llmprotocol.RoleAssistant, Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
		})
		if err != nil {
			return nil, err
		}
		completed, err := decoder.next(llmprotocol.Event{
			Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0, StopReason: decoder.stop,
			Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
		})
		return []llmprotocol.Event{started, completed}, err
	}
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
	if directChatStreamEvent(event.Type) {
		return encoder.encodeDirectEvent(event)
	}
	if event.Type == llmprotocol.EventOutputItemCompleted {
		return nil, nil, nil
	}
	chunk := chatChunkWire{ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model}
	choice := chatChunkChoiceWire{Index: 0}
	diagnostics, emitChunk, err := encoder.applyChunkEvent(event, &chunk, &choice)
	if err != nil || !emitChunk {
		return nil, diagnostics, err
	}
	if event.Type != llmprotocol.EventUsageUpdated {
		chunk.Choices = []chatChunkChoiceWire{choice}
	}
	frame, err := encodeChatStreamFrame(encoder.context, chunk)
	return [][]byte{frame}, diagnostics, err
}

func directChatStreamEvent(eventType llmprotocol.EventType) bool {
	return eventType == llmprotocol.EventResponseCompleted ||
		eventType == llmprotocol.EventResponseFailed || eventType == llmprotocol.EventProviderOpaque
}

func (encoder *chatStreamEncoder) applyChunkEvent(
	event llmprotocol.Event,
	chunk *chatChunkWire,
	choice *chatChunkChoiceWire,
) (llmprotocol.Diagnostics, bool, error) {
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		encoder.startChatItem(choice)
	case llmprotocol.EventOutputItemStarted:
		return nil, encoder.startChatItem(choice), nil
	case llmprotocol.EventOutputTextDelta:
		applyChatTextDelta(event, choice)
	case llmprotocol.EventReasoningDelta:
		return encoder.applyChatReasoningDelta(event, choice)
	case llmprotocol.EventToolCallDelta:
		if err := encoder.applyToolCallDelta(event, choice); err != nil {
			return nil, false, err
		}
	case llmprotocol.EventUsageUpdated:
		if event.Usage == nil {
			return nil, false, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "usage event is invalid", nil)
		}
		// Usage evidence is accumulated by streamState and rendered exactly once
		// from the terminal event when the public client requested it.
		return nil, false, nil
	default:
		return nil, false, nil
	}
	return nil, true, nil
}

func (encoder *chatStreamEncoder) startChatItem(choice *chatChunkChoiceWire) bool {
	if encoder.itemStarted {
		return false
	}
	choice.Delta.Role = "assistant"
	encoder.itemStarted = true
	return true
}

func applyChatTextDelta(event llmprotocol.Event, choice *chatChunkChoiceWire) {
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
		choice.Delta.Refusal = &event.Delta
	} else {
		choice.Delta.Content = &event.Delta
	}
	if event.Content != nil && len(event.Content.Citations) > 0 {
		choice.Delta.Annotations = encodeChatAnnotations(event.Content.Citations)
	}
}

func (encoder *chatStreamEncoder) applyChatReasoningDelta(
	event llmprotocol.Event,
	choice *chatChunkChoiceWire,
) (llmprotocol.Diagnostics, bool, error) {
	diagnostics, err := encoder.reasoningDiagnostics(event)
	if err != nil {
		return diagnostics, false, err
	}
	choice.Delta.Reasoning = &event.Delta
	return diagnostics, true, nil
}

func (encoder *chatStreamEncoder) reasoningDiagnostics(event llmprotocol.Event) (llmprotocol.Diagnostics, error) {
	if event.Content == nil || event.Content.Signature == "" {
		return nil, nil
	}
	var diagnostics llmprotocol.Diagnostics
	err := appendLossy(
		&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target,
		"reasoning.signature", "Chat Completions cannot represent a signed reasoning delta",
	)
	return diagnostics, err
}

func (encoder *chatStreamEncoder) applyToolCallDelta(event llmprotocol.Event, choice *chatChunkChoiceWire) error {
	if event.ToolCall == nil {
		return llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
	}
	index, found := encoder.toolIndexes[event.ToolCall.ID]
	if !found {
		index = len(encoder.toolIndexes)
		encoder.toolIndexes[event.ToolCall.ID] = index
	}
	choice.Delta.ToolCalls = []chatChunkToolCallWire{{
		Index: index, ID: event.ToolCall.ID, Type: "function",
		Function: chatFunctionCallWire{Name: event.ToolCall.Name, Arguments: event.ToolCall.Arguments},
	}}
	return nil
}

func (encoder *chatStreamEncoder) encodeDirectEvent(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	switch event.Type {
	case llmprotocol.EventResponseCompleted:
		return encoder.encodeCompletion(event)
	case llmprotocol.EventResponseFailed:
		encoder.terminal = true
		if event.Error == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
		}
		var diagnostics llmprotocol.Diagnostics
		if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
			appendAccountingOmission(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "usage", "Chat Completions error events cannot carry token usage")
		}
		frame, err := encodeSSE("", openAITransportErrorEnvelope(event.Error))
		return [][]byte{frame}, diagnostics, err
	case llmprotocol.EventProviderOpaque:
		if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
		}
		return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
	default:
		return nil, nil, nil
	}
}

func (encoder *chatStreamEncoder) encodeCompletion(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	encoder.terminal = true
	var diagnostics llmprotocol.Diagnostics
	if event.StopReason == llmprotocol.StopPaused || event.StopReason == llmprotocol.StopContextWindow || event.StopReason == llmprotocol.StopCanceled || event.StopReason == llmprotocol.StopUnknown {
		if err := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "response.stop_reason", "Chat Completions cannot represent the source terminal reason"); err != nil {
			return nil, diagnostics, err
		}
	}
	reason := encodeChatStop(event.StopReason)
	finishFrame, err := encodeChatStreamFrame(encoder.context, chatChunkWire{
		ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model,
		Choices: []chatChunkChoiceWire{{Index: 0, Delta: chatChunkDeltaWire{}, FinishReason: &reason}},
	})
	if err != nil {
		return nil, nil, err
	}
	if streamUsageRequested(encoder.context) && event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
		usageChunk := chatChunkWire{
			ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model,
			Usage: encodeChatUsage(*event.Usage),
		}
		usageFrame, err := encodeChatStreamFrame(encoder.context, usageChunk)
		if err != nil {
			return nil, nil, err
		}
		return [][]byte{finishFrame, usageFrame, []byte("data: [DONE]\n\n")}, diagnostics, nil
	}
	return [][]byte{finishFrame, []byte("data: [DONE]\n\n")}, diagnostics, nil
}

func encodeChatStreamFrame(context llmprotocol.StreamContext, chunk chatChunkWire) ([]byte, error) {
	obfuscation, err := newStreamObfuscation(context)
	if err != nil {
		return nil, err
	}
	chunk.Obfuscation = obfuscation
	return encodeSSE("", chunk)
}

func (encoder *chatStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	if reason != nil {
		body := OpenAIChatCodec{}.EncodeTransportError(llmprotocol.TransportError{
			Error: streamFinalizationError(reason, "stream ended before completion"),
		})
		return [][]byte{append([]byte("data: "), append(body, []byte("\n\n")...)...)}, nil, nil
	}
	return [][]byte{[]byte("data: [DONE]\n\n")}, nil, nil
}
