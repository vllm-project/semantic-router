package protocolcodec

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (encoder *responsesStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, err := encoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	event = normalized
	if frames, diagnostics, handled, directErr := encoder.encodeDirectResponsesEvent(event); handled {
		return frames, diagnostics, directErr
	}
	wire, diagnostics, err := encoder.responsesWireForEvent(event)
	if err != nil || wire.Type == "" {
		return nil, diagnostics, err
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return [][]byte{frame}, diagnostics, err
}

func (encoder *responsesStreamEncoder) encodeDirectResponsesEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, bool, error) {
	if event.Type == llmprotocol.EventResponseFailed {
		if event.Failure == llmprotocol.FailureResponse {
			return nil, nil, false, nil
		}
		frames, err := encoder.encodeResponsesTransportFailure(event)
		return frames, nil, true, err
	}
	return encoder.encodeDirectResponsesNonFailureEvent(event)
}

func (encoder *responsesStreamEncoder) encodeDirectResponsesNonFailureEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, bool, error) {
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		frames, err := encoder.encodeResponsesStart(event)
		return frames, nil, true, err
	case llmprotocol.EventOutputItemStarted:
		frames, err := encoder.encodeResponsesItemStart(event)
		return frames, nil, true, err
	case llmprotocol.EventOutputTextDelta:
		frames, diagnostics, err := encoder.encodeResponsesTextDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventReasoningDelta:
		frames, diagnostics, err := encoder.encodeResponsesReasoningDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventToolCallDelta:
		frames, diagnostics, err := encoder.encodeResponsesToolDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventImageGenerationProgress:
		frames, err := encoder.encodeResponsesImageGenerationProgress(event)
		return frames, nil, true, err
	case llmprotocol.EventOutputItemCompleted:
		frames, diagnostics, err := encoder.encodeCompletedResponsesItem(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventUsageUpdated:
		return nil, nil, true, nil
	case llmprotocol.EventProviderOpaque:
		frames, err := encoder.encodeResponsesOpaque(event)
		return frames, nil, true, err
	default:
		return nil, nil, false, nil
	}
}

func (encoder *responsesStreamEncoder) responsesWireForEvent(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	switch event.Type {
	case llmprotocol.EventResponseCompleted:
		return encoder.responsesCompletionWire(event)
	case llmprotocol.EventResponseFailed:
		return encoder.responsesFailureWire(event)
	default:
		return responsesEventWire{}, nil, nil
	}
}

func (encoder *responsesStreamEncoder) encodeResponsesItemStart(event llmprotocol.Event) ([][]byte, error) {
	encoder.neutralItemIDs[event.ItemIndex] = event.ItemID
	if event.ToolCall != nil {
		frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputTool)
		return frames, err
	}
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentGeneratedImage {
		frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputImage)
		return frames, err
	}
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
		key := contentKey(event)
		encoder.encodedKinds[key] = llmprotocol.ContentReasoning
		// An item-start event may identify a reasoning item without identifying
		// whether its later content is summary or encrypted reasoning text. Bind
		// the scope only when the source actually supplies it; the first content
		// event otherwise establishes the scope for its neutral content index.
		if event.Content.Reasoning != "" {
			encoder.reasoningScopes[key] = normalizedReasoningScope(event.Content.Reasoning)
		}
		frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputReasoning)
		return frames, err
	}
	if event.Content != nil {
		encoder.encodedKinds[contentKey(event)] = event.Content.Kind
		frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputMessage)
		return frames, err
	}
	return nil, nil
}

func (encoder *responsesStreamEncoder) ensureResponsesOutputStarted(
	event llmprotocol.Event,
	kind responsesOutputKind,
) ([][]byte, responsesOutputKey, error) {
	key := responsesOutputKey{item: event.ItemIndex, kind: kind}
	if encoder.outputStarted[key] {
		return nil, key, nil
	}
	encoder.outputStarted[key] = true
	encoder.outputIndexes[key] = encoder.nextOutputIndex
	encoder.nextOutputIndex++
	encoder.itemOutputKeys[event.ItemIndex] = append(encoder.itemOutputKeys[event.ItemIndex], key)
	id := encoder.responsesOutputID(event, key)
	encoder.outputIDs[key] = id
	item := responsesItemWire{Type: string(kind), ID: id, Status: "in_progress"}
	switch kind {
	case responsesOutputMessage:
		item.Role = "assistant"
	case responsesOutputTool:
		item.Type = "function_call"
		if event.ToolCall != nil {
			item.CallID, item.Name, item.Arguments = event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
		}
	case responsesOutputImage:
		item.Type = "image_generation_call"
	}
	wire := responsesEventWire{
		Type:        "response.output_item.added",
		Sequence:    encoder.nextWireSequence(),
		OutputIndex: responsesOutputIndex(encoder.outputIndexes[key]),
		Item:        marshalResponsesEventItem(item),
	}
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return nil, key, err
	}
	return [][]byte{frame}, key, nil
}

func (encoder *responsesStreamEncoder) responsesOutputID(event llmprotocol.Event, key responsesOutputKey) string {
	base := event.ItemID
	if base == "" {
		base = encoder.neutralItemIDs[event.ItemIndex]
	}
	if base == "" {
		base = llmprotocol.StableID("responses-output", event.ResponseID, string(key.kind), strconv.Itoa(event.ItemIndex))
	}
	if len(encoder.itemOutputKeys[event.ItemIndex]) == 1 {
		return base
	}
	return llmprotocol.StableID("responses-output-split", base, string(key.kind))
}

func (encoder *responsesStreamEncoder) encodeResponsesStart(event llmprotocol.Event) ([][]byte, error) {
	if encoder.responseStarted {
		return nil, nil
	}
	encoder.responseStarted = true
	response := newResponsesResponseWire(event.ResponseID, event.Model, "in_progress", 0, encoder.context.PreviousResponseID)
	frames := make([][]byte, 0, 2)
	for _, eventType := range []string{"response.created", "response.in_progress"} {
		wire := responsesEventWire{
			Type: eventType, Sequence: encoder.nextWireSequence(), Response: &response,
		}
		frame, err := encoder.encodeResponsesStreamFrame(wire)
		if err != nil {
			return nil, err
		}
		frames = append(frames, frame)
	}
	return frames, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesToolDelta(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.ToolCall == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
	}
	frames, key, err := encoder.ensureResponsesOutputStarted(event, responsesOutputTool)
	if err != nil {
		return nil, nil, err
	}
	wire := responsesEventWire{
		Type:        "response.function_call_arguments.delta",
		Sequence:    encoder.nextWireSequence(),
		ItemID:      encoder.outputIDs[key],
		OutputIndex: responsesOutputIndex(encoder.outputIndexes[key]),
		Delta:       event.ToolCall.Arguments,
	}
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return frames, nil, err
	}
	return append(frames, frame), nil, nil
}

func (encoder *responsesStreamEncoder) responsesCompletionWire(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	if event.Usage == nil {
		return responsesEventWire{}, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
	}
	encoder.terminal = true
	response := newResponsesResponseWire(event.ResponseID, event.Model, "completed", 0, encoder.context.PreviousResponseID)
	output, err := encoder.responsesCompletedOutput()
	if err != nil {
		return responsesEventWire{}, nil, err
	}
	response.Output = output
	response.Usage = encodeResponsesUsage(*event.Usage)
	wire := responsesEventWire{
		Type:     "response.completed",
		Response: &response,
	}
	switch event.StopReason {
	case llmprotocol.StopMaxTokens, llmprotocol.StopContentFilter:
		wire.Type = "response.incomplete"
		wire.Response.Status = "incomplete"
		reason := "max_output_tokens"
		if event.StopReason == llmprotocol.StopContentFilter {
			reason = "content_filter"
		}
		wire.Response.IncompleteDetails = &struct {
			Reason string `json:"reason"`
		}{Reason: reason}
	case llmprotocol.StopPaused, llmprotocol.StopContextWindow, llmprotocol.StopCanceled, llmprotocol.StopUnknown:
		var diagnostics llmprotocol.Diagnostics
		if err := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "response.stop_reason", "Responses cannot represent the source terminal reason"); err != nil {
			return responsesEventWire{}, diagnostics, err
		}
		return wire, diagnostics, nil
	}
	return wire, nil, nil
}

func (encoder *responsesStreamEncoder) responsesFailureWire(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	if event.Error == nil {
		return responsesEventWire{}, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	response := newResponsesResponseWire(event.ResponseID, event.Model, "failed", 0, encoder.context.PreviousResponseID)
	output, err := encoder.responsesCompletedOutput()
	if err != nil {
		return responsesEventWire{}, nil, err
	}
	response.Output = output
	response.Error = &responsesErrorWire{Code: responsesErrorCode(event.Error), Message: event.Error.Message}
	if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
		response.Usage = encodeResponsesUsage(*event.Usage)
	}
	return responsesEventWire{
		Type:     "response.failed",
		Response: &response,
	}, nil, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesTransportFailure(event llmprotocol.Event) ([][]byte, error) {
	if event.Error == nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	frame, err := encoder.encodeTransportError(event.Error)
	return [][]byte{frame}, err
}

func (encoder *responsesStreamEncoder) encodeResponsesReasoningDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	if event.Content != nil && event.Content.Signature != "" {
		if err := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "reasoning.signature", "Responses cannot represent a signed reasoning delta"); err != nil {
			return nil, diagnostics, err
		}
	}
	key := contentKey(event)
	encoder.encodedKinds[key] = llmprotocol.ContentReasoning
	reasoningScope := eventReasoningScope(event)
	if existing := encoder.reasoningScopes[key]; existing != "" && existing != reasoningScope {
		return nil, diagnostics, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_reasoning_scope_mismatch",
			"upstream stream changed a reasoning content scope",
			nil,
		)
	}
	encoder.reasoningScopes[key] = reasoningScope
	builder := encoder.contentText[key]
	if builder == nil {
		builder = &strings.Builder{}
		encoder.contentText[key] = builder
	}
	builder.WriteString(event.Delta)
	frames, outputKey, err := encoder.ensureResponsesOutputStarted(event, responsesOutputReasoning)
	if err != nil {
		return nil, diagnostics, err
	}
	started, err := encoder.startResponsesContent(event, outputKey, llmprotocol.ContentReasoning)
	if err != nil {
		return nil, diagnostics, err
	}
	frames = append(frames, started...)
	if event.Delta == "" {
		return frames, diagnostics, nil
	}
	scope := responsesScopeForReasoning(reasoningScope)
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, scope)
	wire := responsesEventWire{
		Type: responsesReasoningDeltaType(reasoningScope), Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
		Delta: event.Delta,
	}
	setResponsesReasoningIndex(&wire, reasoningScope, contentIndex)
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return frames, diagnostics, err
	}
	return append(frames, frame), diagnostics, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesOpaque(event llmprotocol.Event) ([][]byte, error) {
	if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
	}
	return [][]byte{append([]byte(nil), event.Opaque...)}, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesTextDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	content, err := encoder.recordResponsesTextDelta(event)
	if err != nil {
		return nil, nil, err
	}
	frames, outputKey, err := encoder.ensureResponsesOutputStarted(event, responsesOutputMessage)
	if err != nil {
		return nil, nil, err
	}
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, responsesContentMessage)
	started, err := encoder.startResponsesContent(event, outputKey, content.kind)
	if err != nil {
		return nil, nil, err
	}
	frames = append(frames, started...)
	frames, err = encoder.appendResponsesTextFrame(frames, outputKey, contentIndex, content.kind, event.Delta)
	if err != nil {
		return nil, nil, err
	}
	frames, err = encoder.appendResponsesAnnotations(frames, outputKey, contentIndex, content)
	if err != nil {
		return nil, nil, err
	}
	return frames, nil, nil
}

type responsesRecordedTextDelta struct {
	kind           llmprotocol.ContentKind
	newCitations   []llmprotocol.Citation
	annotationBase int
}

func (encoder *responsesStreamEncoder) recordResponsesTextDelta(
	event llmprotocol.Event,
) (responsesRecordedTextDelta, error) {
	kind := llmprotocol.ContentText
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
		kind = llmprotocol.ContentRefusal
	}
	key := contentKey(event)
	if existing := encoder.encodedKinds[key]; existing != "" && existing != kind {
		return responsesRecordedTextDelta{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable, "stream_content_kind_mismatch", "upstream stream changed a content block kind", nil,
		)
	}
	encoder.encodedKinds[key] = kind
	builder := encoder.contentText[key]
	if builder == nil {
		builder = &strings.Builder{}
		encoder.contentText[key] = builder
	}
	builder.WriteString(event.Delta)
	recorded := responsesRecordedTextDelta{kind: kind, annotationBase: len(encoder.contentCitations[key])}
	if event.Content != nil && len(event.Content.Citations) > 0 {
		recorded.newCitations = event.Content.Citations
		encoder.contentCitations[key] = append(encoder.contentCitations[key], recorded.newCitations...)
	}
	return recorded, nil
}

func (encoder *responsesStreamEncoder) appendResponsesTextFrame(
	frames [][]byte,
	outputKey responsesOutputKey,
	contentIndex int,
	kind llmprotocol.ContentKind,
	delta string,
) ([][]byte, error) {
	if delta == "" {
		return frames, nil
	}
	wire := responsesEventWire{
		Type: responsesTextDeltaType(kind), Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
		ContentIndex: responsesContentIndex(contentIndex), Delta: delta,
	}
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return frames, err
	}
	return append(frames, frame), nil
}

func (encoder *responsesStreamEncoder) appendResponsesAnnotations(
	frames [][]byte,
	outputKey responsesOutputKey,
	contentIndex int,
	content responsesRecordedTextDelta,
) ([][]byte, error) {
	for offset, annotation := range encodeResponsesAnnotations(content.newCitations) {
		index := content.annotationBase + offset
		wire := responsesEventWire{
			Type: "response.output_text.annotation.added", Sequence: encoder.nextWireSequence(),
			ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]), ContentIndex: responsesContentIndex(contentIndex),
			AnnotationIndex: &index, Annotation: &annotation,
		}
		frame, err := encoder.encodeResponsesStreamFrame(wire)
		if err != nil {
			return nil, err
		}
		frames = append(frames, frame)
	}
	return frames, nil
}

func (encoder *responsesStreamEncoder) responsesContentWireIndex(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
	scope responsesContentScope,
) int {
	key := contentKey(event)
	if index, found := encoder.contentIndexes[key]; found {
		return index
	}
	sequence := responsesOutputContentKey{output: outputKey, scope: scope}
	index := encoder.nextContentIndex[sequence]
	encoder.nextContentIndex[sequence] = index + 1
	encoder.contentIndexes[key] = index
	return index
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesItem(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var frames [][]byte
	keys := append([]responsesOutputKey(nil), encoder.itemOutputKeys[event.ItemIndex]...)
	if len(keys) == 0 {
		kind := responsesCompletionOutputKind(event)
		started, key, err := encoder.ensureResponsesOutputStarted(event, kind)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, started...)
		keys = append(keys, key)
	}
	for _, key := range keys {
		completed, diagnostics, err := encoder.encodeCompletedResponsesOutput(event, key)
		if err != nil {
			return nil, diagnostics, err
		}
		frames = append(frames, completed...)
	}
	return frames, nil, nil
}

func responsesCompletionOutputKind(event llmprotocol.Event) responsesOutputKind {
	if event.ToolCall != nil {
		return responsesOutputTool
	}
	if event.Content == nil {
		return responsesOutputMessage
	}
	switch event.Content.Kind {
	case llmprotocol.ContentReasoning:
		return responsesOutputReasoning
	case llmprotocol.ContentGeneratedImage:
		return responsesOutputImage
	default:
		return responsesOutputMessage
	}
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesOutput(
	event llmprotocol.Event,
	key responsesOutputKey,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if key.kind == responsesOutputMessage || key.kind == responsesOutputReasoning {
		return encoder.encodeCompletedResponsesContent(event, key)
	}
	if key.kind == responsesOutputImage {
		return encoder.encodeCompletedResponsesImage(event, key)
	}
	if event.ToolCall == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
	}
	index, id := encoder.outputIndexes[key], encoder.outputIDs[key]
	done := responsesEventWire{
		Type: "response.function_call_arguments.done", Sequence: encoder.nextWireSequence(),
		ItemID: id, OutputIndex: responsesOutputIndex(index),
		Name: event.ToolCall.Name, Arguments: event.ToolCall.Arguments,
	}
	doneFrame, err := encoder.encodeResponsesStreamFrame(done)
	if err != nil {
		return nil, nil, err
	}
	wire := responsesEventWire{
		Type: "response.output_item.done", Sequence: encoder.nextWireSequence(), OutputIndex: responsesOutputIndex(index),
	}
	item := responsesItemWire{
		Type: "function_call", ID: id, Status: "completed",
		CallID: event.ToolCall.ID, Name: event.ToolCall.Name, Arguments: event.ToolCall.Arguments,
	}
	wire.Item = marshalResponsesEventItem(item)
	encoder.recordResponsesCompletedOutput(index, wire.Item)
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return [][]byte{doneFrame, frame}, nil, err
}

func marshalResponsesEventItem(item responsesItemWire) json.RawMessage {
	body, _ := json.Marshal(item)
	return body
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
) ([][]byte, llmprotocol.Diagnostics, error) {
	parts, frames, err := encoder.collectCompletedResponsesContent(event, outputKey)
	if err != nil {
		return nil, nil, err
	}
	index, id := encoder.outputIndexes[outputKey], encoder.outputIDs[outputKey]
	item, err := marshalCompletedResponsesItem(outputKey, id, parts)
	if err != nil {
		return nil, nil, err
	}
	wire := responsesEventWire{
		Type: "response.output_item.done", Sequence: encoder.nextWireSequence(),
		OutputIndex: responsesOutputIndex(index), Item: marshalResponsesEventItem(item),
	}
	encoder.recordResponsesCompletedOutput(index, wire.Item)
	done, err := encoder.encodeResponsesStreamFrame(wire)
	return append(frames, done), nil, err
}
