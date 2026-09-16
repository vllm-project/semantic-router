package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (decoder *responsesStreamDecoder) decodeResponsesEvent(
	wire responsesEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	var providerDiagnostics llmprotocol.Diagnostics
	if len(wire.Logprobs) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Logprobs), []byte("null")) {
		appendProviderFieldOmission(
			&providerDiagnostics,
			decoder.policy,
			llmprotocol.OpenAIResponsesV1,
			"stream.logprobs",
			"token log probabilities have no protocol-neutral representation",
		)
	}
	outputIndex := responsesWireOutputIndex(wire)
	event := llmprotocol.Event{
		ResponseID: decoder.context.ResponseID,
		Model:      decoder.context.PublicModel,
		ItemIndex:  outputIndex,
		ItemID:     wire.ItemID,
		Delta:      wire.Delta,
	}
	if key, found := responsesEventWireContentKey(wire); found {
		event.ContentIndex = decoder.neutralContentIndex(key)
	}
	if handled, events, diagnostics, err := decoder.decodeResponsesContentEvent(&event, wire); handled {
		return events, appendDiagnostics(providerDiagnostics, diagnostics, decoder.policy.Limits.Diagnostics), err
	}
	events, diagnostics, err := decoder.decodeResponsesLifecycleEvent(event, wire, frame)
	return events, appendDiagnostics(providerDiagnostics, diagnostics, decoder.policy.Limits.Diagnostics), err
}

func (decoder *responsesStreamDecoder) decodeResponsesContentEvent(
	event *llmprotocol.Event,
	wire responsesEventWire,
) (bool, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	handled, err := decoder.applyResponsesContentEvent(event, wire)
	if !handled {
		return false, nil, nil, nil
	}
	if err != nil {
		return true, nil, nil, err
	}
	events, diagnostics, err := decoder.emitResponsesEvent(*event)
	return true, events, diagnostics, err
}

func (decoder *responsesStreamDecoder) applyResponsesContentEvent(
	event *llmprotocol.Event,
	wire responsesEventWire,
) (bool, error) {
	if handled, err := decoder.applyResponsesItemEvent(event, wire); handled {
		return true, err
	}
	if handled, err := applyResponsesPartEvent(event, wire); handled {
		return true, err
	}
	return decoder.applyResponsesDeltaEvent(event, wire)
}

func (decoder *responsesStreamDecoder) applyResponsesItemEvent(
	event *llmprotocol.Event,
	wire responsesEventWire,
) (bool, error) {
	switch wire.Type {
	case "response.output_item.added":
		return true, decoder.applyResponsesItemStart(event, wire)
	case "response.output_item.done":
		return true, decoder.applyCompletedResponseItem(event, wire)
	default:
		return false, nil
	}
}

func applyResponsesPartEvent(event *llmprotocol.Event, wire responsesEventWire) (bool, error) {
	switch wire.Type {
	case "response.content_part.added":
		return true, applyResponsesContentPart(event, wire.Part)
	case "response.reasoning_summary_part.added":
		return true, applyResponsesReasoningPart(event, wire.Part)
	default:
		return false, nil
	}
}

func (decoder *responsesStreamDecoder) applyResponsesDeltaEvent(
	event *llmprotocol.Event,
	wire responsesEventWire,
) (bool, error) {
	switch wire.Type {
	case "response.output_text.delta":
		setResponsesTextDelta(event, wire.Delta, llmprotocol.ContentText)
	case "response.refusal.delta":
		setResponsesTextDelta(event, wire.Delta, llmprotocol.ContentRefusal)
	case "response.reasoning_text.delta":
		setResponsesReasoningDelta(event, wire.Delta, llmprotocol.ReasoningScopeText)
	case "response.reasoning_summary_text.delta":
		setResponsesReasoningDelta(event, wire.Delta, llmprotocol.ReasoningScopeSummary)
	case "response.output_text.annotation.added":
		return true, decoder.applyResponseAnnotation(event, wire)
	case "response.function_call_arguments.delta":
		return true, decoder.applyResponsesToolDelta(event, wire)
	case "response.image_generation_call.in_progress", "response.image_generation_call.generating",
		"response.image_generation_call.partial_image", "response.image_generation_call.completed":
		return true, decoder.applyResponsesImageGenerationProgress(event, wire)
	default:
		return false, nil
	}
	return true, nil
}

func setResponsesTextDelta(event *llmprotocol.Event, delta string, kind llmprotocol.ContentKind) {
	event.Type = llmprotocol.EventOutputTextDelta
	event.Content = &llmprotocol.Content{Kind: kind, Text: delta}
}

func setResponsesReasoningDelta(event *llmprotocol.Event, delta string, scope llmprotocol.ReasoningScope) {
	event.Type = llmprotocol.EventReasoningDelta
	event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: delta, Reasoning: scope}
}

func applyResponsesContentPart(event *llmprotocol.Event, part *responsesContentWire) error {
	if part == nil {
		return invalidProviderResponse("stream_content_part_required", "Responses content part event is incomplete")
	}
	event.Type = llmprotocol.EventOutputTextDelta
	switch part.Type {
	case "output_text":
		event.Delta = part.Text
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text}
	case "refusal":
		event.Delta = part.Refusal
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: part.Refusal}
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_stream_content_part", "Responses stream content part is unsupported", nil)
	}
	return nil
}

func applyResponsesReasoningPart(event *llmprotocol.Event, part *responsesContentWire) error {
	if part == nil {
		return invalidProviderResponse("stream_reasoning_part_required", "Responses reasoning summary part event is incomplete")
	}
	if part.Type != "summary_text" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_stream_reasoning_part", "Responses stream reasoning part is unsupported", nil)
	}
	event.Type = llmprotocol.EventReasoningDelta
	event.Delta = part.Text
	event.Content = &llmprotocol.Content{
		Kind: llmprotocol.ContentReasoning, Text: part.Text, Reasoning: llmprotocol.ReasoningScopeSummary,
	}
	return nil
}

func responsesEventWireContentKey(wire responsesEventWire) (responsesWireContentKey, bool) {
	key := responsesWireContentKey{item: responsesWireOutputIndex(wire)}
	switch wire.Type {
	case "response.content_part.added", "response.content_part.done",
		"response.output_text.delta", "response.output_text.done", "response.output_text.annotation.added",
		"response.refusal.delta", "response.refusal.done":
		if wire.ContentIndex == nil {
			return responsesWireContentKey{}, false
		}
		key.scope, key.index = responsesContentMessage, *wire.ContentIndex
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
		"response.reasoning_summary_text.delta", "response.reasoning_summary_text.done":
		if wire.SummaryIndex == nil {
			return responsesWireContentKey{}, false
		}
		key.scope, key.index = responsesContentReasoningSummary, *wire.SummaryIndex
	case "response.reasoning_text.delta", "response.reasoning_text.done":
		if wire.ContentIndex == nil {
			return responsesWireContentKey{}, false
		}
		key.scope, key.index = responsesContentReasoningText, *wire.ContentIndex
	default:
		return responsesWireContentKey{}, false
	}
	return key, true
}

func (decoder *responsesStreamDecoder) neutralContentIndex(key responsesWireContentKey) int {
	if index, found := decoder.contentIndexes[key]; found {
		return index
	}
	index := decoder.nextContentIndex[key.item]
	decoder.nextContentIndex[key.item] = index + 1
	decoder.contentIndexes[key] = index
	return index
}

func (decoder *responsesStreamDecoder) validateResponsesContentLifecycle(wire responsesEventWire) error {
	key, hasContent := responsesEventWireContentKey(wire)
	if handled, err := decoder.validateResponsesContentStart(wire, key); handled {
		return err
	}
	if handled, err := decoder.validateResponsesContentDelta(wire, key); handled {
		return err
	}
	if handled, err := decoder.validateResponsesContentDone(wire, key); handled {
		return err
	}
	if wire.Type == "response.output_item.done" {
		return decoder.validateCompletedResponsesItemLifecycle(wire)
	}
	if hasContent {
		return invalidProviderResponse("unsupported_stream_content_event", "Responses content event is unsupported")
	}
	return nil
}

func (decoder *responsesStreamDecoder) validateResponsesContentStart(
	wire responsesEventWire,
	key responsesWireContentKey,
) (bool, error) {
	switch wire.Type {
	case "response.content_part.added":
		kind, text, err := responsesMessagePartEvidence(wire.Part)
		if err != nil {
			return true, err
		}
		return true, decoder.startResponsesContent(key, kind, text)
	case "response.reasoning_summary_part.added":
		return true, decoder.startResponsesReasoningSummary(key, wire.Part)
	default:
		return false, nil
	}
}

func (decoder *responsesStreamDecoder) validateResponsesContentDelta(
	wire responsesEventWire,
	key responsesWireContentKey,
) (bool, error) {
	switch wire.Type {
	case "response.output_text.delta":
		return true, decoder.appendResponsesContent(key, "output_text", wire.Delta, false)
	case "response.refusal.delta":
		return true, decoder.appendResponsesContent(key, "refusal", wire.Delta, false)
	case "response.reasoning_summary_text.delta":
		return true, decoder.appendResponsesContent(key, "summary_text", wire.Delta, false)
	case "response.reasoning_text.delta":
		return true, decoder.appendResponsesContent(key, "reasoning_text", wire.Delta, true)
	case "response.output_text.annotation.added":
		return true, decoder.appendResponsesAnnotation(key, wire.Annotation)
	default:
		return false, nil
	}
}

func (decoder *responsesStreamDecoder) validateResponsesContentDone(
	wire responsesEventWire,
	key responsesWireContentKey,
) (bool, error) {
	switch wire.Type {
	case "response.output_text.done":
		return true, decoder.finishResponsesContentText(key, "output_text", wire.Text, false)
	case "response.refusal.done":
		return true, decoder.finishResponsesContentText(key, "refusal", wire.Refusal, false)
	case "response.reasoning_summary_text.done":
		return true, decoder.finishResponsesContentText(key, "summary_text", wire.Text, false)
	case "response.reasoning_text.done":
		return true, decoder.finishResponsesContentText(key, "reasoning_text", wire.Text, true)
	case "response.content_part.done":
		kind, text, err := responsesMessagePartEvidence(wire.Part)
		if err != nil {
			return true, err
		}
		return true, decoder.finishResponsesContentPart(key, kind, text, wire.Part)
	case "response.reasoning_summary_part.done":
		return true, decoder.finishResponsesReasoningSummary(key, wire.Part)
	default:
		return false, nil
	}
}

func (decoder *responsesStreamDecoder) startResponsesReasoningSummary(
	key responsesWireContentKey,
	part *responsesContentWire,
) error {
	if part == nil || part.Type != "summary_text" {
		return invalidProviderResponse("stream_reasoning_part_kind", "Responses reasoning summary part must use summary_text")
	}
	return decoder.startResponsesContent(key, "summary_text", part.Text)
}

func (decoder *responsesStreamDecoder) appendResponsesAnnotation(
	key responsesWireContentKey,
	annotation *responsesAnnotationWire,
) error {
	state, err := decoder.requireResponsesContent(key, "output_text")
	if err != nil {
		return err
	}
	if state.textDone || state.partDone {
		return invalidProviderResponse("stream_content_after_done", "Responses annotation was emitted after content completion")
	}
	state.annotations = append(state.annotations, *annotation)
	return nil
}

func (decoder *responsesStreamDecoder) finishResponsesReasoningSummary(
	key responsesWireContentKey,
	part *responsesContentWire,
) error {
	if part == nil || part.Type != "summary_text" {
		return invalidProviderResponse("stream_reasoning_part_kind", "Responses reasoning summary part must use summary_text")
	}
	return decoder.finishResponsesContentPart(key, "summary_text", part.Text, part)
}

func responsesMessagePartEvidence(part *responsesContentWire) (string, string, error) {
	if part == nil {
		return "", "", invalidProviderResponse("stream_content_part_required", "Responses content part event is incomplete")
	}
	switch part.Type {
	case "output_text":
		return part.Type, part.Text, nil
	case "refusal":
		return part.Type, part.Refusal, nil
	default:
		return "", "", llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_stream_content_part",
			"Responses stream content part is unsupported",
			nil,
		)
	}
}

func (decoder *responsesStreamDecoder) startResponsesContent(
	key responsesWireContentKey,
	kind string,
	initial string,
) error {
	if _, exists := decoder.contentLifecycle[key]; exists {
		return invalidProviderResponse("duplicate_stream_content_start", "Responses content part started more than once")
	}
	scopeKey := responsesWireContentScopeKey{item: key.item, scope: key.scope}
	if key.index != decoder.nextWireContentIndex[scopeKey] {
		return invalidProviderResponse(
			"stream_content_index_order",
			"Responses content indexes must be unique and contiguous within their content namespace",
		)
	}
	decoder.nextWireContentIndex[scopeKey]++
	state := &responsesDecodedContentLifecycle{kind: kind}
	state.text.WriteString(initial)
	decoder.contentLifecycle[key] = state
	return nil
}

func (decoder *responsesStreamDecoder) requireResponsesContent(
	key responsesWireContentKey,
	kind string,
) (*responsesDecodedContentLifecycle, error) {
	state := decoder.contentLifecycle[key]
	if state == nil {
		return nil, invalidProviderResponse("stream_content_start_missing", "Responses content event was emitted before its content start")
	}
	if state.kind != kind {
		return nil, invalidProviderResponse("stream_content_kind_mismatch", "Responses content part changed its kind")
	}
	return state, nil
}

func (decoder *responsesStreamDecoder) appendResponsesContent(
	key responsesWireContentKey,
	kind string,
	delta string,
	allowImplicitStart bool,
) error {
	if decoder.contentLifecycle[key] == nil && allowImplicitStart {
		if err := decoder.startResponsesContent(key, kind, ""); err != nil {
			return err
		}
	}
	state, err := decoder.requireResponsesContent(key, kind)
	if err != nil {
		return err
	}
	if state.textDone || state.partDone {
		return invalidProviderResponse("stream_content_after_done", "Responses content delta was emitted after content completion")
	}
	state.text.WriteString(delta)
	return nil
}

func (decoder *responsesStreamDecoder) finishResponsesContentText(
	key responsesWireContentKey,
	kind string,
	text string,
	allowImplicitStart bool,
) error {
	if decoder.contentLifecycle[key] == nil && allowImplicitStart {
		if err := decoder.startResponsesContent(key, kind, ""); err != nil {
			return err
		}
	}
	state, err := decoder.requireResponsesContent(key, kind)
	if err != nil {
		return err
	}
	if state.textDone || state.partDone {
		return invalidProviderResponse("duplicate_stream_content_done", "Responses content text completed more than once")
	}
	if state.text.String() != text {
		return invalidProviderResponse("stream_content_text_mismatch", "Responses content done text does not match streamed deltas")
	}
	state.textDone = true
	return nil
}

func (decoder *responsesStreamDecoder) finishResponsesContentPart(
	key responsesWireContentKey,
	kind string,
	text string,
	part *responsesContentWire,
) error {
	state, err := decoder.requireResponsesContent(key, kind)
	if err != nil {
		return err
	}
	if !state.textDone {
		return invalidProviderResponse("stream_content_text_incomplete", "Responses content part completed before its text done event")
	}
	if state.partDone {
		return invalidProviderResponse("duplicate_stream_content_part_done", "Responses content part completed more than once")
	}
	if state.text.String() != text {
		return invalidProviderResponse("stream_content_part_mismatch", "Responses completed content part does not match streamed text")
	}
	var annotations []responsesAnnotationWire
	if part != nil && part.Annotations != nil {
		annotations = *part.Annotations
	}
	if !reflect.DeepEqual(state.annotations, annotations) && !(len(state.annotations) == 0 && len(annotations) == 0) {
		return invalidProviderResponse("stream_content_annotations_mismatch", "Responses completed content part does not match streamed annotations")
	}
	state.partDone = true
	return nil
}

func (decoder *responsesStreamDecoder) validateCompletedResponsesItemLifecycle(wire responsesEventWire) error {
	item, err := decodeResponsesItemWire(wire.Item, decoder.policy, true)
	if err != nil {
		return err
	}
	index := responsesWireOutputIndex(wire)
	switch item.Type {
	case "message":
		return decoder.validateResponsesFinalContent(index, responsesContentMessage, item.Content, true)
	case "reasoning":
		if err := decoder.validateResponsesFinalContent(index, responsesContentReasoningSummary, item.Summary, true); err != nil {
			return err
		}
		return decoder.validateResponsesFinalContent(index, responsesContentReasoningText, item.Content, false)
	case "function_call":
		if !decoder.toolArgumentsDone[index] {
			return invalidProviderResponse(
				"stream_tool_arguments_incomplete",
				"Responses function-call item completed before its arguments done event",
			)
		}
		return nil
	default:
		return nil
	}
}

func (decoder *responsesStreamDecoder) validateResponsesFinalContent(
	itemIndex int,
	scope responsesContentScope,
	raw json.RawMessage,
	requirePartDone bool,
) error {
	var parts []responsesContentWire
	if len(raw) > 0 && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if err := decodeProviderValue(raw, &parts, decoder.policy); err != nil {
			return err
		}
	}
	expectedCount := decoder.nextWireContentIndex[responsesWireContentScopeKey{item: itemIndex, scope: scope}]
	if len(parts) != expectedCount {
		return invalidProviderResponse("stream_item_content_mismatch", "Responses completed item content does not match streamed content blocks")
	}
	for index, part := range parts {
		if err := decoder.validateResponsesFinalPart(itemIndex, scope, index, part, requirePartDone); err != nil {
			return err
		}
	}
	return nil
}

func (decoder *responsesStreamDecoder) validateResponsesFinalPart(
	itemIndex int,
	scope responsesContentScope,
	index int,
	part responsesContentWire,
	requirePartDone bool,
) error {
	state := decoder.contentLifecycle[responsesWireContentKey{item: itemIndex, scope: scope, index: index}]
	if state == nil || !state.textDone || requirePartDone && !state.partDone {
		return invalidProviderResponse("stream_item_content_incomplete", "Responses output item completed with unfinished content")
	}
	kind, text, err := responsesFinalPartEvidence(scope, part)
	if err != nil {
		return err
	}
	if state.kind != kind || state.text.String() != text {
		return invalidProviderResponse("stream_item_content_mismatch", "Responses completed item content does not match streamed content")
	}
	annotations := responsesPartAnnotations(part)
	if !reflect.DeepEqual(state.annotations, annotations) && !(len(state.annotations) == 0 && len(annotations) == 0) {
		return invalidProviderResponse("stream_item_annotations_mismatch", "Responses completed item annotations do not match streamed annotations")
	}
	return nil
}

func responsesPartAnnotations(part responsesContentWire) []responsesAnnotationWire {
	if part.Annotations == nil {
		return nil
	}
	return *part.Annotations
}

func responsesFinalPartEvidence(scope responsesContentScope, part responsesContentWire) (string, string, error) {
	switch scope {
	case responsesContentMessage:
		return responsesMessagePartEvidence(&part)
	case responsesContentReasoningSummary:
		if part.Type == "summary_text" {
			return part.Type, part.Text, nil
		}
	case responsesContentReasoningText:
		if part.Type == "reasoning_text" {
			return part.Type, part.Text, nil
		}
	}
	return "", "", invalidProviderResponse("stream_item_content_kind", "Responses completed item contains the wrong content kind")
}

func (decoder *responsesStreamDecoder) validateResponsesTerminalOutput(raw json.RawMessage) error {
	var output []json.RawMessage
	if err := decodeProviderValue(raw, &output, decoder.policy); err != nil {
		return err
	}
	if len(output) != len(decoder.completedOutput) {
		return invalidProviderResponse("stream_terminal_output_mismatch", "Responses terminal output does not contain every completed output item")
	}
	for index, item := range output {
		completed, found := decoder.completedOutput[index]
		if !found || !providerJSONEquivalent(completed, item) {
			return invalidProviderResponse("stream_terminal_output_mismatch", "Responses terminal output does not match completed output items")
		}
	}
	return nil
}

func providerJSONEquivalent(left, right []byte) bool {
	var leftValue any
	leftDecoder := json.NewDecoder(bytes.NewReader(left))
	leftDecoder.UseNumber()
	if err := leftDecoder.Decode(&leftValue); err != nil {
		return false
	}
	var rightValue any
	rightDecoder := json.NewDecoder(bytes.NewReader(right))
	rightDecoder.UseNumber()
	return rightDecoder.Decode(&rightValue) == nil && reflect.DeepEqual(leftValue, rightValue)
}
