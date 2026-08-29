package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type responsesStreamDecoder struct {
	streamState
	framer                sseFramer
	nextAnnotationIndexes map[streamContentKey]int
	contentIndexes        map[responsesWireContentKey]int
	nextContentIndex      map[int]int
	contentLifecycle      map[responsesWireContentKey]*responsesDecodedContentLifecycle
	nextWireContentIndex  map[responsesWireContentScopeKey]int
	itemTypes             map[int]string
	toolArgumentsDone     map[int]bool
	completedOutput       map[int]json.RawMessage
	seenLifecycleEvents   map[string]bool
	wireSequence          uint64
	wireSequenceSeen      bool
}

type responsesOutputKind string

const (
	responsesOutputMessage   responsesOutputKind = "message"
	responsesOutputReasoning responsesOutputKind = "reasoning"
	responsesOutputTool      responsesOutputKind = "tool"
)

type responsesOutputKey struct {
	item int
	kind responsesOutputKind
}

type responsesContentScope string

const (
	responsesContentMessage          responsesContentScope = "message"
	responsesContentReasoningSummary responsesContentScope = "reasoning_summary"
	responsesContentReasoningText    responsesContentScope = "reasoning_text"
)

type responsesWireContentKey struct {
	item  int
	scope responsesContentScope
	index int
}

type responsesWireContentScopeKey struct {
	item  int
	scope responsesContentScope
}

type responsesOutputContentKey struct {
	output responsesOutputKey
	scope  responsesContentScope
}

type responsesDecodedContentLifecycle struct {
	kind        string
	text        strings.Builder
	annotations []responsesAnnotationWire
	textDone    bool
	partDone    bool
}

type responsesStreamEncoder struct {
	streamState
	outputIndexes    map[responsesOutputKey]int
	outputIDs        map[responsesOutputKey]string
	outputStarted    map[responsesOutputKey]bool
	itemOutputKeys   map[int][]responsesOutputKey
	neutralItemIDs   map[int]string
	nextOutputIndex  int
	contentIndexes   map[streamContentKey]int
	nextContentIndex map[responsesOutputContentKey]int
	contentStarted   map[streamContentKey]bool
	encodedKinds     map[streamContentKey]llmprotocol.ContentKind
	reasoningScopes  map[streamContentKey]llmprotocol.ReasoningScope
	contentText      map[streamContentKey]*strings.Builder
	contentCitations map[streamContentKey][]llmprotocol.Citation
	completedOutput  map[int]json.RawMessage
	responseStarted  bool
	wireSequence     uint64
}

func (OpenAIResponsesCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &responsesStreamDecoder{
		streamState:           streamState{context: context, policy: policy},
		framer:                newSSEFramer(policy.Limits.SSEFrameBytes),
		nextAnnotationIndexes: make(map[streamContentKey]int),
		contentIndexes:        make(map[responsesWireContentKey]int),
		nextContentIndex:      make(map[int]int),
		contentLifecycle:      make(map[responsesWireContentKey]*responsesDecodedContentLifecycle),
		nextWireContentIndex:  make(map[responsesWireContentScopeKey]int),
		itemTypes:             make(map[int]string),
		toolArgumentsDone:     make(map[int]bool),
		completedOutput:       make(map[int]json.RawMessage),
		seenLifecycleEvents:   make(map[string]bool),
	}
}

func (OpenAIResponsesCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &responsesStreamEncoder{
		streamState:      streamState{context: context, policy: policy},
		outputIndexes:    make(map[responsesOutputKey]int),
		outputIDs:        make(map[responsesOutputKey]string),
		outputStarted:    make(map[responsesOutputKey]bool),
		itemOutputKeys:   make(map[int][]responsesOutputKey),
		neutralItemIDs:   make(map[int]string),
		contentIndexes:   make(map[streamContentKey]int),
		nextContentIndex: make(map[responsesOutputContentKey]int),
		contentStarted:   make(map[streamContentKey]bool),
		encodedKinds:     make(map[streamContentKey]llmprotocol.ContentKind),
		reasoningScopes:  make(map[streamContentKey]llmprotocol.ReasoningScope),
		contentText:      make(map[streamContentKey]*strings.Builder),
		contentCitations: make(map[streamContentKey][]llmprotocol.Citation),
		completedOutput:  make(map[int]json.RawMessage),
	}
}

type responsesEventWire struct {
	Type            string                   `json:"type"`
	Sequence        uint64                   `json:"sequence_number"`
	Response        *responsesResponseWire   `json:"response,omitempty"`
	Item            json.RawMessage          `json:"item,omitempty"`
	ItemID          string                   `json:"item_id,omitempty"`
	OutputIndex     *int                     `json:"output_index,omitempty"`
	ContentIndex    *int                     `json:"content_index,omitempty"`
	AnnotationIndex *int                     `json:"annotation_index,omitempty"`
	Delta           string                   `json:"delta,omitempty"`
	Text            string                   `json:"text,omitempty"`
	Part            *responsesContentWire    `json:"part,omitempty"`
	Annotation      *responsesAnnotationWire `json:"annotation,omitempty"`
	Name            string                   `json:"name,omitempty"`
	Arguments       string                   `json:"arguments,omitempty"`
	Refusal         string                   `json:"refusal,omitempty"`
	Status          string                   `json:"status,omitempty"`
	SummaryIndex    *int                     `json:"summary_index,omitempty"`
	Logprobs        json.RawMessage          `json:"logprobs,omitempty"`
	Obfuscation     string                   `json:"obfuscation,omitempty"`
	Code            *string                  `json:"code,omitempty"`
	Message         string                   `json:"message,omitempty"`
	Param           *string                  `json:"param,omitempty"`
}

func (wire responsesEventWire) MarshalJSON() ([]byte, error) {
	type eventAlias responsesEventWire
	body, err := json.Marshal(eventAlias(wire))
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, err
	}
	switch wire.Type {
	case "response.output_text.delta", "response.refusal.delta",
		"response.reasoning_text.delta", "response.reasoning_summary_text.delta",
		"response.function_call_arguments.delta":
		object["delta"], _ = json.Marshal(wire.Delta)
	case "response.output_text.done", "response.reasoning_text.done", "response.reasoning_summary_text.done":
		object["text"], _ = json.Marshal(wire.Text)
	case "response.refusal.done":
		object["refusal"], _ = json.Marshal(wire.Refusal)
	case "response.function_call_arguments.done":
		object["name"], _ = json.Marshal(wire.Name)
		object["arguments"], _ = json.Marshal(wire.Arguments)
	}
	return json.Marshal(object)
}

type responsesTransportErrorEventWire struct {
	Type     string  `json:"type"`
	Code     *string `json:"code"`
	Message  string  `json:"message"`
	Param    *string `json:"param"`
	Sequence uint64  `json:"sequence_number"`
}

func (decoder *responsesStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
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

func (decoder *responsesStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := decoder.parseProviderSSEFrame(frame)
	if err != nil || !parsed.HasData {
		return nil, nil, err
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		return nil, nil, invalidProviderResponse(
			"invalid_responses_stream_sentinel",
			"Responses streams terminate with a response terminal event, not a [DONE] sentinel",
		)
	}
	if decoder.terminal {
		return nil, nil, invalidProviderResponse("stream_event_after_terminal", "Responses stream emitted data after its terminal event")
	}
	eventType, err := decoder.validateResponsesFrameEnvelope(parsed.Data, parsed.Event)
	if err != nil {
		return nil, nil, err
	}
	if !isSupportedResponsesEvent(eventType) {
		return decoder.decodeUnknownResponsesFrame(frame)
	}
	return decoder.decodeResponsesWireFrame(parsed.Data, eventType, frame)
}

func (decoder *responsesStreamDecoder) validateResponsesFrameEnvelope(data []byte, eventName string) (string, error) {
	eventType, err := decodeProviderEventType(data, eventName, decoder.policy)
	if err != nil {
		return "", err
	}
	sequence, err := decodeResponsesStreamSequence(data)
	if err != nil {
		return "", err
	}
	if err := decoder.validateWireSequence(sequence); err != nil {
		return "", err
	}
	return eventType, nil
}

func (decoder *responsesStreamDecoder) decodeUnknownResponsesFrame(
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	event := llmprotocol.Event{
		ResponseID: decoder.context.ResponseID,
		Model:      decoder.context.PublicModel,
	}
	if err := decoder.applyUnknownResponsesEvent(&event, frame); err != nil {
		return nil, nil, err
	}
	return decoder.emitResponsesEvent(event)
}

func (decoder *responsesStreamDecoder) decodeResponsesWireFrame(
	data []byte,
	eventType string,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	var wire responsesEventWire
	if err := decodeProviderWire(data, &wire, decoder.policy); err != nil {
		return nil, nil, err
	}
	if wire.Type == "" {
		wire.Type = eventType
	}
	if err := decoder.validateResponsesEventResource(wire, data); err != nil {
		return nil, nil, err
	}
	return decoder.decodeResponsesEvent(wire, frame)
}

func (decoder *responsesStreamDecoder) validateResponsesEventResource(wire responsesEventWire, body []byte) error {
	if err := validateResponsesEventRequiredFields(wire, body); err != nil {
		return err
	}
	if err := decoder.validateResponsesEventItemIdentity(wire); err != nil {
		return err
	}
	if err := decoder.validateResponsesEventItemType(wire); err != nil {
		return err
	}
	if err := decoder.validateResponsesContentLifecycle(wire); err != nil {
		return err
	}
	if err := decoder.recordResponsesLifecycleEvent(wire); err != nil {
		return err
	}
	if wire.Response == nil {
		return nil
	}
	return decoder.validateResponsesLifecycleResource(wire)
}

func (decoder *responsesStreamDecoder) recordResponsesLifecycleEvent(wire responsesEventWire) error {
	if wire.Type == "response.output_item.added" && wire.OutputIndex != nil && *wire.OutputIndex != len(decoder.items) {
		return invalidProviderResponse("stream_output_index_order", "Responses output indexes must be contiguous from zero")
	}
	if !isResponsesLifecycleEvent(wire.Type) {
		return nil
	}
	if decoder.seenLifecycleEvents[wire.Type] {
		return invalidProviderResponse("duplicate_stream_lifecycle", "Responses lifecycle event was emitted more than once")
	}
	decoder.seenLifecycleEvents[wire.Type] = true
	return nil
}

func (decoder *responsesStreamDecoder) validateResponsesLifecycleResource(wire responsesEventWire) error {
	if err := validateResponsesResponseResource(*wire.Response, true); err != nil {
		return err
	}
	if err := decoder.observeProviderIdentity(wire.Response.ID, wire.Response.Model); err != nil {
		return err
	}
	expectedStatus := map[string]string{
		"response.created":     "in_progress",
		"response.queued":      "queued",
		"response.in_progress": "in_progress",
		"response.completed":   "completed",
		"response.incomplete":  "incomplete",
		"response.failed":      "failed",
	}[wire.Type]
	if expectedStatus != "" && wire.Response.Status != "" && wire.Response.Status != expectedStatus {
		return invalidProviderResponse("stream_response_status_mismatch", "Responses event type does not match response status")
	}
	if wire.Type == "response.completed" || wire.Type == "response.incomplete" {
		return decoder.validateResponsesTerminalOutput(wire.Response.Output)
	}
	return nil
}

func (decoder *responsesStreamDecoder) validateResponsesEventItemType(wire responsesEventWire) error {
	if wire.Type == "response.output_item.added" || wire.OutputIndex == nil {
		return nil
	}
	expected := ""
	switch wire.Type {
	case "response.content_part.added", "response.content_part.done",
		"response.output_text.delta", "response.output_text.done", "response.output_text.annotation.added",
		"response.refusal.delta", "response.refusal.done":
		expected = "message"
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
		"response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
		"response.reasoning_text.delta", "response.reasoning_text.done":
		expected = "reasoning"
	case "response.function_call_arguments.delta", "response.function_call_arguments.done":
		expected = "function_call"
	}
	if expected != "" && decoder.itemTypes[*wire.OutputIndex] != expected {
		return invalidProviderResponse(
			"stream_item_kind_mismatch",
			"Responses event type does not match its active output item",
		)
	}
	return nil
}

func isResponsesLifecycleEvent(eventType string) bool {
	switch eventType {
	case "response.created", "response.queued", "response.in_progress",
		"response.completed", "response.incomplete", "response.failed":
		return true
	default:
		return false
	}
}

func (decoder *responsesStreamDecoder) validateResponsesEventItemIdentity(wire responsesEventWire) error {
	if wire.Type == "response.output_item.added" || wire.OutputIndex == nil {
		return nil
	}
	index := *wire.OutputIndex
	if wire.Type == "response.output_item.done" {
		var item struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(wire.Item, &item); err != nil {
			return invalidProviderResponse("invalid_upstream_json", "Responses output item is invalid")
		}
		return decoder.requireResponsesActiveItem(index, item.ID)
	}
	if strings.TrimSpace(wire.ItemID) == "" {
		return nil
	}
	return decoder.requireResponsesActiveItem(index, wire.ItemID)
}

func (decoder *responsesStreamDecoder) requireResponsesActiveItem(index int, itemID string) error {
	if !decoder.items[index] || decoder.completedItems[index] {
		return invalidProviderResponse("invalid_item_lifecycle", "Responses event does not reference an active output item")
	}
	if expected := decoder.itemIDs[index]; expected == "" || itemID == "" || expected != itemID {
		return invalidProviderResponse("stream_item_id_mismatch", "Responses item_id does not match output_index")
	}
	return nil
}

func decodeResponsesStreamSequence(body []byte) (uint64, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return 0, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response JSON is invalid", err)
	}
	raw, present := object["sequence_number"]
	if !present {
		return 0, invalidProviderResponse(
			"missing_stream_sequence",
			"Responses stream event is missing sequence_number",
		)
	}
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return 0, invalidProviderResponse(
			"invalid_stream_sequence",
			"Responses stream sequence_number must be a non-negative integer",
		)
	}
	var sequence uint64
	if err := json.Unmarshal(raw, &sequence); err != nil {
		return 0, invalidProviderResponse(
			"invalid_stream_sequence",
			"Responses stream sequence_number must be a non-negative integer",
		)
	}
	return sequence, nil
}

func (decoder *responsesStreamDecoder) validateWireSequence(sequence uint64) error {
	if !decoder.wireSequenceSeen && sequence != 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_sequence_start",
			"Responses stream sequence numbers must start at zero",
			nil,
		)
	}
	if decoder.wireSequenceSeen && (decoder.wireSequence == ^uint64(0) || sequence != decoder.wireSequence+1) {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_sequence_order",
			"Responses stream sequence numbers must be contiguous and increasing",
			nil,
		)
	}
	decoder.wireSequence = sequence
	decoder.wireSequenceSeen = true
	return nil
}

func isSupportedResponsesEvent(eventType string) bool {
	switch eventType {
	case "response.created", "response.queued", "response.in_progress",
		"response.completed", "response.incomplete", "response.failed", "error",
		"response.output_item.added", "response.output_item.done",
		"response.content_part.added", "response.content_part.done",
		"response.output_text.delta", "response.output_text.done", "response.output_text.annotation.added",
		"response.refusal.delta", "response.refusal.done",
		"response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
		"response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
		"response.reasoning_text.delta", "response.reasoning_text.done",
		"response.function_call_arguments.delta", "response.function_call_arguments.done":
		return true
	default:
		return false
	}
}

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

func (decoder *responsesStreamDecoder) decodeResponsesLifecycleEvent(
	event llmprotocol.Event,
	wire responsesEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	switch wire.Type {
	case "response.created", "response.queued", "response.in_progress":
		if decoder.started {
			return nil, nil, nil
		}
		applyResponsesStart(&event, wire)
	case "response.completed", "response.incomplete":
		applyResponsesCompletion(&event, wire)
	case "response.failed":
		applyResponseFailure(&event, wire)
	case "error":
		applyResponsesTransportFailure(&event, wire)
	case "response.content_part.added", "response.content_part.done",
		"response.output_text.done", "response.refusal.done",
		"response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
		"response.reasoning_summary_text.done", "response.reasoning_text.done":
		return nil, nil, nil
	case "response.function_call_arguments.done":
		if err := decoder.validateResponsesToolDone(wire); err != nil {
			return nil, nil, err
		}
		return nil, nil, nil
	default:
		if err := decoder.applyUnknownResponsesEvent(&event, frame); err != nil {
			return nil, nil, err
		}
	}
	return decoder.emitResponsesEvent(event)
}

func (decoder *responsesStreamDecoder) validateResponsesToolDone(wire responsesEventWire) error {
	index := responsesWireOutputIndex(wire)
	call := decoder.toolCalls[index]
	if call.Name != "" && wire.Name != call.Name {
		return invalidProviderResponse("stream_tool_identity_mismatch", "Responses function-call done event changed the tool name")
	}
	if call.Name == "" {
		call.Name = wire.Name
		decoder.toolCalls[index] = call
	}
	arguments := decoder.toolArguments[index]
	if len(arguments) == 0 {
		if !isJSONObject([]byte(wire.Arguments), decoder.policy.Limits.JSONDepth) {
			return invalidProviderResponse("invalid_stream_tool_arguments", "Responses function-call done arguments must be a JSON object")
		}
		decoder.toolArguments[index] = []byte(wire.Arguments)
		decoder.toolArgumentsDone[index] = true
		return nil
	}
	if string(arguments) != wire.Arguments {
		return invalidProviderResponse("stream_tool_arguments_mismatch", "Responses function-call done arguments do not match streamed arguments")
	}
	decoder.toolArgumentsDone[index] = true
	return nil
}

func applyResponsesStart(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseStarted
	if wire.Response != nil {
		event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
	}
}

func (decoder *responsesStreamDecoder) applyResponsesItemStart(event *llmprotocol.Event, wire responsesEventWire) error {
	event.Type = llmprotocol.EventOutputItemStarted
	if len(wire.Item) == 0 {
		return nil
	}
	item, err := decodeResponsesItemWire(wire.Item, decoder.policy, true)
	if err != nil {
		return err
	}
	if err := validateResponsesOutputItemResource(wire.Item, item); err != nil {
		return err
	}
	if item.Status != "" && item.Status != "in_progress" {
		return invalidProviderResponse(
			"stream_item_status_mismatch",
			"Responses output item added event requires in_progress status",
		)
	}
	event.ItemID = item.ID
	decoder.itemTypes[responsesWireOutputIndex(wire)] = item.Type
	event.Role = llmprotocol.RoleAssistant
	if item.Type == "function_call" {
		event.ToolCall = &llmprotocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments}
	} else if item.Type == "reasoning" {
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}
	}
	return nil
}

func (decoder *responsesStreamDecoder) applyResponsesToolDelta(event *llmprotocol.Event, wire responsesEventWire) error {
	event.Type = llmprotocol.EventToolCallDelta
	call := decoder.toolCalls[responsesWireOutputIndex(wire)]
	if len(wire.Item) > 0 {
		item, err := decodeResponsesItemWire(wire.Item, decoder.policy, true)
		if err != nil {
			return err
		}
		if item.CallID != "" {
			call.ID = item.CallID
		}
	}
	if wire.Name != "" {
		call.Name = wire.Name
	}
	call.Arguments = wire.Delta
	event.ToolCall = &call
	return nil
}

func applyResponsesCompletion(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseCompleted
	event.StopReason = llmprotocol.StopEndTurn
	if wire.Type == "response.incomplete" {
		event.StopReason = llmprotocol.StopUnknown
		if wire.Response != nil && wire.Response.IncompleteDetails != nil {
			switch wire.Response.IncompleteDetails.Reason {
			case "max_output_tokens":
				event.StopReason = llmprotocol.StopMaxTokens
			case "content_filter":
				event.StopReason = llmprotocol.StopContentFilter
			}
		}
	}
	if wire.Response != nil && wire.Response.Usage != nil {
		usage := decodeResponsesUsage(*wire.Response.Usage)
		event.Usage = &usage
	}
}

func (decoder *responsesStreamDecoder) applyUnknownResponsesEvent(event *llmprotocol.Event, frame []byte) error {
	if decoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || decoder.context.Source != decoder.context.Target {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Responses stream event is unsupported", nil)
	}
	event.Type, event.Opaque = llmprotocol.EventProviderOpaque, append([]byte(nil), frame...)
	return nil
}

func (decoder *responsesStreamDecoder) emitResponsesEvent(
	event llmprotocol.Event,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	normalized, err := decoder.next(event)
	return []llmprotocol.Event{normalized}, nil, err
}

func (decoder *responsesStreamDecoder) applyResponseAnnotation(
	event *llmprotocol.Event, wire responsesEventWire,
) error {
	if wire.Annotation == nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_missing", "Responses citation event is missing its annotation", nil)
	}
	outputIndex := responsesWireOutputIndex(wire)
	itemID, itemFound := decoder.itemIDs[outputIndex]
	if !itemFound || wire.ItemID == "" || wire.ItemID != itemID {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_item", "Responses citation event does not match its active output item", nil)
	}
	if wire.ContentIndex == nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_content_index", "Responses citation event is missing its content index", nil)
	}
	key := streamContentKey{item: outputIndex, content: *wire.ContentIndex}
	expected := decoder.nextAnnotationIndexes[key]
	if wire.AnnotationIndex == nil || *wire.AnnotationIndex != expected {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_index", "Responses citation indexes must be monotonic and contiguous", nil)
	}
	citations, err := decodeResponsesAnnotations([]responsesAnnotationWire{*wire.Annotation})
	if err != nil {
		return err
	}
	event.Type = llmprotocol.EventOutputTextDelta
	event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Citations: citations}
	decoder.nextAnnotationIndexes[key] = expected + 1
	return nil
}

func (decoder *responsesStreamDecoder) applyCompletedResponseItem(
	event *llmprotocol.Event, wire responsesEventWire,
) error {
	event.Type = llmprotocol.EventOutputItemCompleted
	if len(wire.Item) == 0 {
		return nil
	}
	item, err := decoder.validateCompletedResponseItem(wire)
	if err != nil {
		return err
	}
	event.ItemID = item.ID
	if err := decoder.applyCompletedResponseItemKind(event, wire, item); err != nil {
		return err
	}
	decoder.completedOutput[responsesWireOutputIndex(wire)] = append(json.RawMessage(nil), wire.Item...)
	return nil
}

func (decoder *responsesStreamDecoder) validateCompletedResponseItem(wire responsesEventWire) (responsesItemWire, error) {
	item, err := decodeResponsesItemWire(wire.Item, decoder.policy, true)
	if err != nil {
		return responsesItemWire{}, err
	}
	if err := validateResponsesOutputItemResource(wire.Item, item); err != nil {
		return responsesItemWire{}, err
	}
	if item.Status != "" && item.Status != "completed" && item.Status != "incomplete" {
		return responsesItemWire{}, invalidProviderResponse(
			"stream_item_status_mismatch", "Responses output item done event requires completed or incomplete status",
		)
	}
	if expected := decoder.itemTypes[responsesWireOutputIndex(wire)]; expected == "" || item.Type != expected {
		return responsesItemWire{}, invalidProviderResponse(
			"stream_item_kind_mismatch", "Responses output item completion changed its item type",
		)
	}
	return item, nil
}

func (decoder *responsesStreamDecoder) applyCompletedResponseItemKind(
	event *llmprotocol.Event,
	wire responsesEventWire,
	item responsesItemWire,
) error {
	switch item.Type {
	case "function_call":
		event.ToolCall = &llmprotocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments}
	case "message":
		if decoder.itemKinds[responsesWireOutputIndex(wire)] == llmprotocol.ContentToolCall {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_kind_mismatch", "upstream completed a tool item as a message", nil)
		}
	case "reasoning":
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_item", "Responses completed an unsupported output item", nil)
	}
	return nil
}

func applyResponseFailure(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseFailed
	event.StopReason = llmprotocol.StopError
	event.Failure = llmprotocol.FailureResponse
	event.Error = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
	var upstreamError *responsesErrorWire
	if wire.Response != nil {
		event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
		upstreamError = wire.Response.Error
		if wire.Response.Usage != nil {
			usage := decodeResponsesUsage(*wire.Response.Usage)
			event.Usage = &usage
		}
	}
	if upstreamError != nil {
		event.Error.Category = decodeProviderErrorCategory(upstreamError.Code)
		event.Error.Code, event.Error.Message = upstreamError.Code, upstreamError.Message
	}
}

func applyResponsesTransportFailure(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseFailed
	event.StopReason = llmprotocol.StopError
	event.Failure = llmprotocol.FailureTransport
	event.Error = llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil,
	)
	if wire.Code != nil {
		event.Error.Code = *wire.Code
		event.Error.Category = decodeProviderErrorCategory(*wire.Code)
	}
	if wire.Message != "" {
		event.Error.Message = wire.Message
	}
	if wire.Param != nil {
		event.Error.Parameter = *wire.Param
	}
}

func (decoder *responsesStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

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
	case llmprotocol.EventOutputItemCompleted:
		frames, diagnostics, err := encoder.encodeCompletedResponsesItem(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventUsageUpdated:
		return nil, nil, true, nil
	case llmprotocol.EventProviderOpaque:
		frames, err := encoder.encodeResponsesOpaque(event)
		return frames, nil, true, err
	case llmprotocol.EventResponseFailed:
		if event.Failure == llmprotocol.FailureResponse {
			return nil, nil, false, nil
		}
		frames, err := encoder.encodeResponsesTransportFailure(event)
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
		kind := responsesOutputMessage
		if event.ToolCall != nil {
			kind = responsesOutputTool
		} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
			kind = responsesOutputReasoning
		}
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

func (encoder *responsesStreamEncoder) encodeCompletedResponsesOutput(
	event llmprotocol.Event,
	key responsesOutputKey,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if key.kind == responsesOutputMessage || key.kind == responsesOutputReasoning {
		return encoder.encodeCompletedResponsesContent(event, key)
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

type completedResponsesContent struct {
	message          []responsesContentWire
	reasoningSummary []responsesContentWire
	reasoningText    []responsesContentWire
}

func (encoder *responsesStreamEncoder) collectCompletedResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
) (completedResponsesContent, [][]byte, error) {
	parts := completedResponsesContent{}
	var frames [][]byte
	for _, key := range encoder.responsesContentKeys(event, outputKey) {
		kind := encoder.encodedKinds[key]
		if kind == "" {
			kind = llmprotocol.ContentText
		}
		contentEvent := event
		contentEvent.ContentIndex = key.content
		completed, err := encoder.completeResponsesContent(contentEvent, outputKey, kind)
		if err != nil {
			return completedResponsesContent{}, nil, err
		}
		frames = append(frames, completed...)
		parts.append(kind, encoder.reasoningScopes[key], responsesContentPart(
			kind, encoder.reasoningScopes[key], encoder.responsesContentText(key), encoder.contentCitations[key],
		))
	}
	return parts, frames, nil
}

func (encoder *responsesStreamEncoder) responsesContentText(key streamContentKey) string {
	if builder := encoder.contentText[key]; builder != nil {
		return builder.String()
	}
	return ""
}

func (parts *completedResponsesContent) append(
	kind llmprotocol.ContentKind,
	reasoningScope llmprotocol.ReasoningScope,
	part responsesContentWire,
) {
	if kind != llmprotocol.ContentReasoning {
		parts.message = append(parts.message, part)
		return
	}
	if normalizedReasoningScope(reasoningScope) == llmprotocol.ReasoningScopeSummary {
		parts.reasoningSummary = append(parts.reasoningSummary, part)
		return
	}
	parts.reasoningText = append(parts.reasoningText, part)
}

func marshalCompletedResponsesItem(
	outputKey responsesOutputKey,
	id string,
	parts completedResponsesContent,
) (responsesItemWire, error) {
	item := responsesItemWire{Type: "message", ID: id, Role: "assistant", Status: "completed"}
	if outputKey.kind != responsesOutputReasoning {
		content, err := json.Marshal(parts.message)
		item.Content = content
		return item, err
	}
	item.Type, item.Role = "reasoning", ""
	var err error
	if len(parts.reasoningSummary) > 0 {
		item.Summary, err = json.Marshal(parts.reasoningSummary)
		if err != nil {
			return responsesItemWire{}, err
		}
	}
	if len(parts.reasoningText) > 0 {
		item.Content, err = json.Marshal(parts.reasoningText)
	}
	return item, err
}

func (encoder *responsesStreamEncoder) responsesContentKeys(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
) []streamContentKey {
	keys := make([]streamContentKey, 0)
	for key, kind := range encoder.encodedKinds {
		if key.item == event.ItemIndex && responsesOutputKindForContent(kind) == outputKey.kind {
			keys = append(keys, key)
		}
	}
	if len(keys) == 0 {
		key := contentKey(event)
		kind := llmprotocol.ContentText
		if event.Content != nil && event.Content.Kind != "" {
			kind = event.Content.Kind
		}
		encoder.encodedKinds[key] = kind
		keys = append(keys, key)
	}
	sort.Slice(keys, func(left, right int) bool {
		return keys[left].content < keys[right].content
	})
	return keys
}

func responsesOutputKindForContent(kind llmprotocol.ContentKind) responsesOutputKind {
	if kind == llmprotocol.ContentReasoning {
		return responsesOutputReasoning
	}
	return responsesOutputMessage
}

func (encoder *responsesStreamEncoder) recordResponsesCompletedOutput(index int, item json.RawMessage) {
	encoder.completedOutput[index] = append(json.RawMessage(nil), item...)
}

func (encoder *responsesStreamEncoder) responsesCompletedOutput() (json.RawMessage, error) {
	indexes := make([]int, 0, len(encoder.completedOutput))
	for index := range encoder.completedOutput {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	items := make([]json.RawMessage, 0, len(indexes))
	for _, index := range indexes {
		items = append(items, encoder.completedOutput[index])
	}
	body, err := json.Marshal(items)
	if err != nil {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorInternal,
			"responses_stream_output",
			"completed Responses stream output could not be encoded",
			err,
		)
	}
	return body, nil
}

func (encoder *responsesStreamEncoder) nextWireSequence() uint64 {
	sequence := encoder.wireSequence
	encoder.wireSequence++
	return sequence
}

func responsesContentIndex(value int) *int {
	return &value
}

func responsesOutputIndex(value int) *int {
	return &value
}

func responsesWireOutputIndex(wire responsesEventWire) int {
	if wire.OutputIndex == nil {
		return 0
	}
	return *wire.OutputIndex
}

func responsesContentPart(
	kind llmprotocol.ContentKind,
	reasoning llmprotocol.ReasoningScope,
	text string,
	citations []llmprotocol.Citation,
) responsesContentWire {
	if kind == llmprotocol.ContentRefusal {
		return responsesContentWire{Type: "refusal", Refusal: text}
	}
	if kind == llmprotocol.ContentReasoning {
		if normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeSummary {
			return responsesContentWire{Type: "summary_text", Text: text}
		}
		return responsesContentWire{Type: "reasoning_text", Text: text}
	}
	return responsesContentWire{Type: "output_text", Text: text, Annotations: responsesAnnotations(citations)}
}

func responsesTextDeltaType(kind llmprotocol.ContentKind) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.delta"
	}
	return "response.output_text.delta"
}

func eventReasoningScope(event llmprotocol.Event) llmprotocol.ReasoningScope {
	if event.Content == nil {
		return llmprotocol.ReasoningScopeText
	}
	return normalizedReasoningScope(event.Content.Reasoning)
}

func normalizedReasoningScope(scope llmprotocol.ReasoningScope) llmprotocol.ReasoningScope {
	if scope == llmprotocol.ReasoningScopeSummary {
		return scope
	}
	return llmprotocol.ReasoningScopeText
}

func responsesScopeForReasoning(scope llmprotocol.ReasoningScope) responsesContentScope {
	if normalizedReasoningScope(scope) == llmprotocol.ReasoningScopeSummary {
		return responsesContentReasoningSummary
	}
	return responsesContentReasoningText
}

func responsesContentScopeFor(
	kind llmprotocol.ContentKind,
	reasoning llmprotocol.ReasoningScope,
) responsesContentScope {
	if kind == llmprotocol.ContentReasoning {
		return responsesScopeForReasoning(reasoning)
	}
	return responsesContentMessage
}

func responsesReasoningDeltaType(scope llmprotocol.ReasoningScope) string {
	if normalizedReasoningScope(scope) == llmprotocol.ReasoningScopeSummary {
		return "response.reasoning_summary_text.delta"
	}
	return "response.reasoning_text.delta"
}

func setResponsesReasoningIndex(
	wire *responsesEventWire,
	scope llmprotocol.ReasoningScope,
	index int,
) {
	if normalizedReasoningScope(scope) == llmprotocol.ReasoningScopeSummary {
		wire.SummaryIndex = responsesContentIndex(index)
		return
	}
	wire.ContentIndex = responsesContentIndex(index)
}

func responsesTextDoneType(kind llmprotocol.ContentKind, reasoning llmprotocol.ReasoningScope) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.done"
	}
	if kind == llmprotocol.ContentReasoning {
		if normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeSummary {
			return "response.reasoning_summary_text.done"
		}
		return "response.reasoning_text.done"
	}
	return "response.output_text.done"
}

func (encoder *responsesStreamEncoder) startResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	key := contentKey(event)
	if encoder.contentStarted[key] {
		return nil, nil
	}
	encoder.contentStarted[key] = true
	reasoning := encoder.reasoningScopes[key]
	scope := responsesContentScopeFor(kind, reasoning)
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, scope)
	if kind == llmprotocol.ContentReasoning {
		if normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeText {
			return nil, nil
		}
	}
	wire := responsesEventWire{
		Type: "response.content_part.added", Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
	}
	if kind == llmprotocol.ContentReasoning {
		wire.Type = "response.reasoning_summary_part.added"
		wire.SummaryIndex = responsesContentIndex(contentIndex)
	} else {
		wire.ContentIndex = responsesContentIndex(contentIndex)
	}
	part := responsesContentPart(kind, reasoning, "", nil)
	wire.Part = &part
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return nil, err
	}
	return [][]byte{frame}, nil
}

func (encoder *responsesStreamEncoder) completeResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	frames, err := encoder.startResponsesContent(event, outputKey, kind)
	if err != nil {
		return nil, err
	}
	text := ""
	key := contentKey(event)
	reasoning := encoder.reasoningScopes[key]
	if builder := encoder.contentText[key]; builder != nil {
		text = builder.String()
	}
	scope := responsesContentScopeFor(kind, reasoning)
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, scope)
	done := responsesEventWire{
		Type: responsesTextDoneType(kind, reasoning), Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
	}
	if kind == llmprotocol.ContentReasoning {
		setResponsesReasoningIndex(&done, reasoning, contentIndex)
		done.Text = text
	} else {
		done.ContentIndex = responsesContentIndex(contentIndex)
		if kind == llmprotocol.ContentRefusal {
			done.Refusal = text
		} else {
			done.Text = text
		}
	}
	doneFrame, err := encoder.encodeResponsesStreamFrame(done)
	if err != nil {
		return nil, err
	}
	if kind == llmprotocol.ContentReasoning && normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeText {
		return append(frames, doneFrame), nil
	}
	part := responsesContentPart(kind, reasoning, text, encoder.contentCitations[key])
	partDone := responsesEventWire{
		Type: "response.content_part.done", Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
		Part: &part,
	}
	if kind == llmprotocol.ContentReasoning {
		partDone.Type = "response.reasoning_summary_part.done"
		partDone.SummaryIndex = responsesContentIndex(contentIndex)
	} else {
		partDone.ContentIndex = responsesContentIndex(contentIndex)
	}
	partFrame, err := encoder.encodeResponsesStreamFrame(partDone)
	if err != nil {
		return nil, err
	}
	return append(frames, doneFrame, partFrame), nil
}

func (encoder *responsesStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	protocolError := streamFinalizationError(reason, "stream ended before completion")
	frame, err := encoder.encodeTransportError(protocolError)
	return [][]byte{frame}, nil, err
}

func (encoder *responsesStreamEncoder) encodeTransportError(protocolError *llmprotocol.ProtocolError) ([]byte, error) {
	wire := responsesTransportErrorEventWire{
		Type: "error", Code: optionalString(protocolError.Code), Message: protocolError.Message,
		Param: optionalString(protocolError.Parameter), Sequence: encoder.nextWireSequence(),
	}
	return encodeSSE(wire.Type, wire)
}

func (encoder *responsesStreamEncoder) encodeResponsesStreamFrame(wire responsesEventWire) ([]byte, error) {
	obfuscation, err := newStreamObfuscation(encoder.context)
	if err != nil {
		return nil, err
	}
	wire.Obfuscation = obfuscation
	return encodeSSE(wire.Type, wire)
}
