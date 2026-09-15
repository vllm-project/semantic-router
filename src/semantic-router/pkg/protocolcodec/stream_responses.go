package protocolcodec

import (
	"bytes"
	"encoding/json"
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
	responsesOutputImage     responsesOutputKind = "image_generation"
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
	outputIndexes          map[responsesOutputKey]int
	outputIDs              map[responsesOutputKey]string
	outputStarted          map[responsesOutputKey]bool
	itemOutputKeys         map[int][]responsesOutputKey
	neutralItemIDs         map[int]string
	nextOutputIndex        int
	contentIndexes         map[streamContentKey]int
	nextContentIndex       map[responsesOutputContentKey]int
	contentStarted         map[streamContentKey]bool
	encodedKinds           map[streamContentKey]llmprotocol.ContentKind
	reasoningScopes        map[streamContentKey]llmprotocol.ReasoningScope
	contentText            map[streamContentKey]*strings.Builder
	contentCitations       map[streamContentKey][]llmprotocol.Citation
	completedOutput        map[int]json.RawMessage
	imageProgressCompleted map[responsesOutputKey]bool
	responseStarted        bool
	wireSequence           uint64
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
		streamState:            streamState{context: context, policy: policy},
		outputIndexes:          make(map[responsesOutputKey]int),
		outputIDs:              make(map[responsesOutputKey]string),
		outputStarted:          make(map[responsesOutputKey]bool),
		itemOutputKeys:         make(map[int][]responsesOutputKey),
		neutralItemIDs:         make(map[int]string),
		contentIndexes:         make(map[streamContentKey]int),
		nextContentIndex:       make(map[responsesOutputContentKey]int),
		contentStarted:         make(map[streamContentKey]bool),
		encodedKinds:           make(map[streamContentKey]llmprotocol.ContentKind),
		reasoningScopes:        make(map[streamContentKey]llmprotocol.ReasoningScope),
		contentText:            make(map[streamContentKey]*strings.Builder),
		contentCitations:       make(map[streamContentKey][]llmprotocol.Citation),
		completedOutput:        make(map[int]json.RawMessage),
		imageProgressCompleted: make(map[responsesOutputKey]bool),
	}
}

type responsesEventWire struct {
	Type              string                   `json:"type"`
	Sequence          uint64                   `json:"sequence_number"`
	Response          *responsesResponseWire   `json:"response,omitempty"`
	Item              json.RawMessage          `json:"item,omitempty"`
	ItemID            string                   `json:"item_id,omitempty"`
	OutputIndex       *int                     `json:"output_index,omitempty"`
	ContentIndex      *int                     `json:"content_index,omitempty"`
	AnnotationIndex   *int                     `json:"annotation_index,omitempty"`
	Delta             string                   `json:"delta,omitempty"`
	Text              string                   `json:"text,omitempty"`
	Part              *responsesContentWire    `json:"part,omitempty"`
	Annotation        *responsesAnnotationWire `json:"annotation,omitempty"`
	Name              string                   `json:"name,omitempty"`
	Arguments         string                   `json:"arguments,omitempty"`
	Refusal           string                   `json:"refusal,omitempty"`
	Status            string                   `json:"status,omitempty"`
	SummaryIndex      *int                     `json:"summary_index,omitempty"`
	Logprobs          json.RawMessage          `json:"logprobs,omitempty"`
	Obfuscation       string                   `json:"obfuscation,omitempty"`
	Code              *string                  `json:"code,omitempty"`
	Message           string                   `json:"message,omitempty"`
	Param             *string                  `json:"param,omitempty"`
	PartialImageIndex *int64                   `json:"partial_image_index,omitempty"`
	PartialImageB64   string                   `json:"partial_image_b64,omitempty"`
	Size              string                   `json:"size,omitempty"`
	Quality           string                   `json:"quality,omitempty"`
	Background        string                   `json:"background,omitempty"`
	OutputFormat      string                   `json:"output_format,omitempty"`
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
	case "response.image_generation_call.partial_image":
		object["partial_image_b64"], _ = json.Marshal(wire.PartialImageB64)
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
	case "response.image_generation_call.in_progress", "response.image_generation_call.generating",
		"response.image_generation_call.partial_image", "response.image_generation_call.completed":
		expected = "image_generation_call"
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
		"response.function_call_arguments.delta", "response.function_call_arguments.done",
		"response.image_generation_call.in_progress", "response.image_generation_call.generating",
		"response.image_generation_call.partial_image", "response.image_generation_call.completed":
		return true
	default:
		return false
	}
}
