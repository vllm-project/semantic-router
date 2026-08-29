package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type sseFrame struct {
	Event   string
	Data    []byte
	HasData bool
}

// streamWireIndexes maps protocol-neutral item identities to the compact,
// contiguous indexes required by provider stream contracts. A source format
// may reserve neutral indexes; those implementation details must never leak
// onto a target wire.
type streamWireIndexes struct {
	items map[int]int
	next  int
}

type streamContentKey struct {
	item    int
	content int
}

func contentKey(event llmprotocol.Event) streamContentKey {
	return streamContentKey{item: event.ItemIndex, content: event.ContentIndex}
}

func (indexes *streamWireIndexes) translate(event llmprotocol.Event) int {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		if indexes.items == nil {
			indexes.items = make(map[int]int)
		}
		if wireIndex, found := indexes.items[event.ItemIndex]; found {
			return wireIndex
		}
		wireIndex := indexes.next
		indexes.next++
		indexes.items[event.ItemIndex] = wireIndex
		return wireIndex
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta,
		llmprotocol.EventToolCallDelta, llmprotocol.EventOutputItemCompleted:
		if wireIndex, found := indexes.items[event.ItemIndex]; found {
			return wireIndex
		}
	}
	return event.ItemIndex
}

// sseFramer turns arbitrary transport chunks into complete SSE events. It is
// request-scoped, retains at most one bounded unfinished event, and accepts LF,
// CRLF, or CR line endings without assuming network read boundaries.
type sseFramer struct {
	buffer []byte
	limit  int
}

func newSSEFramer(limit int) sseFramer { return sseFramer{limit: limit} }

func (framer *sseFramer) Push(chunk []byte) ([][]byte, error) {
	if len(chunk) == 0 {
		return nil, nil
	}
	framer.buffer = append(framer.buffer, chunk...)
	frames := make([][]byte, 0, 1)
	for {
		end, complete := completeSSEFrame(framer.buffer)
		if !complete {
			break
		}
		if end > framer.limit {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is too large", nil)
		}
		frames = append(frames, append([]byte(nil), framer.buffer[:end]...))
		framer.buffer = append(framer.buffer[:0], framer.buffer[end:]...)
	}
	if len(framer.buffer) > framer.limit {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "unfinished SSE frame is too large", nil)
	}
	return frames, nil
}

func (framer *sseFramer) Finalize() ([][]byte, error) {
	if len(bytes.TrimSpace(framer.buffer)) == 0 {
		framer.buffer = nil
		return nil, nil
	}
	if len(framer.buffer) > framer.limit {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "unfinished SSE frame is too large", nil)
	}
	frame := append([]byte(nil), framer.buffer...)
	framer.buffer = nil
	return [][]byte{frame}, nil
}

func completeSSEFrame(payload []byte) (int, bool) {
	lineStart := 0
	for index := 0; index < len(payload); {
		lineEnd := index
		terminator := 0
		switch payload[index] {
		case '\n':
			terminator = 1
		case '\r':
			terminator = 1
			if index+1 < len(payload) && payload[index+1] == '\n' {
				terminator = 2
			}
		default:
			index++
			continue
		}
		if lineEnd == lineStart {
			return index + terminator, true
		}
		index += terminator
		lineStart = index
	}
	return 0, false
}

type (
	decoderFrameFinalizer func() ([][]byte, error)
	decoderFrameProcessor func([]byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error)
)

func finalizeDecoderFrames(
	finalize decoderFrameFinalizer,
	process decoderFrameProcessor,
	diagnosticLimit int,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := finalize()
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := process(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, diagnosticLimit)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func parseSSEFrame(frame []byte, limit int) (sseFrame, error) {
	return parseSSEFrameAtPosition(frame, limit, true)
}

func parseSSEFrameAtPosition(frame []byte, limit int, first bool) (sseFrame, error) {
	if len(frame) == 0 || len(frame) > limit {
		return sseFrame{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is empty or too large", nil)
	}
	if !utf8.Valid(frame) {
		return sseFrame{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_upstream_utf8",
			"upstream SSE frame is not valid UTF-8",
			nil,
		)
	}
	var result sseFrame
	// The EventSource wire grammar permits one UTF-8 BOM only at the start of
	// the stream, not at the start of every event.
	bom := []byte{0xef, 0xbb, 0xbf}
	if !first && bytes.HasPrefix(frame, bom) {
		return sseFrame{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unexpected_stream_bom",
			"SSE stream contains a UTF-8 BOM after its first event",
			nil,
		)
	}
	normalized := frame
	if first {
		normalized = bytes.TrimPrefix(normalized, bom)
	}
	normalized = bytes.ReplaceAll(normalized, []byte("\r\n"), []byte("\n"))
	normalized = bytes.ReplaceAll(normalized, []byte("\r"), []byte("\n"))
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		if len(line) == 0 || line[0] == ':' {
			continue
		}
		name, value, found := bytes.Cut(line, []byte{':'})
		if !found {
			name, value = line, nil
		}
		value = bytes.TrimPrefix(value, []byte{' '})
		switch string(name) {
		case "event":
			result.Event = string(value)
		case "data":
			result.HasData = true
			if len(result.Data) > 0 {
				result.Data = append(result.Data, '\n')
			}
			result.Data = append(result.Data, value...)
		}
	}
	return result, nil
}

func encodeSSE(event string, data any) ([]byte, error) {
	body, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}
	var buffer bytes.Buffer
	if event != "" {
		buffer.WriteString("event: ")
		buffer.WriteString(event)
		buffer.WriteByte('\n')
	}
	buffer.WriteString("data: ")
	buffer.Write(body)
	buffer.WriteString("\n\n")
	return buffer.Bytes(), nil
}

type streamState struct {
	context         llmprotocol.StreamContext
	policy          llmprotocol.Policy
	providerID      string
	providerModel   string
	sequence        uint64
	events          int
	wireFrames      int
	wireBytes       int
	terminal        bool
	started         bool
	usage           llmprotocol.Usage
	stop            llmprotocol.StopReason
	items           map[int]bool
	completedItems  map[int]bool
	itemKinds       map[int]llmprotocol.ContentKind
	itemIDs         map[int]string
	itemIDIndexes   map[string]int
	contentBlocks   map[streamContentKey]bool
	contentKinds    map[streamContentKey]llmprotocol.ContentKind
	reasoningScopes map[streamContentKey]llmprotocol.ReasoningScope
	itemTextBytes   map[streamContentKey]int
	itemTextRunes   map[streamContentKey]int64
	itemCitations   map[streamContentKey]int
	toolCalls       map[int]llmprotocol.ToolCall
	toolCallIndexes map[string]int
	toolArguments   map[int][]byte
}

func (state *streamState) observeProviderStreamBytes(chunk []byte) error {
	limit := state.policy.Limits.BodyBytes
	if limit > 0 && (state.wireBytes > limit || len(chunk) > limit-state.wireBytes) {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_body_limit",
			"upstream response stream exceeds the configured body limit",
			nil,
		)
	}
	state.wireBytes += len(chunk)
	return nil
}

func (state *streamState) parseProviderSSEFrame(frame []byte) (sseFrame, error) {
	parsed, err := parseSSEFrameAtPosition(
		frame,
		state.policy.Limits.SSEFrameBytes,
		state.wireFrames == 0,
	)
	state.wireFrames++
	return parsed, err
}

func (state *streamState) observeProviderIdentity(responseID, model string) error {
	if responseID != "" {
		if state.policy.Limits.IdentifierBytes > 0 && len(responseID) > state.policy.Limits.IdentifierBytes {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_response_id_limit",
				"upstream stream response ID exceeds the configured limit",
				nil,
			)
		}
		if state.providerID != "" && state.providerID != responseID {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_response_id_mismatch",
				"upstream stream changed response ID",
				nil,
			)
		}
		state.providerID = responseID
	}
	if model != "" {
		if state.policy.Limits.ModelBytes > 0 && len(model) > state.policy.Limits.ModelBytes {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_model_limit",
				"upstream stream model exceeds the configured limit",
				nil,
			)
		}
		if state.providerModel != "" && state.providerModel != model {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_model_mismatch",
				"upstream stream changed model",
				nil,
			)
		}
		state.providerModel = model
	}
	return nil
}

func (state *streamState) next(event llmprotocol.Event) (llmprotocol.Event, error) {
	event, err := state.prepareEvent(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	event, err = state.applyItemEvent(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	return state.applyEventEvidence(event)
}

func (state *streamState) prepareEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if state.terminal {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	if !validStreamEventType(event.Type) {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "unknown_stream_event", "upstream stream event type is invalid", nil)
	}
	state.events++
	if state.policy.Limits.Events > 0 && state.events > state.policy.Limits.Events {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_event_limit", "stream event limit exceeded", nil)
	}
	state.sequence++
	event.Sequence = state.sequence
	if event.ResponseID == "" {
		event.ResponseID = state.context.ResponseID
	}
	if state.context.PublicModel != "" {
		event.Model = state.context.PublicModel
	}
	state.ensureCollections()
	return state.prepareLifecycleEvent(event)
}

func validStreamEventType(eventType llmprotocol.EventType) bool {
	switch eventType {
	case llmprotocol.EventResponseStarted, llmprotocol.EventOutputItemStarted,
		llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta,
		llmprotocol.EventToolCallDelta, llmprotocol.EventOutputItemCompleted,
		llmprotocol.EventUsageUpdated, llmprotocol.EventResponseCompleted,
		llmprotocol.EventResponseFailed, llmprotocol.EventProviderOpaque:
		return true
	default:
		return false
	}
}

func (state *streamState) ensureCollections() {
	if state.items == nil {
		state.items = make(map[int]bool)
		state.completedItems = make(map[int]bool)
		state.itemKinds = make(map[int]llmprotocol.ContentKind)
		state.itemIDs = make(map[int]string)
		state.itemIDIndexes = make(map[string]int)
		state.contentBlocks = make(map[streamContentKey]bool)
		state.contentKinds = make(map[streamContentKey]llmprotocol.ContentKind)
		state.reasoningScopes = make(map[streamContentKey]llmprotocol.ReasoningScope)
		state.itemTextBytes = make(map[streamContentKey]int)
		state.itemTextRunes = make(map[streamContentKey]int64)
		state.itemCitations = make(map[streamContentKey]int)
		state.toolCalls = make(map[int]llmprotocol.ToolCall)
		state.toolCallIndexes = make(map[string]int)
		state.toolArguments = make(map[int][]byte)
		if state.usage.State == "" {
			state.usage.State = llmprotocol.UsageUnavailable
		}
	}
}

func (state *streamState) prepareLifecycleEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.Type == llmprotocol.EventResponseStarted {
		return state.prepareStartEvent(event)
	}
	if event.Type != llmprotocol.EventProviderOpaque && event.Type != llmprotocol.EventResponseFailed && !state.started {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream emitted output before response start", nil)
	}
	return event, nil
}

func (state *streamState) prepareStartEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if state.started {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_start", "upstream stream started more than once", nil)
	}
	if event.ResponseID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		event.ResponseID = llmprotocol.StableID("stream-response", state.context.PublicModel, state.context.ProviderModel)
	}
	if event.ResponseID == "" {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_response_id", "upstream stream response ID is missing", nil)
	}
	if state.policy.Limits.IdentifierBytes > 0 && len(event.ResponseID) > state.policy.Limits.IdentifierBytes {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_response_id_limit", "upstream stream response ID exceeds the configured limit", nil)
	}
	if state.policy.Limits.ModelBytes > 0 && len(event.Model) > state.policy.Limits.ModelBytes {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_model_limit", "upstream stream model exceeds the configured limit", nil)
	}
	state.context.ResponseID = event.ResponseID
	state.started = true
	return event, nil
}

func (state *streamState) applyItemEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		return state.startItem(event)
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta, llmprotocol.EventToolCallDelta:
		return state.applyDelta(event)
	case llmprotocol.EventOutputItemCompleted:
		return state.completeItem(event)
	}
	return event, nil
}

func (state *streamState) startItem(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.ItemIndex < 0 {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_index", "upstream output item index is invalid", nil)
	}
	if state.items[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_item_start", "upstream output item started more than once", nil)
	}
	if state.policy.Limits.OutputItems > 0 && len(state.items) >= state.policy.Limits.OutputItems {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "output_items_limit", "upstream output item limit exceeded", nil)
	}
	if event.ItemID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		event.ItemID = llmprotocol.StableID(event.ResponseID, string(event.Type), fmt.Sprint(event.ItemIndex))
	}
	if event.ItemID == "" {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id", "upstream output item ID is missing", nil)
	}
	if state.policy.Limits.IdentifierBytes > 0 && len(event.ItemID) > state.policy.Limits.IdentifierBytes {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id_limit", "upstream output item ID exceeds the configured limit", nil)
	}
	if index, duplicate := state.itemIDIndexes[event.ItemID]; duplicate && index != event.ItemIndex {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_item_id", "upstream stream reused an output item ID", nil)
	}
	if event.ToolCall != nil {
		if err := state.claimContentBlock(event); err != nil {
			return llmprotocol.Event{}, err
		}
		if err := state.validateStreamToolIdentity(*event.ToolCall, true); err != nil {
			return llmprotocol.Event{}, err
		}
		if err := state.validateStreamToolArgumentAppend(nil, event.ToolCall.Arguments); err != nil {
			return llmprotocol.Event{}, err
		}
		if err := state.claimStreamToolCallID(event.ItemIndex, event.ToolCall.ID); err != nil {
			return llmprotocol.Event{}, err
		}
	}
	state.items[event.ItemIndex] = true
	state.itemIDs[event.ItemIndex] = event.ItemID
	state.itemIDIndexes[event.ItemID] = event.ItemIndex
	if event.ToolCall != nil {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
		state.toolCalls[event.ItemIndex] = *event.ToolCall
		state.toolArguments[event.ItemIndex] = append([]byte(nil), event.ToolCall.Arguments...)
	} else if event.Content != nil {
		if err := state.claimContentBlock(event); err != nil {
			return llmprotocol.Event{}, err
		}
		state.itemKinds[event.ItemIndex] = event.Content.Kind
	}
	return event, nil
}

func (state *streamState) applyDelta(event llmprotocol.Event) (llmprotocol.Event, error) {
	if err := state.validateDelta(event); err != nil {
		return llmprotocol.Event{}, err
	}
	if event.Type == llmprotocol.EventOutputTextDelta || event.Type == llmprotocol.EventReasoningDelta {
		if err := state.recordTextDelta(event); err != nil {
			return llmprotocol.Event{}, err
		}
	}
	if event.Type == llmprotocol.EventToolCallDelta {
		return state.recordToolDelta(event)
	}
	state.recordDeltaKind(event)
	return event, nil
}

func (state *streamState) validateDelta(event llmprotocol.Event) error {
	if event.ContentIndex < 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_content_index", "upstream content index is invalid", nil)
	}
	if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream delta does not reference an active output item", nil)
	}
	if event.ItemID != "" && event.ItemID != state.itemIDs[event.ItemIndex] {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id_mismatch", "upstream delta changed its output item ID", nil)
	}
	if event.Content != nil && len(event.Content.Citations) > 0 &&
		(event.Type != llmprotocol.EventOutputTextDelta || event.Content.Kind != llmprotocol.ContentText) {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_citation", "upstream citations require a text delta", nil)
	}
	return state.claimContentBlock(event)
}

func (state *streamState) claimContentBlock(event llmprotocol.Event) error {
	if event.ContentIndex < 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_content_index", "upstream content index is invalid", nil)
	}
	key := contentKey(event)
	if state.contentBlocks[key] {
		if err := state.observeContentKind(key, eventContentKind(event)); err != nil {
			return err
		}
		return state.observeReasoningScope(key, event)
	}
	if state.policy.Limits.ContentBlocks > 0 && len(state.contentBlocks) >= state.policy.Limits.ContentBlocks {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "content_blocks_limit", "upstream content block limit exceeded", nil)
	}
	state.contentBlocks[key] = true
	if err := state.observeContentKind(key, eventContentKind(event)); err != nil {
		return err
	}
	return state.observeReasoningScope(key, event)
}

func eventContentKind(event llmprotocol.Event) llmprotocol.ContentKind {
	if event.ToolCall != nil || event.Type == llmprotocol.EventToolCallDelta {
		return llmprotocol.ContentToolCall
	}
	if event.Content != nil && event.Content.Kind != "" {
		return event.Content.Kind
	}
	if event.Type == llmprotocol.EventReasoningDelta {
		return llmprotocol.ContentReasoning
	}
	if event.Type == llmprotocol.EventOutputTextDelta {
		return llmprotocol.ContentText
	}
	return ""
}

func (state *streamState) observeContentKind(key streamContentKey, kind llmprotocol.ContentKind) error {
	if kind == "" {
		return nil
	}
	if existing := state.contentKinds[key]; existing != "" && existing != kind {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_content_kind_mismatch",
			"upstream stream changed a content block kind",
			nil,
		)
	}
	state.contentKinds[key] = kind
	return nil
}

func (state *streamState) observeReasoningScope(key streamContentKey, event llmprotocol.Event) error {
	if event.Content == nil || event.Content.Kind != llmprotocol.ContentReasoning || event.Content.Reasoning == "" {
		return nil
	}
	scope := event.Content.Reasoning
	if scope != llmprotocol.ReasoningScopeText && scope != llmprotocol.ReasoningScopeSummary {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_reasoning_scope",
			"upstream stream emitted an unsupported reasoning scope",
			nil,
		)
	}
	if existing := state.reasoningScopes[key]; existing != "" && existing != scope {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_reasoning_scope_mismatch",
			"upstream stream changed a reasoning content scope",
			nil,
		)
	}
	state.reasoningScopes[key] = scope
	return nil
}

func (state *streamState) recordDeltaKind(event llmprotocol.Event) {
	if event.Content != nil {
		state.itemKinds[event.ItemIndex] = event.Content.Kind
	} else if event.Type == llmprotocol.EventReasoningDelta {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentReasoning
	} else if state.itemKinds[event.ItemIndex] == "" {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentText
	}
}

func (state *streamState) recordTextDelta(event llmprotocol.Event) error {
	key := contentKey(event)
	textBytes := state.itemTextBytes[key] + len(event.Delta)
	if state.policy.Limits.TextBytes > 0 && textBytes > state.policy.Limits.TextBytes {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "text_limit", "content text exceeds the configured limit", nil)
	}
	textRunes := state.itemTextRunes[key] + int64(utf8.RuneCountInString(event.Delta))
	citationCount := state.itemCitations[key]
	var citationBatch []llmprotocol.Citation
	if event.Content != nil && len(event.Content.Citations) > 0 {
		citationBatch = event.Content.Citations
		citationCount += len(citationBatch)
	}
	if err := llmprotocol.ValidateCitationBatch(textRunes, citationCount, citationBatch, state.policy.Limits); err != nil {
		return err
	}
	state.itemTextBytes[key] = textBytes
	state.itemTextRunes[key] = textRunes
	state.itemCitations[key] = citationCount
	return nil
}

func (state *streamState) recordToolDelta(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.ToolCall == nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "tool_delta_missing", "upstream tool delta is missing", nil)
	}
	state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
	call := state.toolCalls[event.ItemIndex]
	merged, err := state.mergeStreamToolIdentity(call, *event.ToolCall)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	call = merged
	if err := state.claimStreamToolCallID(event.ItemIndex, call.ID); err != nil {
		return llmprotocol.Event{}, err
	}
	state.toolCalls[event.ItemIndex] = call
	event.ToolCall.ID, event.ToolCall.Name = call.ID, call.Name
	current := state.toolArguments[event.ItemIndex]
	if bytes.Equal(bytes.TrimSpace(current), []byte("{}")) && event.ToolCall.Arguments != "" {
		current = nil
	}
	if err := state.validateStreamToolArgumentAppend(current, event.ToolCall.Arguments); err != nil {
		return llmprotocol.Event{}, err
	}
	state.toolArguments[event.ItemIndex] = append(current, event.ToolCall.Arguments...)
	return event, nil
}

func (state *streamState) completeItem(event llmprotocol.Event) (llmprotocol.Event, error) {
	if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream completed an inactive output item", nil)
	}
	expectedItemID := state.itemIDs[event.ItemIndex]
	if event.ItemID != "" && event.ItemID != expectedItemID {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id_mismatch", "upstream completion changed its output item ID", nil)
	}
	event.ItemID = expectedItemID
	if state.itemKinds[event.ItemIndex] == llmprotocol.ContentToolCall {
		completed, err := state.completeToolItem(event)
		if err != nil {
			return llmprotocol.Event{}, err
		}
		event = completed
	} else if event.Content == nil && state.itemKinds[event.ItemIndex] != "" {
		event.Content = &llmprotocol.Content{Kind: state.itemKinds[event.ItemIndex]}
	}
	state.markItemComplete(event.ItemIndex)
	return event, nil
}

func (state *streamState) completeToolItem(event llmprotocol.Event) (llmprotocol.Event, error) {
	arguments, err := state.finalToolArguments(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	call := state.toolCalls[event.ItemIndex]
	if event.ToolCall != nil {
		call, err = state.mergeStreamToolIdentity(call, *event.ToolCall)
		if err != nil {
			return llmprotocol.Event{}, err
		}
	}
	if err := state.validateStreamToolIdentity(call, true); err != nil {
		return llmprotocol.Event{}, err
	}
	if err := state.claimStreamToolCallID(event.ItemIndex, call.ID); err != nil {
		return llmprotocol.Event{}, err
	}
	call.Arguments = string(arguments)
	event.ToolCall = &call
	return event, nil
}

func (state *streamState) mergeStreamToolIdentity(current, incoming llmprotocol.ToolCall) (llmprotocol.ToolCall, error) {
	if err := state.validateStreamToolIdentity(incoming, false); err != nil {
		return llmprotocol.ToolCall{}, err
	}
	if incoming.ID != "" {
		if current.ID != "" && current.ID != incoming.ID {
			return llmprotocol.ToolCall{}, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_tool_identity_mismatch",
				"upstream stream changed a tool call ID",
				nil,
			)
		}
		current.ID = incoming.ID
	}
	if incoming.Name != "" {
		if current.Name != "" && current.Name != incoming.Name {
			return llmprotocol.ToolCall{}, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_tool_identity_mismatch",
				"upstream stream changed a tool name",
				nil,
			)
		}
		current.Name = incoming.Name
	}
	return current, nil
}

func (state *streamState) claimStreamToolCallID(itemIndex int, callID string) error {
	if callID == "" {
		return nil
	}
	if index, duplicate := state.toolCallIndexes[callID]; duplicate && index != itemIndex {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"duplicate_stream_tool_call_id",
			"upstream stream reused a tool call ID",
			nil,
		)
	}
	state.toolCallIndexes[callID] = itemIndex
	return nil
}

func (state *streamState) validateStreamToolIdentity(call llmprotocol.ToolCall, required bool) error {
	if len(call.ID) > state.policy.Limits.IdentifierBytes || len(call.Name) > state.policy.Limits.ToolNameBytes {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_tool_identity_limit",
			"upstream streamed tool identity exceeds the configured limit",
			nil,
		)
	}
	if required && (strings.TrimSpace(call.ID) == "" || strings.TrimSpace(call.Name) == "") {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_tool_identity_required",
			"upstream streamed tool call requires an ID and name",
			nil,
		)
	}
	return nil
}

func (state *streamState) validateStreamToolArgumentAppend(current []byte, incoming string) error {
	for _, limit := range []int{
		state.policy.Limits.UnfinishedArguments,
		state.policy.Limits.ToolArgumentsBytes,
	} {
		if limit > 0 && (len(current) > limit || len(incoming) > limit-len(current)) {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"tool_arguments_limit",
				"streamed tool arguments exceed the configured limit",
				nil,
			)
		}
	}
	return nil
}

func (state *streamState) finalToolArguments(event llmprotocol.Event) ([]byte, error) {
	arguments := state.toolArguments[event.ItemIndex]
	if event.ToolCall != nil && event.ToolCall.Arguments != "" {
		if err := state.validateStreamToolArgumentAppend(nil, event.ToolCall.Arguments); err != nil {
			return nil, err
		}
		if len(arguments) > 0 && string(arguments) != event.ToolCall.Arguments {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_mismatch", "upstream final tool arguments do not match streamed arguments", nil)
		}
		arguments = []byte(event.ToolCall.Arguments)
	}
	if !isJSONObject(arguments, state.policy.Limits.JSONDepth) {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_tool_arguments", "upstream streamed tool arguments are not a JSON object", nil)
	}
	return arguments, nil
}

func (state *streamState) markItemComplete(itemIndex int) {
	state.completedItems[itemIndex] = true
	for key := range state.itemTextBytes {
		if key.item != itemIndex {
			continue
		}
		delete(state.itemTextBytes, key)
		delete(state.itemTextRunes, key)
		delete(state.itemCitations, key)
	}
	delete(state.toolArguments, itemIndex)
}

func (state *streamState) applyEventEvidence(event llmprotocol.Event) (llmprotocol.Event, error) {
	prepared, err := state.applyUsageEvidence(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	event = prepared
	if event.StopReason != "" {
		state.stop = event.StopReason
	}
	if event.Type != llmprotocol.EventResponseCompleted && event.Type != llmprotocol.EventResponseFailed {
		return event, nil
	}
	return state.applyTerminalEvent(event)
}

func (state *streamState) applyUsageEvidence(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.Usage == nil {
		return event, nil
	}
	merged, err := mergeMonotonicUsage(state.usage, *event.Usage)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	state.usage = merged
	usage := state.usage
	event.Usage = &usage
	return event, nil
}

func (state *streamState) applyTerminalEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	var err error
	if event.Type == llmprotocol.EventResponseCompleted {
		event, err = state.validateCompletedEvent(event)
	} else {
		if event.Usage == nil && state.usage.State == llmprotocol.UsageAvailable {
			usage := state.usage
			event.Usage = &usage
		}
		event, err = validateFailedEvent(event)
	}
	if err != nil {
		return llmprotocol.Event{}, err
	}
	state.terminal = true
	return event, nil
}

func (state *streamState) validateCompletedEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if !state.started {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream completed before response start", nil)
	}
	if len(state.items) == 0 && event.StopReason != llmprotocol.StopContentFilter {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_output_missing", "upstream stream completed without output", nil)
	}
	for itemIndex := range state.items {
		if !state.completedItems[itemIndex] {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_incomplete", "upstream stream completed with an active output item", nil)
		}
	}
	if len(state.toolArguments) != 0 {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_incomplete", "upstream stream completed with unfinished tool arguments", nil)
	}
	if event.Error != nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_terminal_shape", "completed stream cannot contain an error", nil)
	}
	if event.StopReason == "" {
		event.StopReason = llmprotocol.StopUnknown
	}
	if event.StopReason == llmprotocol.StopSequence && event.MatchedStopSequence == "" {
		return llmprotocol.Event{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"matched_stop_sequence_required",
			"upstream stop_sequence reason requires the matched sequence",
			nil,
		)
	}
	if event.StopReason != llmprotocol.StopSequence && event.MatchedStopSequence != "" {
		return llmprotocol.Event{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"matched_stop_sequence_reason",
			"upstream matched stop sequence requires stop_sequence reason",
			nil,
		)
	}
	if event.Usage == nil {
		usage := state.usage
		event.Usage = &usage
	}
	if err := llmprotocol.ValidateUsage(*event.Usage); err != nil {
		return llmprotocol.Event{}, err
	}
	return event, nil
}

func validateFailedEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.Error == nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_failure_shape", "failed stream requires an error", nil)
	}
	if event.Usage != nil {
		if err := llmprotocol.ValidateUsage(*event.Usage); err != nil {
			return llmprotocol.Event{}, err
		}
	}
	switch event.Failure {
	case "":
		event.Failure = llmprotocol.FailureTransport
	case llmprotocol.FailureTransport, llmprotocol.FailureResponse:
	default:
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_failure_scope", "failed stream has an invalid failure scope", nil)
	}
	return event, nil
}

func mergeMonotonicUsage(current, update llmprotocol.Usage) (llmprotocol.Usage, error) {
	if err := llmprotocol.ValidateUsage(update); err != nil {
		return llmprotocol.Usage{}, err
	}
	if current.State == llmprotocol.UsageAvailable && update.State == llmprotocol.UsageUnavailable {
		return llmprotocol.Usage{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_evidence_decreased", "upstream streaming usage evidence became unavailable", nil)
	}
	currentCounts := usageCounts(current)
	updateCounts := usageCounts(update)
	merged := make([]llmprotocol.TokenCount, len(currentCounts))
	for index := range merged {
		value, err := mergeTokenCount(currentCounts[index], updateCounts[index])
		if err != nil {
			return llmprotocol.Usage{}, err
		}
		merged[index] = value
	}
	result := usageFromCounts(mergedUsageState(current.State, update.State), merged)
	if err := deriveUsageTotal(&result); err != nil {
		return llmprotocol.Usage{}, err
	}
	if err := llmprotocol.ValidateUsage(result); err != nil {
		return llmprotocol.Usage{}, err
	}
	return result, nil
}

func mergeTokenCount(existing, incoming llmprotocol.TokenCount) (llmprotocol.TokenCount, error) {
	if incoming.Value == nil {
		return existing, nil
	}
	if existing.Value != nil && *existing.Value > *incoming.Value {
		return llmprotocol.TokenCount{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_decreased", "upstream streaming usage counter decreased", nil)
	}
	if existing.Value != nil && usageEvidenceRank(incoming.Provenance) < usageEvidenceRank(existing.Provenance) {
		return llmprotocol.TokenCount{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_evidence_decreased", "upstream streaming usage evidence quality decreased", nil)
	}
	return incoming, nil
}

func mergedUsageState(current, update llmprotocol.UsageState) llmprotocol.UsageState {
	if update != "" {
		return update
	}
	if current != "" {
		return current
	}
	return llmprotocol.UsageUnavailable
}

func usageCounts(usage llmprotocol.Usage) []llmprotocol.TokenCount {
	return []llmprotocol.TokenCount{
		usage.InputUncached, usage.InputCacheRead, usage.InputCacheWrite,
		usage.OutputReasoning, usage.OutputOther, usage.InputTotal, usage.OutputTotal, usage.Total,
	}
}

func usageFromCounts(state llmprotocol.UsageState, counts []llmprotocol.TokenCount) llmprotocol.Usage {
	return llmprotocol.Usage{
		State: state, InputUncached: counts[0], InputCacheRead: counts[1], InputCacheWrite: counts[2],
		OutputReasoning: counts[3], OutputOther: counts[4], InputTotal: counts[5],
		OutputTotal: counts[6], Total: counts[7],
	}
}

func deriveUsageTotal(usage *llmprotocol.Usage) error {
	if usage.InputTotal.Value == nil || usage.OutputTotal.Value == nil ||
		usage.Total.Value != nil && usage.Total.Provenance != llmprotocol.UsageDerived {
		return nil
	}
	if *usage.OutputTotal.Value > math.MaxInt64-*usage.InputTotal.Value {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_overflow", "upstream streaming usage total overflowed", nil)
	}
	value := *usage.InputTotal.Value + *usage.OutputTotal.Value
	usage.Total = llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageDerived}
	return nil
}

func usageEvidenceRank(provenance llmprotocol.UsageProvenance) int {
	switch provenance {
	case llmprotocol.UsageAuthoritative:
		return 3
	case llmprotocol.UsageDerived:
		return 2
	case llmprotocol.UsageEstimated:
		return 1
	default:
		return 0
	}
}

func (state *streamState) finalize(reason error) ([]llmprotocol.Event, error) {
	if state.terminal {
		return nil, nil
	}
	if reason == nil {
		reason = errors.New("upstream stream ended without a terminal event")
	}
	protocolError := streamFinalizationError(reason, "upstream stream ended before completion")
	event, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: protocolError, StopReason: llmprotocol.StopError})
	if err != nil {
		return nil, err
	}
	return []llmprotocol.Event{event}, nil
}

func streamFinalizationError(reason error, incompleteMessage string) *llmprotocol.ProtocolError {
	var protocolError *llmprotocol.ProtocolError
	if errors.As(reason, &protocolError) {
		return protocolError
	}
	protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", incompleteMessage, reason)
	switch {
	case errors.Is(reason, context.Canceled):
		protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_canceled", "stream was canceled", reason)
	case errors.Is(reason, context.DeadlineExceeded):
		protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_timeout", "stream deadline was exceeded", reason)
	}
	return protocolError
}
