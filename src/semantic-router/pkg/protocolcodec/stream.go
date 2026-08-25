package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type sseFrame struct {
	Event string
	Data  []byte
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
	if len(frame) == 0 || len(frame) > limit {
		return sseFrame{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is empty or too large", nil)
	}
	var result sseFrame
	normalized := bytes.ReplaceAll(frame, []byte("\r\n"), []byte("\n"))
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
			if len(result.Data) > 0 {
				result.Data = append(result.Data, '\n')
			}
			result.Data = append(result.Data, value...)
		}
	}
	if len(result.Data) == 0 {
		return sseFrame{}, nil
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
	context        llmprotocol.StreamContext
	policy         llmprotocol.Policy
	sequence       uint64
	events         int
	terminal       bool
	started        bool
	usage          llmprotocol.Usage
	stop           llmprotocol.StopReason
	items          map[int]bool
	completedItems map[int]bool
	itemKinds      map[int]llmprotocol.ContentKind
	itemIDs        map[int]string
	itemTextBytes  map[int]int
	itemTextRunes  map[int]int64
	itemCitations  map[int]int
	toolCalls      map[int]llmprotocol.ToolCall
	toolArguments  map[int][]byte
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
	switch event.Type {
	case llmprotocol.EventResponseStarted, llmprotocol.EventOutputItemStarted,
		llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta,
		llmprotocol.EventToolCallDelta, llmprotocol.EventOutputItemCompleted,
		llmprotocol.EventUsageUpdated, llmprotocol.EventResponseCompleted,
		llmprotocol.EventResponseFailed, llmprotocol.EventProviderOpaque:
	default:
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
	if state.items == nil {
		state.items = make(map[int]bool)
		state.completedItems = make(map[int]bool)
		state.itemKinds = make(map[int]llmprotocol.ContentKind)
		state.itemIDs = make(map[int]string)
		state.itemTextBytes = make(map[int]int)
		state.itemTextRunes = make(map[int]int64)
		state.itemCitations = make(map[int]int)
		state.toolCalls = make(map[int]llmprotocol.ToolCall)
		state.toolArguments = make(map[int][]byte)
		if state.usage.State == "" {
			state.usage.State = llmprotocol.UsageUnavailable
		}
	}
	if event.Type == llmprotocol.EventResponseStarted {
		if state.started {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_start", "upstream stream started more than once", nil)
		}
		if event.ResponseID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
			event.ResponseID = llmprotocol.StableID("stream-response", state.context.PublicModel, state.context.ProviderModel)
		}
		if event.ResponseID == "" {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_response_id", "upstream stream response ID is missing", nil)
		}
		state.context.ResponseID = event.ResponseID
		state.started = true
	} else if event.Type != llmprotocol.EventProviderOpaque && event.Type != llmprotocol.EventResponseFailed && !state.started {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream emitted output before response start", nil)
	}
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
	state.items[event.ItemIndex] = true
	if event.ItemID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		event.ItemID = llmprotocol.StableID(event.ResponseID, string(event.Type), fmt.Sprint(event.ItemIndex))
	}
	if event.ItemID == "" {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id", "upstream output item ID is missing", nil)
	}
	state.itemIDs[event.ItemIndex] = event.ItemID
	if event.ToolCall != nil {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
		state.toolCalls[event.ItemIndex] = *event.ToolCall
		state.toolArguments[event.ItemIndex] = append([]byte(nil), event.ToolCall.Arguments...)
	} else if event.Content != nil {
		state.itemKinds[event.ItemIndex] = event.Content.Kind
	}
	return event, nil
}

func (state *streamState) applyDelta(event llmprotocol.Event) (llmprotocol.Event, error) {
	if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream delta does not reference an active output item", nil)
	}
	if event.Content != nil && len(event.Content.Citations) > 0 &&
		(event.Type != llmprotocol.EventOutputTextDelta || event.Content.Kind != llmprotocol.ContentText) {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_citation", "upstream citations require a text delta", nil)
	}
	if event.Type == llmprotocol.EventOutputTextDelta {
		if err := state.recordTextDelta(event); err != nil {
			return llmprotocol.Event{}, err
		}
	}
	if event.Type == llmprotocol.EventToolCallDelta {
		return state.recordToolDelta(event)
	}
	if event.Content != nil {
		state.itemKinds[event.ItemIndex] = event.Content.Kind
	} else if event.Type == llmprotocol.EventReasoningDelta {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentReasoning
	} else if state.itemKinds[event.ItemIndex] == "" {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentText
	}
	return event, nil
}

func (state *streamState) recordTextDelta(event llmprotocol.Event) error {
	textBytes := state.itemTextBytes[event.ItemIndex] + len(event.Delta)
	if state.policy.Limits.TextBytes > 0 && textBytes > state.policy.Limits.TextBytes {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "text_limit", "content text exceeds the configured limit", nil)
	}
	textRunes := state.itemTextRunes[event.ItemIndex] + int64(utf8.RuneCountInString(event.Delta))
	citationCount := state.itemCitations[event.ItemIndex]
	var citationBatch []llmprotocol.Citation
	if event.Content != nil && len(event.Content.Citations) > 0 {
		citationBatch = event.Content.Citations
		citationCount += len(citationBatch)
	}
	if err := llmprotocol.ValidateCitationBatch(textRunes, citationCount, citationBatch, state.policy.Limits); err != nil {
		return err
	}
	state.itemTextBytes[event.ItemIndex] = textBytes
	state.itemTextRunes[event.ItemIndex] = textRunes
	state.itemCitations[event.ItemIndex] = citationCount
	return nil
}

func (state *streamState) recordToolDelta(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.ToolCall == nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "tool_delta_missing", "upstream tool delta is missing", nil)
	}
	state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
	call := state.toolCalls[event.ItemIndex]
	if event.ToolCall.ID != "" {
		call.ID = event.ToolCall.ID
	}
	if event.ToolCall.Name != "" {
		call.Name = event.ToolCall.Name
	}
	state.toolCalls[event.ItemIndex] = call
	event.ToolCall.ID, event.ToolCall.Name = call.ID, call.Name
	current := state.toolArguments[event.ItemIndex]
	if bytes.Equal(bytes.TrimSpace(current), []byte("{}")) && event.ToolCall.Arguments != "" {
		current = nil
	}
	if state.policy.Limits.UnfinishedArguments > 0 && len(event.ToolCall.Arguments) > state.policy.Limits.UnfinishedArguments-len(current) {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "tool_arguments_limit", "unfinished tool arguments exceed the configured limit", nil)
	}
	state.toolArguments[event.ItemIndex] = append(current, event.ToolCall.Arguments...)
	return event, nil
}

func (state *streamState) completeItem(event llmprotocol.Event) (llmprotocol.Event, error) {
	if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream completed an inactive output item", nil)
	}
	if state.itemKinds[event.ItemIndex] == llmprotocol.ContentToolCall {
		arguments := state.toolArguments[event.ItemIndex]
		if event.ToolCall != nil && event.ToolCall.Arguments != "" {
			if len(arguments) > 0 && string(arguments) != event.ToolCall.Arguments {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_mismatch", "upstream final tool arguments do not match streamed arguments", nil)
			}
			arguments = []byte(event.ToolCall.Arguments)
		}
		if !isJSONObject(arguments, state.policy.Limits.JSONDepth) {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_tool_arguments", "upstream streamed tool arguments are not a JSON object", nil)
		}
		call := state.toolCalls[event.ItemIndex]
		if event.ToolCall != nil {
			if event.ToolCall.ID != "" {
				call.ID = event.ToolCall.ID
			}
			if event.ToolCall.Name != "" {
				call.Name = event.ToolCall.Name
			}
		}
		call.Arguments = string(arguments)
		event.ToolCall = &call
	} else if event.Content == nil && state.itemKinds[event.ItemIndex] != "" {
		event.Content = &llmprotocol.Content{Kind: state.itemKinds[event.ItemIndex]}
	}
	state.completedItems[event.ItemIndex] = true
	delete(state.itemTextBytes, event.ItemIndex)
	delete(state.itemTextRunes, event.ItemIndex)
	delete(state.itemCitations, event.ItemIndex)
	delete(state.toolArguments, event.ItemIndex)
	return event, nil
}

func (state *streamState) applyEventEvidence(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.Usage != nil {
		merged, err := mergeMonotonicUsage(state.usage, *event.Usage)
		if err != nil {
			return llmprotocol.Event{}, err
		}
		state.usage = merged
		usage := state.usage
		event.Usage = &usage
	}
	if event.StopReason != "" {
		state.stop = event.StopReason
	}
	if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
		if event.Type == llmprotocol.EventResponseCompleted {
			if !state.started {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream completed before response start", nil)
			}
			if len(state.items) == 0 {
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
			if event.Usage == nil {
				usage := state.usage
				event.Usage = &usage
			}
			if err := llmprotocol.ValidateUsage(*event.Usage); err != nil {
				return llmprotocol.Event{}, err
			}
		} else if event.Error == nil || event.Usage != nil {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_failure_shape", "failed stream requires an error and cannot contain successful usage", nil)
		}
		state.terminal = true
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
	merge := func(existing, incoming llmprotocol.TokenCount) (llmprotocol.TokenCount, error) {
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
	state := update.State
	if state == "" {
		state = current.State
	}
	if state == "" {
		state = llmprotocol.UsageUnavailable
	}
	currentCounts := []llmprotocol.TokenCount{current.InputUncached, current.InputCacheRead, current.InputCacheWrite, current.OutputReasoning, current.OutputOther, current.InputTotal, current.OutputTotal, current.Total}
	updateCounts := []llmprotocol.TokenCount{update.InputUncached, update.InputCacheRead, update.InputCacheWrite, update.OutputReasoning, update.OutputOther, update.InputTotal, update.OutputTotal, update.Total}
	merged := make([]llmprotocol.TokenCount, len(currentCounts))
	for index := range merged {
		value, err := merge(currentCounts[index], updateCounts[index])
		if err != nil {
			return llmprotocol.Usage{}, err
		}
		merged[index] = value
	}
	result := llmprotocol.Usage{State: state, InputUncached: merged[0], InputCacheRead: merged[1], InputCacheWrite: merged[2], OutputReasoning: merged[3], OutputOther: merged[4], InputTotal: merged[5], OutputTotal: merged[6], Total: merged[7]}
	if result.InputTotal.Value != nil && result.OutputTotal.Value != nil &&
		(result.Total.Value == nil || result.Total.Provenance == llmprotocol.UsageDerived) {
		if *result.OutputTotal.Value > math.MaxInt64-*result.InputTotal.Value {
			return llmprotocol.Usage{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_overflow", "upstream streaming usage total overflowed", nil)
		}
		value := *result.InputTotal.Value + *result.OutputTotal.Value
		result.Total = llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageDerived}
	}
	if err := llmprotocol.ValidateUsage(result); err != nil {
		return llmprotocol.Usage{}, err
	}
	return result, nil
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
	protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", "upstream stream ended before completion", reason)
	if errors.Is(reason, context.Canceled) {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_canceled", "stream was canceled", reason)
	}
	event, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: protocolError, StopReason: llmprotocol.StopError})
	if err != nil {
		return nil, err
	}
	return []llmprotocol.Event{event}, nil
}
