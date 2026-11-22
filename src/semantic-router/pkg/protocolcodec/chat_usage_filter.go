package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ChatUsageStreamFilter frames a same-format Chat stream before exposing it to
// the client. It removes Router-requested accounting evidence when the client
// did not opt into usage, and provider decorations that the neutral codec
// reports as dropped. When the client opted into usage, a chunk without usage
// gains "usage": null and keeps its other bytes; other undecorated frames
// retain their original bytes. A provider error event ends the public stream:
// the filter drops it and every later frame, and UpstreamFailed reports it.
// Tool-call argument fragments are additionally held until their call
// finishes, so an upstream that dies mid-call cannot feed the client an
// arguments string no JSON decoder can read.
type ChatUsageStreamFilter struct {
	framer          sseFramer
	stripUsage      bool
	seenFinish      map[int]bool
	frames          int
	pendingTerminal []byte
	failure         error
	finalized       bool
	upstreamFailed  bool
	// outputLimit is the output token limit the backend received, 0 for none.
	// A frame that ends the choice for its tool calls is held with the frames
	// after it until usage tells whether the output reached the limit, which
	// the Router then reports as the stop, as the neutral stream decodes it.
	outputLimit int64
	held        *heldChatFinish
	// Tool-call argument fragments held until their call finishes.
	// releaseHeldToolFrames sends them verbatim when the call finishes as
	// sent, and repairHeldToolCalls closes them when the stream ends without
	// one.
	heldFrames      [][]byte
	heldFirst       []byte
	heldOrder       []heldChatToolKey
	heldCalls       map[heldChatToolKey]*heldChatToolCall
	maximumDepth    int
	heldAtReadStart int
}

// heldChatFinish holds a frame that ends the choice for its tool calls and the
// frames after it.
type heldChatFinish struct {
	frames [][]byte
	first  bool // the finish frame opened the stream
}

// chatFrameEvidence is what a frame tells the output limit check.
type chatFrameEvidence struct {
	toolCallFinish bool   // the frame ends choice 0 for its tool calls
	outputTokens   *int64 // the completion tokens of the frame's usage
}

// heldChatToolCall accumulates one streamed tool call's identity and argument
// fragments. The key pairs the choice index with the call's index inside the
// choice, because one choice can stream several calls in parallel.
type heldChatToolKey struct {
	choice int
	tool   int
}

type heldChatToolCall struct {
	id        string
	name      string
	callType  string
	arguments []byte
}

// chatFilterToolCallWire is the tool-call delta the public filter reads to
// hold a call's arguments together.
type chatFilterToolCallWire struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// errChatUpstreamErrorFrame marks a provider error event. The filter drops it
// and every frame after it; the Router sends its own error frame instead.
var errChatUpstreamErrorFrame = errors.New("chat stream carried a provider error event")

func NewChatUsageStreamFilter(limit int) *ChatUsageStreamFilter {
	return NewChatPublicStreamFilter(limit, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
}

// NewChatPublicStreamFilter preserves opted-in usage while removing provider
// decorations that cannot appear in the public Chat response contract. The
// maximumDepth bounds the JSON a held call's arguments may nest when the
// stream ends without a finish and the repair closes them.
func NewChatPublicStreamFilter(limit int, includeUsage bool, maximumDepth int) *ChatUsageStreamFilter {
	return &ChatUsageStreamFilter{
		framer: newSSEFramer(limit), stripUsage: !includeUsage,
		seenFinish:   make(map[int]bool),
		maximumDepth: maximumDepth,
	}
}

func (filter *ChatUsageStreamFilter) Push(chunk []byte) ([]byte, error) {
	if filter == nil {
		return nil, nil
	}
	if filter.failure != nil {
		return nil, filter.failure
	}
	if filter.finalized {
		return nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already finalized", nil)
	}
	filter.heldAtReadStart = len(filter.heldFrames)
	frames, err := filter.framer.Push(chunk)
	if err != nil {
		return nil, filter.poison(err)
	}
	return filter.filterFrames(frames)
}

// UpstreamFailed reports whether the stream carried a provider error event.
// The filter dropped that event, so when this is true the caller must send the
// Router's error frame in its place.
func (filter *ChatUsageStreamFilter) UpstreamFailed() bool {
	return filter != nil && filter.upstreamFailed
}

func (filter *ChatUsageStreamFilter) Finalize() ([]byte, error) {
	if filter == nil {
		return nil, nil
	}
	if filter.finalized {
		return nil, nil
	}
	filter.finalized = true
	if filter.failure != nil {
		return nil, filter.failure
	}
	frames, err := filter.framer.Finalize()
	if err != nil {
		return nil, filter.poison(err)
	}
	output, err := filter.filterFrames(frames)
	if err != nil {
		return nil, err
	}
	if repaired, ok := filter.repairHeldToolCalls(); ok {
		mainHeld, heldErr := filter.releaseHeld(false)
		if heldErr != nil {
			return nil, filter.poison(heldErr)
		}
		return append(append(output, mainHeld...), repaired...), nil
	}
	if released := filter.routeRelease(filter.releaseHeldToolFrames()); len(released) != 0 {
		output = append(output, released...)
	}
	mainHeld, err := filter.releaseHeld(false)
	if err != nil {
		return nil, filter.poison(err)
	}
	output = append(output, mainHeld...)
	output = append(output, filter.pendingTerminal...)
	filter.pendingTerminal = nil
	return output, nil
}

func (filter *ChatUsageStreamFilter) filterFrames(frames [][]byte) ([]byte, error) {
	var output bytes.Buffer
	for _, frame := range frames {
		if filter.upstreamFailed {
			continue
		}
		first := filter.frames == 0
		var evidence *chatFrameEvidence
		if filter.outputLimit > 0 {
			evidence = &chatFrameEvidence{}
		}
		filtered, keep, hasData, terminal, choices, err := filterChatUsageFrame(
			frame, filter.framer.limit, first, filter.stripUsage, filter.seenFinish, evidence,
		)
		filter.frames++
		if errors.Is(err, errChatUpstreamErrorFrame) {
			if len(filter.pendingTerminal) == 0 {
				filter.upstreamFailed = true
				output.Write(filter.routeRelease(filter.releaseHeldToolFrames()))
				mainHeld, heldErr := filter.releaseHeld(false)
				if heldErr != nil {
					return nil, filter.poison(heldErr)
				}
				output.Write(mainHeld)
				continue
			}
			// After [DONE], an error event is data after the terminal, as before.
			err = nil
		}
		if err != nil {
			return nil, filter.poison(err)
		}
		if len(filter.pendingTerminal) != 0 && hasData {
			return nil, filter.poison(invalidProviderResponse(
				"stream_event_after_terminal",
				"Chat stream emitted data after its terminal sentinel",
			))
		}
		if terminal {
			output.Write(filter.routeRelease(filter.releaseHeldToolFrames()))
			mainHeld, heldErr := filter.releaseHeld(false)
			if heldErr != nil {
				return nil, filter.poison(heldErr)
			}
			output.Write(mainHeld)
			filter.pendingTerminal = append(filter.pendingTerminal[:0], filtered...)
			continue
		}
		if keep && hasData && filter.holdToolFrame(filtered, choices) {
			continue
		}
		// The held fragments go out only when the call's fate is decided: a
		// finish reason completes or abandons the call, [DONE] and provider
		// error events release below, and a stream that ends without either
		// reaches the repair. Frames that carry no finish — usage-only
		// chunks, empty deltas, content — say nothing about the call and
		// keep the hold.
		if keep && hasData && chatChoicesCarryFinish(choices) {
			output.Write(filter.routeRelease(filter.releaseHeldToolFrames()))
		}
		passed, err := filter.pass(filtered, keep, first, evidence)
		if err != nil {
			return nil, filter.poison(err)
		}
		output.Write(passed)
	}
	return output.Bytes(), nil
}

// routeRelease returns the released tool-argument frames, or queues them
// behind the output limit's held frames when that hold is active, so both
// holds emit the stream in arrival order.
func (filter *ChatUsageStreamFilter) routeRelease(frames [][]byte) []byte {
	if len(frames) == 0 {
		return nil
	}
	if filter.held != nil {
		filter.held.frames = append(filter.held.frames, frames...)
		return nil
	}
	return bytes.Join(frames, nil)
}

// pass returns what the client gets now of a filtered frame. A frame that
// ends the choice for its tool calls waits, with the frames after it, for the
// usage that tells whether the output reached the output limit.
func (filter *ChatUsageStreamFilter) pass(frame []byte, keep, first bool, evidence *chatFrameEvidence) ([]byte, error) {
	if evidence == nil || filter.held == nil && (!keep || !evidence.toolCallFinish) {
		if !keep {
			return nil, nil
		}
		return frame, nil
	}
	if filter.held == nil {
		filter.held = &heldChatFinish{first: first}
	}
	if keep {
		filter.held.frames = append(filter.held.frames, frame)
	}
	if evidence.outputTokens == nil {
		return nil, nil
	}
	return filter.releaseHeld(*evidence.outputTokens >= filter.outputLimit)
}

// releaseHeld returns the held frames, with the finish frame reporting the
// output limit stop when atLimit.
func (filter *ChatUsageStreamFilter) releaseHeld(atLimit bool) ([]byte, error) {
	held := filter.held
	if held == nil {
		return nil, nil
	}
	filter.held = nil
	if atLimit {
		finish, err := chatFinishAtOutputLimit(held.frames[0], filter.framer.limit, held.first)
		if err != nil {
			return nil, err
		}
		held.frames[0] = finish
	}
	return bytes.Join(held.frames, nil), nil
}

// holdToolFrame buffers a frame whose choices stream tool-call arguments. The
// Router cannot know a call is complete until its finish arrives, so holding
// the fragments keeps an upstream that dies mid-call from feeding the client
// an arguments string no JSON decoder can read. A choice that already carries
// its finish finished as sent. It reports whether the frame was held.
func (filter *ChatUsageStreamFilter) holdToolFrame(frame []byte, choices []json.RawMessage) bool {
	held := false
	for _, raw := range choices {
		var choice struct {
			Index        *int    `json:"index"`
			FinishReason *string `json:"finish_reason"`
			Delta        struct {
				ToolCalls []chatFilterToolCallWire `json:"tool_calls"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(raw, &choice); err != nil ||
			choice.Index == nil || choice.FinishReason != nil || len(choice.Delta.ToolCalls) == 0 {
			continue
		}
		held = true
		for _, toolCall := range choice.Delta.ToolCalls {
			key := heldChatToolKey{choice: *choice.Index, tool: toolCall.Index}
			call := filter.heldCalls[key]
			if call == nil {
				call = &heldChatToolCall{}
				if filter.heldCalls == nil {
					filter.heldCalls = make(map[heldChatToolKey]*heldChatToolCall)
				}
				filter.heldCalls[key] = call
				filter.heldOrder = append(filter.heldOrder, key)
			}
			if call.id == "" && toolCall.ID != "" {
				call.id = toolCall.ID
			}
			if call.name == "" && toolCall.Function.Name != "" {
				call.name = toolCall.Function.Name
			}
			if call.callType == "" && toolCall.Type != "" {
				call.callType = toolCall.Type
			}
			call.arguments = append(call.arguments, toolCall.Function.Arguments...)
		}
	}
	if held {
		filter.heldFrames = append(filter.heldFrames, append([]byte(nil), frame...))
		if filter.heldFirst == nil {
			filter.heldFirst = append([]byte(nil), frame...)
		}
	}
	return held
}

// releaseHeldToolFrames forgets the held frames and returns them in their
// arrival order. The held frames carry no finish or usage, so sending them
// verbatim preserves the bytes the provider sent.
func (filter *ChatUsageStreamFilter) releaseHeldToolFrames() [][]byte {
	frames := filter.heldFrames
	filter.heldFrames = nil
	filter.heldFirst = nil
	filter.heldCalls = nil
	filter.heldOrder = nil
	filter.heldAtReadStart = 0
	return frames
}

// ReleaseHeldForAbort returns the tool-argument frames held before the read
// that failed the stream, verbatim, and drops the rest. A stream failure
// switches the client to the codec's frames, which carry only the failing
// read's output, so the prefix earlier reads held back — the accepted part of
// a call the provider later broke — must ride along or the client loses bytes
// it would have received read by read. The frames the failing read held stay
// dropped: the failure names them broken, as the pass-through filter's own
// output was dropped on this path before the hold existed.
func (filter *ChatUsageStreamFilter) ReleaseHeldForAbort() []byte {
	if filter == nil {
		return nil
	}
	boundary := filter.heldAtReadStart
	var prefix [][]byte
	if boundary > 0 && boundary <= len(filter.heldFrames) {
		prefix = filter.heldFrames[:boundary]
	}
	filter.releaseHeldToolFrames()
	return bytes.Join(prefix, nil)
}

// repairHeldToolCalls closes the held calls when the stream ended without a
// finish: an upstream that died mid-call leaves arguments that stop inside a
// JSON object, and the members the model finished still close into a valid
// call. The synthesized frames carry the complete arguments, the limit stop
// the output limit's own cut uses and the [DONE] sentinel, so the client keeps
// a call it can act on instead of a broken one. Anything the close cannot
// finish releases verbatim instead.
func (filter *ChatUsageStreamFilter) repairHeldToolCalls() ([]byte, bool) {
	if filter.upstreamFailed || len(filter.pendingTerminal) != 0 || len(filter.heldCalls) == 0 {
		return nil, false
	}
	envelope, ok := filter.heldCallEnvelope()
	if !ok {
		return nil, false
	}
	type repairedCall struct {
		choice   int
		tool     int
		id       string
		name     string
		callType string
		members  string
	}
	calls := make([]repairedCall, 0, len(filter.heldOrder))
	for _, key := range filter.heldOrder {
		call := filter.heldCalls[key]
		if call.id == "" || call.name == "" {
			return nil, false
		}
		members, ok := llmprotocol.TruncatedJSONObject(string(call.arguments), filter.maximumDepth)
		if !ok {
			return nil, false
		}
		calls = append(calls, repairedCall{
			choice: key.choice, tool: key.tool, id: call.id, name: call.name,
			callType: call.callType, members: members,
		})
	}
	toolCallsByChoice := make(map[int][]map[string]any, len(calls))
	choiceOrder := make([]int, 0, len(calls))
	for _, call := range calls {
		if _, seen := toolCallsByChoice[call.choice]; !seen {
			choiceOrder = append(choiceOrder, call.choice)
		}
		callType := call.callType
		if callType == "" {
			callType = "function"
		}
		toolCallsByChoice[call.choice] = append(toolCallsByChoice[call.choice], map[string]any{
			"index": call.tool,
			"id":    call.id,
			"type":  callType,
			"function": map[string]any{
				"name":      call.name,
				"arguments": call.members,
			},
		})
	}
	deltaChoices := make([]map[string]any, 0, len(choiceOrder))
	finishChoices := make([]map[string]any, 0, len(choiceOrder))
	for _, choice := range choiceOrder {
		deltaChoices = append(deltaChoices, map[string]any{
			"index":         choice,
			"delta":         map[string]any{"tool_calls": toolCallsByChoice[choice]},
			"finish_reason": nil,
		})
		finishChoices = append(finishChoices, map[string]any{
			"index":         choice,
			"delta":         map[string]any{},
			"finish_reason": "length",
		})
	}
	deltaEncoded, err := json.Marshal(deltaChoices)
	if err != nil {
		return nil, false
	}
	finishEncoded, err := json.Marshal(finishChoices)
	if err != nil {
		return nil, false
	}
	envelope["choices"] = deltaEncoded
	deltaFrame, err := encodeSSE("", envelope)
	if err != nil {
		return nil, false
	}
	envelope["choices"] = finishEncoded
	finishFrame, err := encodeSSE("", envelope)
	if err != nil {
		return nil, false
	}
	filter.releaseHeldToolFrames()
	return append(append(deltaFrame, finishFrame...), []byte("data: [DONE]\n\n")...), true
}

// heldCallEnvelope decodes the first held frame's envelope, so the synthesized
// frames name the same response, model and creation the provider sent.
func (filter *ChatUsageStreamFilter) heldCallEnvelope() (map[string]json.RawMessage, bool) {
	parsed, err := parseSSEFrameAtPosition(filter.heldFirst, filter.framer.limit, false)
	if err != nil || !parsed.HasData {
		return nil, false
	}
	var envelope map[string]json.RawMessage
	if err := decodeProviderWire(parsed.Data, &envelope, llmprotocol.DefaultPolicy()); err != nil {
		return nil, false
	}
	delete(envelope, "choices")
	return envelope, true
}

func (filter *ChatUsageStreamFilter) poison(err error) error {
	if err != nil && filter.failure == nil {
		filter.failure = err
	}
	filter.pendingTerminal = nil
	filter.held = nil
	filter.releaseHeldToolFrames()
	return filter.failure
}

// chatFinishAtOutputLimit rewrites the tool-call finish of choice 0 in a
// frame as the output limit stop.
func chatFinishAtOutputLimit(frame []byte, limit int, first bool) ([]byte, error) {
	parsed, err := parseSSEFrameAtPosition(frame, limit, first)
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err = json.Unmarshal(parsed.Data, &object); err != nil {
		return nil, err
	}
	var choices []map[string]json.RawMessage
	if err = json.Unmarshal(object["choices"], &choices); err != nil {
		return nil, err
	}
	stop, err := json.Marshal(encodeChatStop(llmprotocol.StopMaxTokens))
	if err != nil {
		return nil, err
	}
	for _, choice := range choices {
		var index int
		var reason string
		if json.Unmarshal(choice["index"], &index) == nil && index == 0 &&
			json.Unmarshal(choice["finish_reason"], &reason) == nil && decodeChatStop(reason) == llmprotocol.StopToolCall {
			choice["finish_reason"] = stop
		}
	}
	if object["choices"], err = json.Marshal(choices); err != nil {
		return nil, err
	}
	return encodeSSE(parsed.Event, object)
}

// chatChoicesCarryFinish reports whether any choice in the frame declares a
// finish reason: the moment the call's fate is decided, the held fragments go
// out verbatim with it.
func chatChoicesCarryFinish(choices []json.RawMessage) bool {
	for _, raw := range choices {
		var choice struct {
			FinishReason *string `json:"finish_reason"`
		}
		if json.Unmarshal(raw, &choice) == nil && choice.FinishReason != nil {
			return true
		}
	}
	return false
}

// chatChoicesEndForToolCalls reports choices that end choice 0 for its tool
// calls.
func chatChoicesEndForToolCalls(choices []json.RawMessage) bool {
	for _, raw := range choices {
		var choice struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason"`
		}
		if json.Unmarshal(raw, &choice) == nil && choice.Index == 0 && choice.FinishReason != nil &&
			decodeChatStop(*choice.FinishReason) == llmprotocol.StopToolCall {
			return true
		}
	}
	return false
}

func chatUsageCompletionTokens(raw json.RawMessage) *int64 {
	var usage map[string]json.RawMessage
	if json.Unmarshal(raw, &usage) != nil {
		return nil
	}
	// Match Chat provider decoding: a present canonical counter takes precedence.
	counter, present := usage["completion_tokens"]
	if !present {
		counter = usage["output_tokens"]
	}
	var tokens *int64
	if counter == nil || json.Unmarshal(counter, &tokens) != nil {
		return nil
	}
	return tokens
}

func filterChatUsageFrame(
	frame []byte, limit int, first, stripUsage bool, seenFinish map[int]bool, evidence *chatFrameEvidence,
) ([]byte, bool, bool, bool, []json.RawMessage, error) {
	parsed, err := parseSSEFrameAtPosition(frame, limit, first)
	if err != nil {
		return nil, false, false, false, nil, err
	}
	if !parsed.HasData {
		return frame, true, false, false, nil, nil
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		return frame, true, true, true, nil, nil
	}
	var object map[string]json.RawMessage
	if err := decodeProviderWire(parsed.Data, &object, llmprotocol.DefaultPolicy()); err != nil {
		return nil, false, true, false, nil, err
	}
	if raw, failed := object["error"]; failed && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, false, true, false, nil, errChatUpstreamErrorFrame
	}
	changed, err := deleteChatServerMembers(object)
	if err != nil {
		return nil, false, true, false, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_stream", "Chat stream choices are invalid", err)
	}
	if _, present := object["provider"]; present {
		delete(object, "provider")
		changed = true
	}
	var choices []json.RawMessage
	if rawChoices, exists := object["choices"]; exists && !bytes.Equal(bytes.TrimSpace(rawChoices), []byte("null")) {
		if err := json.Unmarshal(rawChoices, &choices); err != nil {
			return nil, false, true, false, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_stream", "Chat stream choices are invalid", err)
		}
	}
	for index, rawChoice := range choices {
		choice, removed, choiceErr := removeChatJSONFields(rawChoice, "native_finish_reason")
		if choiceErr != nil {
			return nil, false, true, false, nil, choiceErr
		}
		if removed {
			choices[index] = choice
			changed = true
		}
	}
	var removedRepeatedFinish bool
	choices, removedRepeatedFinish, err = removeRepeatedChatFinishChoices(choices, seenFinish)
	if err != nil {
		return nil, false, true, false, nil, err
	}
	changed = changed || removedRepeatedFinish
	if changed && (len(choices) > 0 || removedRepeatedFinish) {
		encoded, marshalErr := json.Marshal(choices)
		if marshalErr != nil {
			return nil, false, true, false, nil, marshalErr
		}
		object["choices"] = encoded
	}
	usage, hasUsage := object["usage"]
	if evidence != nil {
		evidence.toolCallFinish = chatChoicesEndForToolCalls(choices)
		if hasUsage {
			evidence.outputTokens = chatUsageCompletionTokens(usage)
		}
	}
	if removedRepeatedFinish && len(choices) == 0 && (!hasUsage || bytes.Equal(bytes.TrimSpace(usage), []byte("null"))) {
		return nil, false, true, false, choices, nil
	}
	if hasUsage && stripUsage {
		if len(choices) == 0 && !bytes.Equal(bytes.TrimSpace(usage), []byte("null")) {
			return nil, false, true, false, choices, nil
		}
		delete(object, "usage")
		changed = true
	} else if hasUsage && !bytes.Equal(bytes.TrimSpace(usage), []byte("null")) {
		usage, changed, err = stripOpenRouterUsageDecorations(usage, changed)
		if err != nil {
			return nil, false, true, false, choices, err
		}
		object["usage"] = usage
	} else if !hasUsage && !stripUsage {
		if !changed {
			frame, err = encodeNullUsageFrame(parsed.Event, parsed.Data)
			return frame, err == nil, true, false, choices, err
		}
		object["usage"] = json.RawMessage("null")
	}
	if !changed {
		return frame, true, true, false, choices, nil
	}
	filtered, err := encodeSSE(parsed.Event, object)
	return filtered, err == nil, true, false, choices, err
}

// encodeNullUsageFrame gives a chunk without usage the "usage": null that
// OpenAI sends on every chunk before the usage chunk of a stream whose client
// asked for usage, leaving the provider's members as they were.
func encodeNullUsageFrame(event string, data []byte) ([]byte, error) {
	var chunk bytes.Buffer
	if err := json.Compact(&chunk, data); err != nil {
		return nil, err
	}
	body := chunk.Bytes()
	separator := ","
	if len(body) == len("{}") {
		separator = ""
	}
	var frame bytes.Buffer
	if event != "" {
		frame.WriteString("event: " + event + "\n")
	}
	frame.WriteString("data: ")
	frame.Write(body[:len(body)-1])
	frame.WriteString(separator + `"usage":null}` + "\n\n")
	return frame.Bytes(), nil
}

// Some providers send an extra content-free finish chunk after an earlier
// finish for the same choice. Keep the first terminal, and retain a repeated
// chunk's usage as a usage-only frame if the client requested accounting.
func removeRepeatedChatFinishChoices(choices []json.RawMessage, seen map[int]bool) ([]json.RawMessage, bool, error) {
	if len(choices) == 0 {
		return choices, false, nil
	}
	kept := make([]json.RawMessage, 0, len(choices))
	removed := false
	for _, raw := range choices {
		var choice struct {
			Index        *int            `json:"index"`
			FinishReason *string         `json:"finish_reason"`
			Delta        json.RawMessage `json:"delta"`
		}
		if err := json.Unmarshal(raw, &choice); err != nil {
			return nil, false, err
		}
		if choice.Index == nil || choice.FinishReason == nil {
			kept = append(kept, raw)
			continue
		}
		index := *choice.Index
		if seen[index] && chatFinishChoiceIsContentFree(raw, choice.Delta) {
			removed = true
			continue
		}
		seen[index] = true
		kept = append(kept, raw)
	}
	return kept, removed, nil
}

func chatFinishChoiceIsContentFree(rawChoice, rawDelta json.RawMessage) bool {
	var choice map[string]json.RawMessage
	if err := json.Unmarshal(rawChoice, &choice); err != nil {
		return false
	}
	for field, value := range choice {
		if field != "index" && field != "finish_reason" && field != "delta" &&
			(field != "logprobs" || !bytes.Equal(bytes.TrimSpace(value), []byte("null"))) {
			return false
		}
	}
	if len(rawDelta) == 0 || bytes.Equal(bytes.TrimSpace(rawDelta), []byte("null")) {
		return true
	}
	var delta map[string]json.RawMessage
	if err := json.Unmarshal(rawDelta, &delta); err != nil {
		return false
	}
	for field, value := range delta {
		if field != "content" ||
			!bytes.Equal(bytes.TrimSpace(value), []byte(`""`)) &&
				!bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return false
		}
	}
	return true
}

func removeChatJSONFields(raw json.RawMessage, fields ...string) (json.RawMessage, bool, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, false, err
	}
	changed := false
	for _, field := range fields {
		if _, present := object[field]; present {
			delete(object, field)
			changed = true
		}
	}
	if !changed {
		return raw, false, nil
	}
	encoded, err := json.Marshal(object)
	return encoded, true, err
}

func stripOpenRouterUsageDecorations(raw json.RawMessage, changed bool) (json.RawMessage, bool, error) {
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw, &usage); err != nil {
		return nil, false, err
	}
	for _, field := range []string{"cost", "is_byok", "cost_details", "server_tool_use"} {
		if _, present := usage[field]; present {
			delete(usage, field)
			changed = true
		}
	}
	for _, detail := range []struct{ field, nested string }{
		{"prompt_tokens_details", "video_tokens"},
		{"completion_tokens_details", "image_tokens"},
	} {
		value, present := usage[detail.field]
		if !present || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			continue
		}
		filtered, removed, err := removeChatJSONFields(value, detail.nested)
		if err != nil {
			return nil, false, err
		}
		if removed {
			usage[detail.field] = filtered
			changed = true
		}
	}
	if !changed {
		return raw, false, nil
	}
	encoded, err := json.Marshal(usage)
	return encoded, true, err
}
