package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An upstream stream can end mid tool call without a limit signal: no finish
// reason, no [DONE], and arguments that stop inside their JSON object. The
// codec closes the call with the members the model finished instead of failing
// the stream, and the public filter holds the fragments so the client never
// receives an arguments string no JSON decoder can read.

const cutArguments = `{"command":"echo hi","timeout":30`

// cutMembers is the close the repair gives cutArguments: the value the cut
// ended inside is left out, as TruncatedJSONObject's contract defines.
const cutMembers = `{"command":"echo hi"}`

func chatToolDeltaFrame(arguments string, identity bool) []byte {
	function := map[string]any{"arguments": arguments}
	delta := map[string]any{}
	if identity {
		function["name"] = "shell"
		delta["role"] = "assistant"
	}
	call := map[string]any{"index": 0, "function": function}
	if identity {
		call["id"] = "call_1"
		call["type"] = "function"
	}
	delta["tool_calls"] = []map[string]any{call}
	frame, err := json.Marshal(map[string]any{
		"id":      "response_1",
		"model":   "source-model",
		"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": nil}},
	})
	if err != nil {
		panic(err)
	}
	return []byte("data: " + string(frame) + "\n\n")
}

func pushCutToolCallStream(t *testing.T, arguments string, target llmprotocol.WireFormat) []byte {
	t.Helper()
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "source-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = stream.Push(chatToolDeltaFrame(arguments, true))
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Join(frames, nil)
}

func assertRepairDiagnostic(t *testing.T, diagnostics llmprotocol.Diagnostics) {
	t.Helper()
	if len(diagnostics) != 1 || diagnostics[0].Field != "tool_call.arguments" ||
		diagnostics[0].Action != llmprotocol.DiagnosticApproximated || diagnostics[0].Reason == "" {
		t.Fatalf("missing repair diagnostic: %+v", diagnostics)
	}
}

func startCutToolCall(t *testing.T, state *streamState, arguments string) {
	t.Helper()
	startTestStream(t, state)
	start := llmprotocol.ToolCall{ID: "call_1", Name: "shell"}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1",
		ToolCall: &start,
	}); err != nil {
		t.Fatal(err)
	}
	delta := llmprotocol.ToolCall{Arguments: arguments}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventToolCallDelta, ItemIndex: 0, ToolCall: &delta,
	}); err != nil {
		t.Fatal(err)
	}
}

func TestStreamStateFinalizeRepairsACutToolCall(t *testing.T) {
	state := newTestStreamState()
	startCutToolCall(t, state, cutArguments)
	events, diagnostics, err := state.finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 ||
		events[0].Type != llmprotocol.EventOutputItemCompleted ||
		events[0].ToolCall == nil || events[0].ToolCall.Arguments != cutMembers ||
		events[1].Type != llmprotocol.EventResponseCompleted || events[1].StopReason != llmprotocol.StopMaxTokens {
		t.Fatalf("repaired terminal events = %+v", events)
	}
	assertRepairDiagnostic(t, diagnostics)
}

func TestStreamStateFinalizeKeepsFailureForUnrepairableArguments(t *testing.T) {
	state := newTestStreamState()
	startCutToolCall(t, state, "not json")
	events, diagnostics, err := state.finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
		events[0].Error == nil || events[0].Error.Code != "stream_incomplete" {
		t.Fatalf("unrepairable arguments terminal events = %+v", events)
	}
	if len(diagnostics) != 0 {
		t.Fatalf("a failure stream reported repair diagnostics: %+v", diagnostics)
	}
}

func TestStreamStateFinalizeKeepsFailureForSeveralPendingToolCalls(t *testing.T) {
	state := newTestStreamState()
	startCutToolCall(t, state, cutArguments)
	second := llmprotocol.ToolCall{ID: "call_2", Name: "shell"}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 1, ItemID: "item_2",
		ToolCall: &second,
	}); err != nil {
		t.Fatal(err)
	}
	events, diagnostics, err := state.finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
		events[0].Error == nil || events[0].Error.Code != "stream_incomplete" {
		t.Fatalf("several pending calls terminal events = %+v", events)
	}
	if len(diagnostics) != 0 {
		t.Fatalf("a failure stream reported repair diagnostics: %+v", diagnostics)
	}
}

func TestStreamStateFinalizeKeepsFailureWhenTheIdentityIsMissing(t *testing.T) {
	state := newTestStreamState()
	startTestStream(t, state)
	start := llmprotocol.ToolCall{ID: "call_1"}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1",
		ToolCall: &start,
	}); err != nil {
		t.Fatal(err)
	}
	events, diagnostics, err := state.finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
		events[0].Error == nil || events[0].Error.Code != "stream_incomplete" {
		t.Fatalf("missing identity terminal events = %+v", events)
	}
	if len(diagnostics) != 0 {
		t.Fatalf("a failure stream reported repair diagnostics: %+v", diagnostics)
	}
}

func TestResponsesStreamEndedMidToolCallClosesTheCall(t *testing.T) {
	wire := pushCutToolCallStream(t, cutArguments, llmprotocol.OpenAIResponsesV1)
	quoted, err := json.Marshal(cutMembers)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(wire, []byte(`"arguments":`+string(quoted))) {
		t.Fatalf("the closed arguments are missing: %s", wire)
	}
	if !bytes.Contains(wire, []byte(`"status":"incomplete"`)) {
		t.Fatalf("the repaired stream did not end with the limit stop: %s", wire)
	}
}

func TestChatStreamEndedMidToolCallStaysComplete(t *testing.T) {
	wire := pushCutToolCallStream(t, cutArguments, llmprotocol.OpenAIChatV1)
	if !bytes.Contains(wire, []byte(`"finish_reason":"length"`)) || !bytes.Contains(wire, []byte("data: [DONE]")) {
		t.Fatalf("the repaired stream did not end with the limit stop: %s", wire)
	}
	assertStreamHasNoErrorTerminal(t, wire)
}

func assertStreamHasNoErrorTerminal(t *testing.T, wire []byte) {
	t.Helper()
	if bytes.Contains(wire, []byte(`"error"`)) {
		t.Fatalf("the repaired stream also emitted a failure terminal: %s", wire)
	}
}

func TestChatStreamEndedWithoutAnObjectKeepsTheFailure(t *testing.T) {
	wire := pushCutToolCallStream(t, "not json", llmprotocol.OpenAIChatV1)
	assertStreamHasNoSuccessTerminal(t, wire)
	if !bytes.Contains(wire, []byte("stream_incomplete")) {
		t.Fatalf("the failure terminal is missing: %s", wire)
	}
}

func TestChatPublicStreamFilterReleasesHeldToolFramesUnchanged(t *testing.T) {
	first := chatToolDeltaFrame(`{"command":`, true)
	second := chatToolDeltaFrame(` "echo hi"}`, false)
	finish := []byte("data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
	done := []byte("data: [DONE]\n\n")
	filter := NewChatPublicStreamFilter(1<<20, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	var output []byte
	for _, chunk := range [][]byte{first, second, finish, done} {
		out, err := filter.Push(chunk)
		if err != nil {
			t.Fatal(err)
		}
		output = append(output, out...)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	output = append(output, final...)
	if want := bytes.Join([][]byte{first, second, finish, done}, nil); !bytes.Equal(output, want) {
		t.Fatalf("a complete tool stream changed: output=%q want=%q", output, want)
	}
}

func TestChatPublicStreamFilterKeepsTheHeldPrefixOnAStreamFailure(t *testing.T) {
	// A provider streams an accepted tool-call prefix in one read and breaks
	// the call in the next. The failure switches the client to the codec's
	// frames, so the prefix the filter held must ride along with them, while
	// the frame the failing read broke stays dropped.
	role := []byte("data: {\"id\":\"chatcmpl_cut\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
	prefix := chatToolDeltaFrame(`{"command":"echo hi"}`, true)
	broken := chatToolDeltaFrame(`{"x":1}`, false)
	filter := NewChatPublicStreamFilter(1<<20, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	if out, err := filter.Push(role); err != nil || len(out) == 0 {
		t.Fatalf("the role frame did not pass: output=%q err=%v", out, err)
	}
	if out, err := filter.Push(prefix); err != nil || len(out) != 0 {
		t.Fatalf("a held fragment escaped: output=%q err=%v", out, err)
	}
	if out, err := filter.Push(broken); err != nil || len(out) != 0 {
		t.Fatalf("a held fragment escaped: output=%q err=%v", out, err)
	}
	aborted := filter.ReleaseHeldForAbort()
	if !bytes.Equal(aborted, prefix) {
		t.Fatalf("the accepted prefix did not ride along on the abort: %q", aborted)
	}
	if bytes.Contains(aborted, []byte(`"x":1`)) {
		t.Fatalf("the failing read's fragment leaked on the abort: %q", aborted)
	}
	if out, err := filter.Push(chatToolDeltaFrame(`{"command":"more"}`, true)); err != nil || len(out) != 0 {
		t.Fatalf("held state survived the abort: output=%q err=%v", out, err)
	}
	if out, err := filter.Finalize(); err != nil {
		t.Fatalf("finalize after the abort failed: %v", err)
	} else if bytes.Contains(out, []byte(`{"command":"echo hi"}`)) {
		t.Fatalf("the released prefix came back on finalize: %q", out)
	}
}

func TestChatPublicStreamFilterKeepsTheToolHoldAcrossAUsageOnlyFrame(t *testing.T) {
	// With include_usage a provider can send a choices:[] usage frame while
	// the tool call is still pending. The frame says nothing about the call's
	// fate, so the hold survives it and a stream that ends afterwards still
	// reaches the repair instead of releasing the broken arguments early.
	filter := NewChatPublicStreamFilter(1<<20, true, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	if out, err := filter.Push(chatToolDeltaFrame(cutArguments, true)); err != nil || len(out) != 0 {
		t.Fatalf("a held fragment escaped: output=%q err=%v", out, err)
	}
	usageOnly := []byte("data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":8,\"total_tokens\":9}}\n\n")
	out, err := filter.Push(usageOnly)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, usageOnly) {
		t.Fatalf("the usage-only frame changed: output=%q want=%q", out, usageOnly)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	quoted, err := json.Marshal(cutMembers)
	if err != nil {
		t.Fatal(err)
	}
	broken, err := json.Marshal(cutArguments)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(final, []byte(`"arguments":`+string(quoted))) ||
		!bytes.Contains(final, []byte(`"finish_reason":"length"`)) ||
		!bytes.Contains(final, []byte("data: [DONE]")) {
		t.Fatalf("the usage-only frame did not keep the repair: %s", final)
	}
	if bytes.Contains(final, broken) {
		t.Fatalf("the broken arguments leaked: %s", final)
	}
}

func TestChatPublicStreamFilterKeepsTheToolHoldAcrossAnEmptyDelta(t *testing.T) {
	// An empty delta with no finish reason says nothing about the pending
	// call: the hold survives it, and a stream that ends afterwards still
	// reaches the repair instead of releasing the broken arguments early.
	filter := NewChatPublicStreamFilter(1<<20, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	if out, err := filter.Push(chatToolDeltaFrame(cutArguments, true)); err != nil || len(out) != 0 {
		t.Fatalf("a held fragment escaped: output=%q err=%v", out, err)
	}
	empty := []byte("data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}]}\n\n")
	out, err := filter.Push(empty)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, empty) {
		t.Fatalf("the empty delta changed: output=%q want=%q", out, empty)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	quoted, err := json.Marshal(cutMembers)
	if err != nil {
		t.Fatal(err)
	}
	broken, err := json.Marshal(cutArguments)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(final, []byte(`"arguments":`+string(quoted))) ||
		!bytes.Contains(final, []byte(`"finish_reason":"length"`)) ||
		!bytes.Contains(final, []byte("data: [DONE]")) {
		t.Fatalf("the empty delta did not keep the repair: %s", final)
	}
	if bytes.Contains(final, broken) {
		t.Fatalf("the broken arguments leaked: %s", final)
	}
}

func TestChatPublicStreamFilterRepairsAToolCallCutByStreamEnd(t *testing.T) {
	filter := NewChatPublicStreamFilter(1<<20, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	if out, err := filter.Push(chatToolDeltaFrame(cutArguments, true)); err != nil || len(out) != 0 {
		t.Fatalf("a held fragment escaped: output=%q err=%v", out, err)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	quoted, err := json.Marshal(cutMembers)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Count(final, []byte(`"arguments":`)) != 1 ||
		!bytes.Contains(final, []byte(`"arguments":`+string(quoted))) ||
		!bytes.Contains(final, []byte(`"finish_reason":"length"`)) ||
		!bytes.Contains(final, []byte(`"id":"response_1"`)) ||
		bytes.Count(final, []byte("data: [DONE]")) != 1 {
		t.Fatalf("the repaired stream is incomplete: %s", final)
	}
}

func TestChatPublicStreamFilterRepairsArgumentsCutInsideAString(t *testing.T) {
	filter := NewChatPublicStreamFilter(1<<20, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	if _, err := filter.Push(chatToolDeltaFrame(`{"command": "echo hi`, true)); err != nil {
		t.Fatal(err)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(final, []byte(`"arguments":"{}"`)) ||
		!bytes.Contains(final, []byte(`"finish_reason":"length"`)) ||
		!bytes.Contains(final, []byte("data: [DONE]")) {
		t.Fatalf("the value the cut ended inside was not dropped: %s", final)
	}
}

func TestChatPublicStreamFilterReleasesUnrepairableToolArgumentsVerbatim(t *testing.T) {
	fragment := chatToolDeltaFrame("not json", true)
	filter := NewChatPublicStreamFilter(1<<20, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	if out, err := filter.Push(fragment); err != nil || len(out) != 0 {
		t.Fatalf("a held fragment escaped: output=%q err=%v", out, err)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(final, fragment) || bytes.Contains(final, []byte("data: [DONE]")) {
		t.Fatalf("unrepairable arguments were not released verbatim: %q", final)
	}
}

func TestChatPublicStreamFilterReleasesHeldFramesWhenTheProviderFails(t *testing.T) {
	fragment := chatToolDeltaFrame(`{"command":`, true)
	event := []byte("data: {\"error\":{\"message\":\"failed\",\"type\":\"server_error\",\"code\":500}}\n\n")
	filter := NewChatPublicStreamFilter(1<<20, false, llmprotocol.DefaultPolicy().Limits.JSONDepth)
	if out, err := filter.Push(fragment); err != nil || len(out) != 0 {
		t.Fatalf("a held fragment escaped: output=%q err=%v", out, err)
	}
	out, err := filter.Push(event)
	if err != nil || !bytes.Equal(out, fragment) || !filter.UpstreamFailed() {
		t.Fatalf("the provider error did not release the held frames: out=%q err=%v failed=%v", out, err, filter.UpstreamFailed())
	}
	final, err := filter.Finalize()
	if err != nil || len(final) != 0 {
		t.Fatalf("a failed stream kept frames after its error: out=%q err=%v", final, err)
	}
}
