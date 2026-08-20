package anthropic

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// --- test fixtures -----------------------------------------------------
//
// These mirror real Anthropic SSE payloads. Anthropic's API omits
// cache_creation_input_tokens / cache_read_input_tokens on message_delta.usage
// when there's nothing new to report, which (pre-fix) unmarshalled as an
// explicit zero and stomped the good values captured on message_start.

const sseMessageStartWithCache = "event: message_start\n" +
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test123\",\"type\":\"message\"," +
	"\"role\":\"assistant\",\"model\":\"claude-3-5-sonnet-20241022\",\"content\":[]," +
	"\"stop_reason\":null,\"stop_sequence\":null," +
	"\"usage\":{\"input_tokens\":25,\"cache_creation_input_tokens\":200," +
	"\"cache_read_input_tokens\":1800,\"output_tokens\":1}}}\n\n"

const sseMessageStartNoCache = "event: message_start\n" +
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test456\",\"type\":\"message\"," +
	"\"role\":\"assistant\",\"model\":\"claude-3-5-sonnet-20241022\",\"content\":[]," +
	"\"stop_reason\":null,\"stop_sequence\":null," +
	"\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n"

// message_delta as Anthropic actually sends it when there is no new cache
// activity to report: the cache fields are simply absent, which unmarshals
// onto anthropic.MessageDeltaUsage as zero values.
const sseMessageDeltaZeroCache = "event: message_delta\n" +
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"," +
	"\"stop_sequence\":null},\"usage\":{\"output_tokens\":15}}\n\n"

// message_delta reporting a larger cache_read figure, simulating a
// multi-turn / tool-loop request where the cache grew after message_start.
const sseMessageDeltaLargerCache = "event: message_delta\n" +
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"," +
	"\"stop_sequence\":null},\"usage\":{\"output_tokens\":15," +
	"\"cache_creation_input_tokens\":200,\"cache_read_input_tokens\":2500}}\n\n"

const sseContentBlockStartText = "event: content_block_start\n" +
	"data: {\"type\":\"content_block_start\",\"index\":0," +
	"\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"

const sseContentBlockDeltaText = "event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0," +
	"\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n"

const sseContentBlockStop = "event: content_block_stop\n" +
	"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"

const sseMessageStop = "event: message_stop\n" +
	"data: {\"type\":\"message_stop\"}\n\n"

// --- tests ---------------------------------------------------------------

// TestHandleMessageStart_CapturesCacheUsage covers Test Plan item:
// "confirm terminal message_delta.usage keeps the same cache counters as
// message_start" — first half: message_start must land the counters on ext.
func TestHandleMessageStart_CapturesCacheUsage(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}

	_, done, err := TransformSSEChunkToOpenAI([]byte(sseMessageStartWithCache), state, "claude-3-5-sonnet-20241022", ext)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if done {
		t.Fatalf("message_start should not signal stream completion")
	}

	if ext.CacheReadInputTokens != 1800 {
		t.Errorf("CacheReadInputTokens = %d, want 1800", ext.CacheReadInputTokens)
	}
	if ext.CacheCreationInputTokens != 200 {
		t.Errorf("CacheCreationInputTokens = %d, want 200", ext.CacheCreationInputTokens)
	}
}

// TestHandleMessageStart_NoCacheUsage ensures a request with no prompt
// caching in play leaves ext at zero (no false-positive cache reporting).
func TestHandleMessageStart_NoCacheUsage(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}

	if _, _, err := TransformSSEChunkToOpenAI([]byte(sseMessageStartNoCache), state, "claude-3-5-sonnet-20241022", ext); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if ext.CacheReadInputTokens != 0 {
		t.Errorf("CacheReadInputTokens = %d, want 0", ext.CacheReadInputTokens)
	}
	if ext.CacheCreationInputTokens != 0 {
		t.Errorf("CacheCreationInputTokens = %d, want 0", ext.CacheCreationInputTokens)
	}
}

// TestHandleMessageStart_NilExt guards against a nil-pointer panic when ext
// is nil (the existing OpenAI-client cell where cache counters have no
// downstream consumer).
func TestHandleMessageStart_NilExt(t *testing.T) {
	state := NewStreamState()

	if _, _, err := TransformSSEChunkToOpenAI([]byte(sseMessageStartWithCache), state, "claude-3-5-sonnet-20241022", nil); err != nil {
		t.Fatalf("unexpected error with nil ext: %v", err)
	}
}

// TestHandleMessageDelta_PreservesCacheUsage is the core regression test for
// issue #2947: a terminal message_delta that reports zero cache counters
// (because Anthropic omitted the fields) must NOT stomp the non-zero values
// already captured from message_start.
func TestHandleMessageDelta_PreservesCacheUsage(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}

	if _, _, err := TransformSSEChunkToOpenAI([]byte(sseMessageStartWithCache), state, "claude-3-5-sonnet-20241022", ext); err != nil {
		t.Fatalf("message_start: unexpected error: %v", err)
	}

	_, done, err := TransformSSEChunkToOpenAI([]byte(sseMessageDeltaZeroCache), state, "claude-3-5-sonnet-20241022", ext)
	if err != nil {
		t.Fatalf("message_delta: unexpected error: %v", err)
	}
	if done {
		t.Fatalf("message_delta alone should not signal stream completion")
	}

	if ext.CacheReadInputTokens != 1800 {
		t.Errorf("CacheReadInputTokens after message_delta = %d, want 1800 (preserved from message_start)", ext.CacheReadInputTokens)
	}
	if ext.CacheCreationInputTokens != 200 {
		t.Errorf("CacheCreationInputTokens after message_delta = %d, want 200 (preserved from message_start)", ext.CacheCreationInputTokens)
	}
}

// TestHandleMessageDelta_UpdatesWithLargerCacheUsage ensures message_delta
// is still allowed to *update* ext when it reports a genuine non-zero value
// (e.g. cache grew during a tool loop) — the fix must not freeze the
// counters permanently at whatever message_start reported.
func TestHandleMessageDelta_UpdatesWithLargerCacheUsage(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}

	if _, _, err := TransformSSEChunkToOpenAI([]byte(sseMessageStartWithCache), state, "claude-3-5-sonnet-20241022", ext); err != nil {
		t.Fatalf("message_start: unexpected error: %v", err)
	}
	if _, _, err := TransformSSEChunkToOpenAI([]byte(sseMessageDeltaLargerCache), state, "claude-3-5-sonnet-20241022", ext); err != nil {
		t.Fatalf("message_delta: unexpected error: %v", err)
	}

	if ext.CacheReadInputTokens != 2500 {
		t.Errorf("CacheReadInputTokens = %d, want 2500 (updated by message_delta)", ext.CacheReadInputTokens)
	}
	if ext.CacheCreationInputTokens != 200 {
		t.Errorf("CacheCreationInputTokens = %d, want 200", ext.CacheCreationInputTokens)
	}
}

// TestHandleMessageDelta_NilExt guards against a nil-pointer panic on the
// message_delta path when ext is nil.
func TestHandleMessageDelta_NilExt(t *testing.T) {
	state := NewStreamState()

	if _, _, err := TransformSSEChunkToOpenAI([]byte(sseMessageStartWithCache), state, "claude-3-5-sonnet-20241022", nil); err != nil {
		t.Fatalf("message_start: unexpected error: %v", err)
	}
	if _, _, err := TransformSSEChunkToOpenAI([]byte(sseMessageDeltaZeroCache), state, "claude-3-5-sonnet-20241022", nil); err != nil {
		t.Fatalf("message_delta: unexpected error with nil ext: %v", err)
	}
}

// TestTransformSSEChunkToOpenAI_FullStream_PreservesCacheUsage is an
// end-to-end reproduction of Test Plan item: "stream a cached Anthropic
// request through the router and confirm terminal message_delta.usage keeps
// the same cache counters as message_start." It runs the full event
// sequence a real Anthropic stream would send.
func TestTransformSSEChunkToOpenAI_FullStream_PreservesCacheUsage(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}
	model := "claude-3-5-sonnet-20241022"

	events := []string{
		sseMessageStartWithCache,
		sseContentBlockStartText,
		sseContentBlockDeltaText,
		sseContentBlockStop,
		sseMessageDeltaZeroCache,
		sseMessageStop,
	}

	var streamDone bool
	for _, e := range events {
		_, done, err := TransformSSEChunkToOpenAI([]byte(e), state, model, ext)
		if err != nil {
			t.Fatalf("unexpected error processing event %q: %v", e, err)
		}
		if done {
			streamDone = true
		}
	}

	if !streamDone {
		t.Fatalf("expected message_stop to signal stream completion")
	}
	if ext.CacheReadInputTokens != 1800 {
		t.Errorf("final CacheReadInputTokens = %d, want 1800", ext.CacheReadInputTokens)
	}
	if ext.CacheCreationInputTokens != 200 {
		t.Errorf("final CacheCreationInputTokens = %d, want 200", ext.CacheCreationInputTokens)
	}
}

// TestTransformSSEChunkToOpenAI_MultiChunkStream verifies the fix also holds
// when message_start and message_delta arrive in separate SSE reads
// (Envoy body chunks don't necessarily align with SSE event boundaries),
// which is the realistic case the streaming translate layer has to handle.
func TestTransformSSEChunkToOpenAI_MultiChunkStream(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}
	model := "claude-3-5-sonnet-20241022"

	chunk1 := []byte(sseMessageStartWithCache + sseContentBlockStartText)
	chunk2 := []byte(sseContentBlockDeltaText + sseContentBlockStop)
	chunk3 := []byte(sseMessageDeltaZeroCache + sseMessageStop)

	for _, c := range [][]byte{chunk1, chunk2, chunk3} {
		if _, _, err := TransformSSEChunkToOpenAI(c, state, model, ext); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if ext.CacheReadInputTokens != 1800 {
		t.Errorf("CacheReadInputTokens = %d, want 1800", ext.CacheReadInputTokens)
	}
	if ext.CacheCreationInputTokens != 200 {
		t.Errorf("CacheCreationInputTokens = %d, want 200", ext.CacheCreationInputTokens)
	}
}
