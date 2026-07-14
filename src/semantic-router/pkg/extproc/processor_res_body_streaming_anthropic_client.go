package extproc

import (
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// anthropicPingCadence is how long the emitter waits between
// surfacing keepalive pings. Matches the observed upstream cadence
// (~15s) and the Claude Code SDK's idle tolerance.
//
// Hardcoded in v1; can be made configurable in a follow-up if operators ask.
const anthropicPingCadence = 15 * time.Second

// handleAnthropicClientStreamingResponseBody dispatches streaming
// responses for clients that called the /v1/messages endpoint
// (ClientProtocol == "anthropic"). The body Envoy hands us may be in
// either Anthropic SSE (when the upstream is an Anthropic backend) or
// OpenAI SSE (when the upstream is an OpenAI backend); both arrive on
// this path because the inbound parser already pinned the client to
// the Anthropic protocol. The function:
//
//  1. translates Anthropic upstream SSE to OpenAI SSE first, so the
//     accounting accumulator, plugins, and the outbound emitter all
//     see the uniform OpenAI shape vsr's pipeline expects;
//  2. runs the standard OpenAI streaming accumulator to update the
//     per-request StreamingMetadata / usage (and flag HasStreamingChunks)
//     so metrics, cache, and replay continue to work;
//  3. re-emits the OpenAI chunk as Anthropic SSE via
//     anthropic.EmitAnthropicSSEChunk and returns those bytes via
//     BodyMutation so the client sees Anthropic-shape events.
//
// This is the symmetric counterpart to
// handleAnthropicStreamingResponseBody (which handles OpenAI clients
// hitting an Anthropic backend); the two share the AnthropicStream
// state but only one runs per request because ClientProtocol is pinned
// at the inbound parser.
func (r *OpenAIRouter) handleAnthropicClientStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	recordStreamingTTFT(ctx)
	ensureStreamingState(ctx)
	if ctx.AnthropicStream == nil {
		ctx.AnthropicStream = anthropic.NewStreamState()
	}

	// Envoy STREAMED mode delivers the response body split at arbitrary
	// byte offsets, so an upstream SSE frame may straddle two chunks.
	// Reassemble complete frames before translation and hold any trailing
	// partial frame for the next chunk. Without this, a split frame fails
	// to parse, the translated output is empty, and — because this handler
	// suppresses the raw upstream bytes to keep ownership of the wire (see
	// emptyAnthropicBodyMutation) — the frame vanishes entirely, hanging
	// the client (issue #2316). The keepalive ping still flows below even
	// when only a partial frame arrived, so a long split does not stall the
	// connection.
	frames, remainder := reassembleSSEFrames(ctx.PendingSSEBytes, responseBody)
	ctx.PendingSSEBytes = remainder

	// Inject a keepalive ping if the gap since the last chunk crossed
	// the cadence threshold. Anthropic clients (including the SDK
	// MessageStream accumulator) treat pings as no-ops; the value is
	// keeping middleboxes from severing the long-lived connection
	// during sparse-token windows. The ping is prepended below so it
	// reaches the client before this chunk's content events.
	pingBytes := maybeEmitPing(ctx.AnthropicStream)

	openAIBytes, err := r.translateUpstreamToOpenAI(frames, ctx)
	if err != nil {
		logging.Errorf("Failed to translate upstream streaming chunk to OpenAI: %s", safeErrorForLog(err))
		return emptyAnthropicBodyMutation()
	}

	chunkStr := string(openAIBytes)
	if chunkStr != "" {
		ctx.HasStreamingChunks = true
		r.parseStreamingChunk(chunkStr, ctx)
	}

	anthropicBytes, streamDone, err := anthropic.EmitAnthropicSSEChunk(
		openAIBytes, ctx.AnthropicStream, ctx.IRExtensions, ctx.RequestModel,
	)
	if err != nil {
		logging.Errorf("Failed to emit Anthropic SSE chunk: %s", safeErrorForLog(err))
		return emptyAnthropicBodyMutation()
	}

	if streamDone || strings.Contains(chunkStr, "data: [DONE]") {
		r.finalizeStreamingResponse(ctx)
	}

	// Stamp the chunk-arrival time so the next pass can compute the
	// silence gap for ping injection. Done after the emitter so a
	// chunk that produced no bytes still extends the keepalive window
	// (otherwise zero-emit chunks would fire a ping on every call).
	ctx.AnthropicStream.LastChunkAt = time.Now()

	combined := make([]byte, 0, len(pingBytes)+len(anthropicBytes))
	combined = append(combined, pingBytes...)
	combined = append(combined, anthropicBytes...)
	return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: combined},
	}, nil)
}

// emptyAnthropicBodyMutation returns a non-nil BodyMutation carrying an
// empty body. Once execution enters the Anthropic-client streaming
// handler, vsr owns the wire: every return must replace the body so
// Envoy never forwards the raw upstream chunk. A nil BodyMutation would
// make Envoy pass the upstream bytes through unmodified, leaking
// Anthropic-shape backend frames on top of the events we synthesize —
// most visibly a second message_stop. Error and zero-emission paths use
// this so that invariant holds even when we have no content to send.
func emptyAnthropicBodyMutation() *ext_proc.ProcessingResponse {
	return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: []byte{}},
	}, nil)
}

// maybeEmitPing returns ping event bytes when the gap since the last
// chunk arrival crossed anthropicPingCadence. Returns nil otherwise.
// A zero LastChunkAt (first chunk after stream init) suppresses the
// ping — the spec only asks for pings during sparse-token windows, not
// for synthetic pings before any content has flowed.
func maybeEmitPing(state *anthropic.StreamState) []byte {
	if state == nil || state.LastChunkAt.IsZero() {
		return nil
	}
	if time.Since(state.LastChunkAt) < anthropicPingCadence {
		return nil
	}
	return anthropic.EmitAnthropicPingEvent()
}

// translateUpstreamToOpenAI normalizes the upstream SSE bytes into the
// OpenAI chunk shape. Anthropic upstreams go through
// TransformSSEChunkToOpenAI so plugins and the emitter see uniform IR;
// OpenAI upstreams pass through unchanged. Returns the normalized
// bytes and any translator error.
func (r *OpenAIRouter) translateUpstreamToOpenAI(
	responseBody []byte,
	ctx *RequestContext,
) ([]byte, error) {
	if ctx.APIFormat != config.APIFormatAnthropic {
		return responseBody, nil
	}
	// streamDone is intentionally discarded here; finalization is driven by
	// the outbound emitter's own streamDone return in
	// handleAnthropicClientStreamingResponseBody, not by the inbound translator.
	translated, _, err := anthropic.TransformSSEChunkToOpenAI(
		responseBody, ctx.AnthropicStream, ctx.RequestModel, ctx.IRExtensions,
	)
	if err != nil {
		return nil, err
	}
	return translated, nil
}
