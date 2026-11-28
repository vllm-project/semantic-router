package extproc

import (
	"bytes"
	"errors"
	"time"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// responseGuardHoldLimit bounds the upstream bytes of a streamed answer the
// Router holds for its guard: the size the protocol codec accepts for any
// provider body, and that the ExtProc server admits as one message.
var responseGuardHoldLimit = llmprotocol.DefaultPolicy().Limits.BodyBytes

// What became of a held stream the guard could not read: it ran past
// responseGuardHoldLimit, or it ended without a complete answer.
const (
	responseGuardOverLimit  = "over_limit"
	responseGuardIncomplete = "incomplete"
)

var errResponseGuardHoldLimit = errors.New("streamed answer exceeds the response guard's hold limit")

// responseGuardHold is a streamed answer the Router holds until the selected
// decision's response_guard has read all of it, or until the structured
// output pass knows whether the answer is the turn's: the client frames the
// stream would have sent so far and the upstream bytes it read.
type responseGuardHold struct {
	frames []byte
	read   int
	// outcome is what became of the answer once the hold ended: the guard's
	// outcome, responseGuardOverLimit or responseGuardIncomplete.
	outcome string
	// guarded reports that the guard asked for the hold; only then does the
	// hold record guard outcomes.
	guarded bool
	// diagnostics counts the request's diagnostics when the answer began.
	diagnostics int
}

// holdStreamedAnswer asks the gateway to buffer a streamed answer whose
// response_guard can refuse or rewrite it, or whose structured output pass
// can replace it, so that no byte reaches the client before the Router has
// read the whole answer. A gateway that streams the body anyway still sends
// the client nothing early: the Router holds every part itself.
func holdStreamedAnswer(ctx *RequestContext, response *ext_proc.ProcessingResponse) {
	if ctx == nil || !ctx.IsStreamingResponse {
		return
	}
	guarded := requestResponseGuard(ctx).Enforces()
	if !guarded && ctx.structuredOutputPass == nil {
		return
	}
	ctx.ResponseGuardHold = &responseGuardHold{guarded: guarded, diagnostics: len(ctx.ProtocolDiagnostics)}
	response.ModeOverride = &http_ext.ProcessingMode{ResponseBodyMode: http_ext.ProcessingMode_BUFFERED}
}

// handleHeldStreamBody reads one part of a held stream as the streaming path
// does and keeps what it would have sent. Once the stream ends, the guard reads
// the whole answer and the client receives, in its protocol's streaming form,
// the held events, the guard's refusal or, for a stream that ended without an
// answer, only its failure.
func (r *OpenAIRouter) handleHeldStreamBody(body []byte, endOfStream bool, ctx *RequestContext) *ext_proc.ProcessingResponse {
	hold := ctx.ResponseGuardHold
	if hold.outcome != "" {
		// The client already has the hold's last word.
		return heldStreamPart(nil)
	}
	r.initializeSemanticResponseStream(ctx)
	if len(body) > responseGuardHoldLimit-hold.read {
		r.completeSemanticStream(ctx, errResponseGuardHoldLimit)
		ctx.ResponsePath = headers.ResponsePathBlocked
		message := "Response blocked: the answer is larger than the response guard can check"
		if !hold.guarded {
			message = "Response blocked: the answer is larger than the Router can hold"
		}
		return r.endUnreadStream(ctx, responseGuardOverLimit, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest,
			"response_guard_hold_limit", message, nil), "request_rejected")
	}
	hold.read += len(body)
	buffers := semanticStreamBuffers{}
	body, endOfStream = r.bridgeBufferedBody(&buffers, body, endOfStream, ctx)
	buffers.push(body, ctx)
	if endOfStream && buffers.unread(ctx) {
		r.recordUnreadStream(ctx, responseGuardIncomplete)
		return r.invalidUpstreamResponse(ctx, "The selected model returned an invalid response")
	}
	// A held stream is still the model's answer arriving: it records TTFT here
	// too, before the release decides what the client sees.
	recordStreamingTTFT(ctx)
	if endOfStream {
		buffers.finalize(ctx)
	}
	hold.frames = append(hold.frames, clientStreamBytes(buffers.processingResponse(ctx), body)...)
	if !endOfStream {
		return heldStreamPart(nil)
	}
	clean := buffers.streamErr == nil && !ctx.StreamingAborted
	var passErr error
	semanticResponse := r.completeSemanticStreamWith(ctx, buffers.streamErr, func(first *llmprotocol.Response) *llmprotocol.Response {
		second, err := r.structuredOutputAnswer(ctx, first, hold.diagnostics)
		if second == nil {
			passErr = err
			return first
		}
		return second
	})
	if semanticResponse == nil || ctx.StreamingAborted {
		return r.endUnreadStream(ctx, responseGuardIncomplete, heldStreamFailure(ctx, buffers.streamErr), heldStreamFailureReason(ctx, clean))
	}
	if passErr == nil && ctx.structuredOutputPass != nil && ctx.structuredOutputPass.answered {
		hold.frames, passErr = r.encodeClientStream(ctx, *semanticResponse)
	}
	if passErr != nil {
		_, protocolError, reason := r.structuredOutputRefusal(ctx, passErr)
		return r.refuseHeldStream(ctx, protocolError, reason)
	}
	return r.releaseHeldStream(ctx, semanticResponse)
}

// releaseHeldStream applies the buffered response policy to the answer of a
// held stream: the client receives the held events, or the events of the
// answer with the guard's notes, or the policy's refusal.
func (r *OpenAIRouter) releaseHeldStream(ctx *RequestContext, semanticResponse *llmprotocol.Response) *ext_proc.ProcessingResponse {
	hold := ctx.ResponseGuardHold
	answer, err := r.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "neutral_stream_replay_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
	}
	r.updateKVAddressRegistry(ctx)
	plan := r.prepareResponsePolicy(ctx, semanticResponse, answer)
	hold.outcome = ctx.ResponseGuard.outcome()
	if hold.guarded {
		metrics.RecordResponseGuardBuffered(requestDecisionStateKey(ctx), hold.outcome)
	}
	if plan.blocked != nil {
		r.finishNativeFallbackResponse(ctx, "policy_blocked")
		plan.commit()
		return r.refuseHeldStream(ctx, plan.refusal, "request_rejected")
	}
	// The cache keeps the answer the client receives, as a buffered answer's.
	frames, delivered := hold.frames, answer
	if !bytes.Equal(plan.finalBody, answer) {
		if noted := r.encodeNotedStream(ctx); noted != nil {
			frames, delivered = noted, plan.finalBody
		}
	}
	if delivered != nil {
		r.updateResponseCache(ctx, delivered)
	}
	r.finishNativeFallbackResponse(ctx, "")
	plan.commit()
	response := heldStreamPart(frames)
	if options := plan.headerOptions(); len(options) > 0 {
		response.GetResponseBody().GetResponse().HeaderMutation = &ext_proc.HeaderMutation{SetHeaders: options}
	}
	if warnings, ok := bufferedWarnings(ctx); ok {
		setResponseBodyHeaderOverwrite(response, headers.VSRWarnings, warnings)
	}
	return response
}

// encodeNotedStream renders the answer with the guard's notes, which the
// policy wrote into it, as the client's stream. It is nil when the noted
// answer has no streaming form: the client then receives the events as the
// model sent them, as a buffered answer keeps its body when the noted one
// cannot be encoded.
func (r *OpenAIRouter) encodeNotedStream(ctx *RequestContext) []byte {
	stream, err := r.encodeClientStream(ctx, *ctx.SemanticResponse)
	if err == nil {
		return stream
	}
	logging.ComponentErrorEvent("extproc", "response_guard_note_stream_encode_failed", map[string]interface{}{
		"request_id": ctx.RequestID,
		"format":     ctx.SourceFormat,
		"error":      err.Error(),
	})
	return nil
}

// endUnreadStream ends a held stream the guard could not read. Nothing of
// the answer leaves unread: the client receives only protocolError.
func (r *OpenAIRouter) endUnreadStream(ctx *RequestContext, outcome string, protocolError *llmprotocol.ProtocolError, reason string) *ext_proc.ProcessingResponse {
	r.recordUnreadStream(ctx, outcome)
	return r.refuseHeldStream(ctx, protocolError, reason)
}

// recordHeldStreamOverMessageLimit records a held stream that arrived as one
// ExtProc message larger than the server admits. The gateway buffered the
// whole answer, the server refused the message, and the gateway fails the
// request with none of the answer sent.
func (r *OpenAIRouter) recordHeldStreamOverMessageLimit(ctx *RequestContext, err error) {
	if hold := ctx.ResponseGuardHold; hold != nil && hold.outcome == "" && status.Code(err) == codes.ResourceExhausted {
		r.recordUnreadStream(ctx, responseGuardOverLimit)
	}
}

func (r *OpenAIRouter) recordUnreadStream(ctx *RequestContext, outcome string) {
	ctx.ResponseGuardHold.outcome = outcome
	if !ctx.ResponseGuardHold.guarded {
		return
	}
	metrics.RecordResponseGuardBuffered(requestDecisionStateKey(ctx), outcome)
	traceResponseGuard(ctx, outcome)
	r.recordRouterReplayUnreadAnswer(ctx, outcome)
	logging.ComponentWarnEvent("extproc", "response_guard_stream_unread", map[string]interface{}{
		"request_id": ctx.RequestID,
		"decision":   ctx.VSRSelectedDecisionName,
		"outcome":    outcome,
		"bytes":      ctx.ResponseGuardHold.read,
	})
}

// refuseHeldStream answers the client with protocolError in its protocol's
// streaming form, in place of everything the hold kept, and fails the replay
// record with it for reason. The guard's own refusal names the blocked
// response path; a stream that failed keeps its. A stream that ended early,
// with no reason, stays an abort.
func (r *OpenAIRouter) refuseHeldStream(ctx *RequestContext, protocolError *llmprotocol.ProtocolError, reason string) *ext_proc.ProcessingResponse {
	refusal := r.encodeStreamRefusal(ctx, protocolError)
	if reason != "" {
		r.failStreamedRouterReplay(ctx, refusal, reason)
	}
	response := heldStreamPart(refusal)
	if ctx.ResponsePath == headers.ResponsePathBlocked {
		setResponseBodyHeaderOverwrite(response, headers.VSRResponsePath, ctx.ResponsePath)
	}
	if trace := responseTrace(ctx); trace != "" {
		setResponseBodyHeaderOverwrite(response, headers.VSRTrace, trace)
	}
	return response
}

// responseGuardRefusal is the error a client receives for an answer a rule
// blocked: a buffered answer's 400 and a stream's error event. It is the
// invalid request a blocking fast_response gives, not a permission error,
// which Claude Code reads as a failed login and retries.
func responseGuardRefusal(rule *responseGuardRuleRun) *llmprotocol.ProtocolError {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "response_blocked", responseGuardBlockMessage(rule), nil)
}

// streamedFallbackRefusal answers a streaming client whose fallback answer
// the response policy refused: the refusal in its protocol's streaming form,
// as for a held stream.
func (r *OpenAIRouter) streamedFallbackRefusal(ctx *RequestContext, refusal *llmprotocol.ProtocolError) *ext_proc.ProcessingResponse {
	return r.buildFallbackImmediateResponse(r.encodeStreamRefusal(ctx, refusal), "text/event-stream", ctx)
}

// encodeStreamRefusal renders protocolError as a stream of the client's
// protocol that fails at once: a Chat error chunk, a Responses response.failed
// event or a Messages error event. It is empty when the refusal cannot be
// encoded, so the answer stays refused.
func (r *OpenAIRouter) encodeStreamRefusal(ctx *RequestContext, protocolError *llmprotocol.ProtocolError) []byte {
	body, err := r.encodeClientStream(ctx, llmprotocol.Response{
		Generation: 1, ID: heldStreamResponseID(ctx), CreatedAt: time.Now().UTC(),
		Model: ctx.RequestModel, StopReason: llmprotocol.StopError, Error: protocolError,
	})
	if err != nil {
		logging.ComponentErrorEvent("extproc", "response_guard_refusal_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return nil
	}
	return body
}

// encodeClientStream renders a whole neutral response as a stream of the
// client's protocol.
func (r *OpenAIRouter) encodeClientStream(ctx *RequestContext, response llmprotocol.Response) ([]byte, error) {
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	if format == llmprotocol.AnthropicMessagesV1 {
		source, _ := responseWireFormats(ctx)
		response = protocolcodec.AnthropicReplyIdentifiers(source, response)
	}
	body, diagnostics, err := engine.EncodeResponseStream(format, response, llmprotocol.StreamContext{
		Context: ctx.TraceContext, Source: format, Target: format,
		Options: clientStreamOptions(ctx), ResponseRender: responseObjectRender(ctx),
	})
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	return body, err
}

// heldStreamResponseID is the response ID the client's stream carries: the
// public ID of a Responses object, else the one the upstream stream named.
func heldStreamResponseID(ctx *RequestContext) string {
	if id := responseObjectPublicID(ctx); id != "" {
		return id
	}
	if state := ctx.SemanticStreamState; state != nil && state.responseID != "" {
		return state.responseID
	}
	return "resp_" + ctx.RequestID
}

// heldStreamFailure is the error a held stream ended on: the upstream's
// failure event, else the Router's own reason for stopping.
func heldStreamFailure(ctx *RequestContext, streamErr error) *llmprotocol.ProtocolError {
	if state := ctx.SemanticStreamState; state != nil && state.failed != nil {
		return state.failed
	}
	var protocolError *llmprotocol.ProtocolError
	if errors.As(streamErr, &protocolError) {
		return protocolError
	}
	return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete",
		"the model's stream ended before the answer was complete", streamErr)
}

// heldStreamFailureReason is why the record of a held stream that ended
// without an answer the guard could read fails: the upstream's failure event,
// or a stream the Router read cleanly to its end but could not rebuild an
// answer from. A stream that ended early has none.
func heldStreamFailureReason(ctx *RequestContext, clean bool) string {
	state := ctx.SemanticStreamState
	switch {
	case state != nil && state.failed != nil:
		return "upstream_error_response"
	case clean && state.delivered():
		return "upstream_invalid_response"
	}
	return ""
}

// heldStreamPart is the reply to one part of a held stream: body replaces
// what the gateway read, nothing while the Router holds it.
func heldStreamPart(body []byte) *ext_proc.ProcessingResponse {
	if body == nil {
		body = []byte{}
	}
	return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: body},
	}, nil)
}

// clientStreamBytes is what a streaming reply sends the client for one
// upstream part: its body mutation, else the part itself.
func clientStreamBytes(response *ext_proc.ProcessingResponse, part []byte) []byte {
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		return mutation.GetBody()
	}
	return part
}
