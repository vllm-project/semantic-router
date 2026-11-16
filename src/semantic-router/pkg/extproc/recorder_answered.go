package extproc

import (
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// recordFastResponse opens the record of a request a fast_response answered
// with response. The message as the answer completes the record. A block
// fails it with request_rejected, under the 200 a streamed block answers. A
// block's 400, or a fast response that could not be encoded, is an error,
// which ends the record when requestBodyReply sends it.
func (r *OpenAIRouter) recordFastResponse(ctx *RequestContext, response *ext_proc.ProcessingResponse, originalModel, selectedModel, decisionName string) {
	r.startRouterReplay(ctx, originalModel, selectedModel, decisionName)
	immediate := response.GetImmediateResponse()
	status := int(immediate.GetStatus().GetCode())
	r.updateRouterReplayStatus(ctx, status, false)
	if status < http.StatusBadRequest && ctx.ResponsePath == headers.ResponsePathBlocked {
		r.failStreamedRouterReplay(ctx, immediate.GetBody(), "request_rejected")
		return
	}
	r.attachRouterReplayResponse(ctx, immediate.GetBody(), status < http.StatusBadRequest)
}

// openRoutedRouterReplay opens the record of a request the Router answers
// itself after routing it, before any dispatch opened one: a refusal of the
// selected model, such as a capability its wire lacks, or of a stage after the
// decision, such as a rate limit. A request answered before routing has no
// route to record.
func (r *OpenAIRouter) openRoutedRouterReplay(ctx *RequestContext) {
	if ctx == nil || ctx.VSRSelectedDecision == nil && ctx.VSRSelectedModel == "" {
		return
	}
	decisionName := ctx.VSRSelectedDecisionName
	if ctx.VSRSelectedDecision != nil {
		decisionName = ctx.VSRSelectedDecision.Name
	}
	r.startRouterReplay(ctx, ctx.RequestModel, ctx.VSRSelectedModel, decisionName)
}

// routerReplayInProgress reports whether the request has a replay record that
// has not ended yet.
func (r *OpenAIRouter) routerReplayInProgress(ctx *RequestContext) bool {
	recorder := r.responseGuardReplayRecorder(ctx)
	if recorder == nil {
		return false
	}
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	return ok && record.LifecycleState == routerreplay.LifecycleInProgress
}

// failStreamedRouterReplay fails a record still in progress with what its
// client got, under the status the record holds: for a stream, the 200 it
// began with before an error event took the place of the answer.
func (r *OpenAIRouter) failStreamedRouterReplay(ctx *RequestContext, body []byte, reason string) {
	if !r.routerReplayInProgress(ctx) {
		return
	}
	r.attachRouterReplayResponse(ctx, body, false)
	r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, reason)
}

// requestAnswerReason names why the Router answered a request itself with
// status: a refusal, or a failure of its own for a 5xx. A guard overload is a
// refusal its client retries, whichever status its protocol carries it in.
func requestAnswerReason(ctx *RequestContext, status int) string {
	if status >= http.StatusInternalServerError && (ctx == nil || !ctx.guardOverloadAnswered) {
		return "request_processing_failed"
	}
	return "request_rejected"
}

// endStreamRouterReplay ends the record of a stream whose answer the Router
// did not keep: failed when the stream carried the upstream's error event,
// and complete when it reached its terminal event and the Router read it
// cleanly, since its client then has the whole stream. Any other stream
// ended early and stays an abort.
func (r *OpenAIRouter) endStreamRouterReplay(ctx *RequestContext, clean bool) {
	state := ctx.SemanticStreamState
	switch {
	case state != nil && state.failed != nil:
		r.failStreamedRouterReplay(ctx, nil, "upstream_error_response")
	case clean && state.delivered() && r.routerReplayInProgress(ctx):
		r.attachRouterReplayResponse(ctx, nil, true)
	}
}
