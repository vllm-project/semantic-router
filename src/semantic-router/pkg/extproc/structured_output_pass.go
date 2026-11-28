package extproc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// structuredOutputCallTimeout is the second call's fallback timeout. The
// upstream route deadline can bound it further.
const structuredOutputCallTimeout = 300 * time.Second

// structuredOutputPass is a turn with callable tools and a JSON output format
// (llmprotocol.StructuredOutputWithTools). Model servers such as vLLM hold
// the whole completion to the format, so the primary dispatch goes without
// it, and a final answer that calls no tool is asked for again with it: body
// is that second call, for the same backend.
type structuredOutputPass struct {
	dispatch *providerDispatch
	body     []byte
	deadline time.Time
	// answered reports that the second call's answer replaced the first.
	answered bool
}

// structuredOutputCallError is a second call that no backend answered, or
// that its backend answered with an error.
type structuredOutputCallError struct {
	attempt *failedAttempt
}

func (e *structuredOutputCallError) Error() string {
	if e.attempt.transport {
		return fmt.Sprintf("structured output call failed: %s", e.attempt.replayReason())
	}
	return fmt.Sprintf("structured output call returned status %d", e.attempt.status)
}

// planStructuredOutputPass is the pass of a primary dispatch the Router can
// send the second call of itself: to a backend it knows the address of,
// outside Looper, for a request that asks for one answer.
func planStructuredOutputPass(dispatch *providerDispatch, ctx *RequestContext) *structuredOutputPass {
	request := ctx.SemanticRequest
	if dispatch == nil || dispatch.backendAddress == "" || ctx.LooperRequest || request == nil ||
		(request.CandidateCount != nil && *request.CandidateCount > 1) ||
		!llmprotocol.StructuredOutputWithTools(*request) {
		return nil
	}
	return &structuredOutputPass{dispatch: dispatch}
}

// encodeStructuredOutputRequest encodes the second call as the primary's
// was encoded, from the request as dispatched, with no tool callable and the
// format kept. What the format loses toward the backend joins the request's
// diagnostics; the tool_choice none the call asks with is the Router's.
func (r *OpenAIRouter) encodeStructuredOutputRequest(dispatch *providerDispatch, ctx *RequestContext) ([]byte, error) {
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	format := dispatch.targetFormat
	request := llmprotocol.ConstrainedAnswerRequest(*ctx.SemanticRequest)
	if policyErr := r.applyPromptCachePolicy(&request, ctx, format); policyErr != nil {
		return nil, policyErr
	}
	request, _ = llmprotocol.ProjectToolChoiceNone(request, format)
	request, diagnostics, err := r.projectRequestForBackendWithDiagnostics(request, dispatch.effectiveBackendModel(), format)
	if err != nil {
		return nil, err
	}
	r.requestUnstoredOnCallerLogin(format, &request)
	encoded, err := engine.EncodeRequest(format, request, ctx.ProtocolEnvelope)
	if err != nil {
		return nil, err
	}
	appendUnheldDiagnostics(ctx, slices.Concat(diagnostics, encoded.Diagnostics))
	return r.adaptProviderRequest(encoded.Body, dispatch, ctx)
}

// structuredOutputAnswer is the answer the client of a structured output
// pass receives for first, the answer of the dispatch without the format:
// nil when first is the turn's answer as it is, else the second call's, with
// both calls' usage. The diagnostics past firstDiagnostics describe first,
// which the second answer replaces. On error ctx keeps first, whose usage
// the request still settles.
func (r *OpenAIRouter) structuredOutputAnswer(
	ctx *RequestContext, first *llmprotocol.Response, firstDiagnostics int,
) (*llmprotocol.Response, error) {
	pass := ctx.structuredOutputPass
	if pass == nil || first == nil {
		return nil, nil
	}
	if !llmprotocol.FinalAnswer(*first) {
		logging.ComponentDebugEvent("extproc", "structured_output_pass_not_needed", map[string]interface{}{
			"request_id":  ctx.RequestID,
			"stop_reason": first.StopReason,
		})
		return nil, nil
	}
	started := time.Now()
	second, err := r.secondStructuredOutputAnswer(ctx, pass, firstDiagnostics)
	var usage llmprotocol.Usage
	if err == nil {
		usage, err = llmprotocol.AddUsage(first.Usage, second.Usage)
	}
	r.recordStructuredOutputPass(ctx, first, second, err, time.Since(started))
	if err != nil {
		ctx.SemanticResponse = first
		return nil, err
	}
	second.Usage = usage
	// The upstream bytes carry the second call's usage alone.
	second.Generation++
	pass.answered = true
	if ctx.BenchmarkInferenceCalls > 0 {
		ctx.BenchmarkInferenceCalls++
	}
	return second, nil
}

// secondStructuredOutputAnswer sends the second call and decodes its answer
// for the client in place of the first one's.
func (r *OpenAIRouter) secondStructuredOutputAnswer(
	ctx *RequestContext, pass *structuredOutputPass, firstDiagnostics int,
) (*llmprotocol.Response, error) {
	body, err := r.sendStructuredOutputCall(ctx, pass)
	if err != nil {
		return nil, err
	}
	if len(ctx.ProtocolDiagnostics) > firstDiagnostics {
		ctx.ProtocolDiagnostics = ctx.ProtocolDiagnostics[:firstDiagnostics]
	}
	ctx.protocolDiagnosticsReported = min(ctx.protocolDiagnosticsReported, len(ctx.ProtocolDiagnostics))
	second, err := r.decodeClientResponse(body, ctx)
	if err != nil {
		return nil, fmt.Errorf("decode the structured output answer: %w", err)
	}
	return second, nil
}

// sendStructuredOutputCall sends the second call to the backend of the
// primary dispatch, as the response-phase fallback sends a candidate's.
func (r *OpenAIRouter) sendStructuredOutputCall(ctx *RequestContext, pass *structuredOutputPass) ([]byte, error) {
	callCtx := ctx.TraceContext
	if callCtx == nil {
		callCtx = context.Background()
	}
	if !pass.deadline.IsZero() {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithDeadline(callCtx, pass.deadline)
		defer cancel()
	}
	dispatch := pass.dispatch
	if err := callCtx.Err(); err != nil {
		return nil, &structuredOutputCallError{attempt: failedCandidateAttempt(dispatch, 0, nil, err)}
	}
	body, status, err := r.dispatchProviderHTTP(callCtx, ctx, dispatch, providerCall{
		name:    "structured_output_pass",
		body:    pass.body,
		headers: fallbackRequestHeaders(ctx, dispatch.profile, r.fallbackOrchestratorForContext(ctx)),
		timeout: r.structuredOutputTimeout(ctx, dispatch),
	})
	if err != nil || status < http.StatusOK || status >= http.StatusMultipleChoices {
		return nil, &structuredOutputCallError{attempt: failedCandidateAttempt(dispatch, status, body, err)}
	}
	return body, nil
}

// structuredOutputTimeout picks the shortest positive bound from the effective
// total timeout and the configured per-try timeouts. A decision's explicit zero
// disables the inherited model total timeout.
func (r *OpenAIRouter) structuredOutputTimeout(ctx *RequestContext, dispatch *providerDispatch) time.Duration {
	var bounds []*time.Duration
	override := callReliability(ctx.VSRSelectedDecision)
	if override != nil {
		bounds = append(bounds, override.TotalTimeout, override.PerTryTimeout)
	}
	if r.Config != nil {
		reliability := r.Config.ModelConfig[dispatch.effectiveBackendModel()].Reliability
		if override == nil || override.TotalTimeout == nil {
			bounds = append(bounds, reliabilityDuration(reliability.TotalTimeout))
		}
		bounds = append(bounds, reliabilityDuration(reliability.PerTryTimeout))
	}
	timeout := time.Duration(0)
	for _, bound := range bounds {
		if bound != nil && *bound > 0 && (timeout == 0 || *bound < timeout) {
			timeout = *bound
		}
	}
	if timeout == 0 {
		return structuredOutputCallTimeout
	}
	return timeout
}

// structuredOutputRefusal is what the client receives for a second call
// that gave no answer, with the status and the reason that ends the replay
// record: the backend's error in the client's terms, as for a failed
// fallback attempt, or 502 for an answer the Router could not read.
func (r *OpenAIRouter) structuredOutputRefusal(ctx *RequestContext, err error) (int, *llmprotocol.ProtocolError, string) {
	var callErr *structuredOutputCallError
	if errors.As(err, &callErr) {
		return callErr.attempt.status, r.failedAttemptError(ctx, callErr.attempt), callErr.attempt.replayReason()
	}
	return http.StatusBadGateway, immediateProtocolError(http.StatusBadGateway), "upstream_invalid_response"
}

// refuseStructuredOutput answers a buffered request whose second call gave
// no answer. The first answer is not sent instead: it does not follow the
// format the client asked for.
func (r *OpenAIRouter) refuseStructuredOutput(ctx *RequestContext, err error) *ext_proc.ProcessingResponse {
	status, protocolError, reason := r.structuredOutputRefusal(ctx, err)
	return r.answerFailedCall(ctx, status, protocolError, reason)
}

// recordStructuredOutputPass logs a second call and appends it to the
// request's replay record: replaced, with each call's output tokens, or
// failed, with the reason the record ends with.
func (r *OpenAIRouter) recordStructuredOutputPass(
	ctx *RequestContext, first, second *llmprotocol.Response, err error, elapsed time.Duration,
) {
	fields := map[string]interface{}{
		"request_id":          ctx.RequestID,
		"model":               ctx.VSRSelectedModel,
		"duration_ms":         elapsed.Milliseconds(),
		"first_output_tokens": tokenCountText(first.Usage.OutputTotal),
	}
	outcome := routerreplay.Outcome{
		Timestamp: time.Now().UTC(), Source: "router", Target: "structured_output",
		TargetRef: ctx.VSRSelectedModel, Verdict: "replaced",
		Metadata: map[string]string{"first_output_tokens": tokenCountText(first.Usage.OutputTotal)},
	}
	if ctx.VSRSelectedDecisionName != "" {
		outcome.Metadata["decision"] = ctx.VSRSelectedDecisionName
	}
	if err != nil {
		_, _, reason := r.structuredOutputRefusal(ctx, err)
		outcome.Verdict, outcome.Reason = "failed", reason
		fields["error"] = err.Error()
		logging.ComponentWarnEvent("extproc", "structured_output_pass_failed", fields)
	} else {
		outcome.Metadata["second_output_tokens"] = tokenCountText(second.Usage.OutputTotal)
		logging.ComponentEvent("extproc", "structured_output_pass_answered", fields)
	}
	recorder := r.responseGuardReplayRecorder(ctx)
	if recorder == nil {
		return
	}
	if appendErr := recorder.AppendOutcome(ctx.RouterReplayID, outcome); appendErr != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_structured_output_outcome_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      appendErr.Error(),
		})
	}
}

// tokenCountText is a count as a replay record keeps it, empty when unknown.
func tokenCountText(count llmprotocol.TokenCount) string {
	if count.Value == nil {
		return ""
	}
	return strconv.FormatInt(*count.Value, 10)
}
