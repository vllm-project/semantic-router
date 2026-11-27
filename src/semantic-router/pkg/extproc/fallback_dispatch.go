package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"
	"unicode/utf8"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type fallbackTransportCaller func(ctx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error)

// isClientStreamingRequested reports whether the original downstream client requested a streaming response.
func isClientStreamingRequested(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	return ctx.ExpectStreamingResponse || (ctx.SemanticRequest != nil && ctx.SemanticRequest.Stream)
}

// candidateReasoningChoice resolves the effective reasoning policy for a
// fallback candidate, off when a request that states no reasoning preference
// sets an output limit too small for it or stop sequences.
func (r *OpenAIRouter) candidateReasoningChoice(ctx *RequestContext, candidateModel string) bool {
	if ctx == nil {
		return false
	}
	return r.decisionReasoningChoice(ctx, candidateModel) && r.unstatedReasoningOffReason(ctx, candidateModel) == ""
}

func (r *OpenAIRouter) decisionReasoningChoice(ctx *RequestContext, candidateModel string) bool {
	for i := range ctx.VSREligibleModelRefs {
		ref := &ctx.VSREligibleModelRefs[i]
		if candidateModelIdentity(*ref) == candidateModel {
			if ref.UseReasoning != nil {
				return *ref.UseReasoning
			}
			break
		}
	}
	if ctx.VSRSelectedDecision != nil {
		useReasoning, _ := r.getReasoningInfoFromDecision(ctx.VSRSelectedDecision, candidateModel)
		return useReasoning
	}
	return false
}

// recordPrimarySuccess resets consecutive failures for the primary backend when an attempt succeeds.
func (r *OpenAIRouter) recordPrimarySuccess(ctx *RequestContext) {
	if r == nil || ctx == nil || ctx.nativeFallbackResponse != nil {
		return
	}
	orch := r.fallbackOrchestratorForContext(ctx)
	if orch == nil || orch.CircuitBreaker() == nil {
		return
	}
	primaryModel := ctx.VSRSelectedModel
	if primaryModel == "" {
		primaryModel = ctx.RequestModel
	}
	backendName := r.primaryBackendForAccounting(ctx, primaryModel)
	orch.CircuitBreaker().RecordSuccess(backendName)
}

func (r *OpenAIRouter) primaryBackendForAccounting(ctx *RequestContext, primaryModel string) string {
	if ctx != nil && ctx.primaryBackendName != "" {
		return ctx.primaryBackendName
	}
	backendName := primaryModel
	if ctx != nil {
		if dispatch, err := r.resolveProviderDispatchForCandidate(primaryModel, ctx.VSRSelectedDecisionName, false, ctx); err == nil && dispatch != nil {
			backendName = dispatch.backendName
		}
	}
	return backendName
}

// shouldAttemptFallback reports whether fallback evaluation should be attempted for the request context.
func (r *OpenAIRouter) shouldAttemptFallback(ctx *RequestContext) bool {
	orch := r.fallbackOrchestratorForContext(ctx)
	if r == nil || orch == nil || ctx == nil || ctx.SemanticRequest == nil {
		return false
	}
	if !orch.Policy().Enabled {
		return false
	}
	if ctx.LooperRequest || ctx.ResponseHeadersContinued || ctx.fallbackExecutedByCaller {
		return false
	}
	return len(ctx.VSREligibleModelRefs) > 1
}

func ensureFallbackRecord(orch *fallback.Orchestrator, ctx *RequestContext, model string) *fallback.ExecutionRecord {
	if ctx.FallbackRecord == nil {
		record := orch.NewExecutionRecord(ctx.RequestID, ctx.VSRSelectedDecisionName, model)
		record.SessionID = ctx.SessionID
		if ctx.SemanticRequest != nil {
			record.ConversationID = ctx.SemanticRequest.ConversationID
		}
		if record.ConversationID == "" && ctx.ResponseObjectState != nil {
			record.ConversationID = ctx.ResponseObjectState.ConversationID
		}
		ctx.FallbackRecord = record
	}
	return ctx.FallbackRecord
}

// maybeExecuteFallback evaluates the upstream failure and, if eligible, coordinates
// bounded fallback across candidate models from the decision's hard-eligible list.
func (r *OpenAIRouter) maybeExecuteFallback(body []byte, ctx *RequestContext) *ext_proc.ProcessingResponse {
	orch := r.fallbackOrchestratorForContext(ctx)
	if !r.shouldAttemptFallback(ctx) || orch == nil {
		return nil
	}

	commit := fallbackCommitState(ctx)
	commit.ResponseCommitted = ctx.ResponseHeadersContinued || ctx.StreamingComplete || (ctx.IsStreamingResponse && ctx.TTFTRecorded)

	primaryModel := ctx.VSRSelectedModel
	if primaryModel == "" {
		primaryModel = ctx.RequestModel
	}
	backendName := r.primaryBackendForAccounting(ctx, primaryModel)

	ensureFallbackRecord(orch, ctx, primaryModel)
	defer r.recordFallbackExecution(ctx)

	var primaryDuration time.Duration
	if !ctx.ProcessingStartTime.IsZero() {
		primaryDuration = time.Since(ctx.ProcessingStartTime)
	}

	primaryErrorMsg := fmt.Sprintf("upstream returned status %d", ctx.UpstreamStatusCode)
	if len(body) > 0 {
		primaryErrorMsg = fmt.Sprintf("upstream returned status %d: %s", ctx.UpstreamStatusCode, string(body))
	}
	primaryError := errors.New(primaryErrorMsg)

	primaryOutcome := fallback.AttemptOutcome{
		Model:        primaryModel,
		Backend:      backendName,
		StatusCode:   ctx.UpstreamStatusCode,
		Error:        primaryError,
		ErrorMessage: primaryErrorMsg,
		Duration:     primaryDuration,
		Discarded:    true,
	}

	callCtx := ctx.TraceContext
	if callCtx == nil {
		callCtx = context.Background()
	}

	source, _ := responseWireFormats(ctx)
	r.noteLengthRefusal(ctx, &primaryOutcome, source, body)
	evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, primaryOutcome, commit)
	if !evalResult.CanFallback {
		logging.ComponentDebugEvent("extproc", "fallback_not_allowed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"reason":     evalResult.Reason,
			"error":      evalResult.Err,
		})
		return nil
	}

	if orch.Policy().TotalTimeout > 0 {
		var elapsed time.Duration
		if !ctx.ProcessingStartTime.IsZero() {
			elapsed = time.Since(ctx.ProcessingStartTime)
		}
		remainingTotal := orch.Policy().TotalTimeout - elapsed
		if remainingTotal <= 0 {
			ctx.FallbackRecord.TotalDuration = elapsed
			ctx.FallbackRecord.FinalStatus = "total_deadline_exceeded"
			fallback.CorrelateExecution(ctx.FallbackRecord)
			logging.ComponentDebugEvent("extproc", "fallback_total_timeout_exceeded", map[string]interface{}{
				"request_id": ctx.RequestID,
				"elapsed":    elapsed.String(),
			})
			return nil
		}
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(callCtx, remainingTotal)
		defer cancel()
	}

	logging.ComponentEvent("extproc", "fallback_initiated", map[string]interface{}{
		"request_id":    ctx.RequestID,
		"primary_model": primaryModel,
		"status":        ctx.UpstreamStatusCode,
		"candidates":    len(ctx.VSREligibleModelRefs),
	})

	for {
		candIdx, err := orch.SelectNextCandidateIndex(
			ctx.FallbackRecord,
			len(ctx.VSREligibleModelRefs),
			func(i int) string { return candidateModelIdentity(ctx.VSREligibleModelRefs[i]) },
			func(model string) (string, error) {
				useReasoning := r.candidateReasoningChoice(ctx, model)
				dispatch, err := r.resolveProviderDispatch(model, ctx.VSRSelectedDecisionName, useReasoning)
				if err != nil {
					for _, ref := range ctx.VSREligibleModelRefs {
						if candidateModelIdentity(ref) == model && ref.Model != "" {
							if baseDispatch, baseErr := r.resolveProviderDispatch(ref.Model, ctx.VSRSelectedDecisionName, useReasoning); baseErr == nil {
								return baseDispatch.backendName, nil
							}
						}
					}
					return model, nil
				}
				return dispatch.backendName, nil
			},
		)
		if err != nil || candIdx < 0 || candIdx >= len(ctx.VSREligibleModelRefs) {
			logging.ComponentWarnEvent("extproc", "fallback_candidates_exhausted", map[string]interface{}{
				"request_id": ctx.RequestID,
				"attempts":   len(ctx.FallbackRecord.Attempts),
			})
			break
		}
		nextCandidate := &ctx.VSREligibleModelRefs[candIdx]

		resp, candidateEval, attemptErr := r.executeFallbackCandidate(callCtx, nextCandidate, commit, ctx, orch)
		if attemptErr == nil && resp != nil {
			return resp
		}

		if !candidateEval.CanFallback {
			if ctx.FallbackRecord.FinalStatus == "" {
				ctx.FallbackRecord.FinalStatus = fallbackHaltStatus(attemptErr)
			}
			logging.ComponentDebugEvent("extproc", "fallback_loop_halted", map[string]interface{}{
				"request_id":   ctx.RequestID,
				"final_status": ctx.FallbackRecord.FinalStatus,
				"reason":       candidateEval.Reason,
			})
			break
		}
	}

	if refused := r.fallbackCredentialRefusal(ctx); refused != nil {
		return refused
	}
	return r.failedAttemptResponse(ctx, ctx.fallbackLastFailure)
}

// fallbackHaltStatus names why a chain stopped at a candidate that could not
// be tried.
func fallbackHaltStatus(err error) string {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "total_deadline_exceeded"
	case errors.Is(err, context.Canceled):
		return "context_canceled"
	case errors.Is(err, principal.ErrCredentialMissing):
		return delegatedCredentialCode
	default:
		return "fallback_halted"
	}
}

func (r *OpenAIRouter) executeFallbackCandidate(
	callCtx context.Context,
	candidateRef *config.ModelRef,
	commit fallback.CommitState,
	ctx *RequestContext,
	orch *fallback.Orchestrator,
) (*ext_proc.ProcessingResponse, fallback.EvaluationResult, error) {
	if orch == nil {
		orch = r.fallbackOrchestratorForContext(ctx)
	}
	if callCtx == nil {
		callCtx = context.Background()
	}
	attemptCtx := callCtx
	if orch != nil {
		var cancel context.CancelFunc
		attemptCtx, cancel = orch.AttemptContext(callCtx)
		defer cancel()
	}
	if isContextExpired(callCtx) {
		return nil, fallback.EvaluationResult{CanFallback: false}, context.DeadlineExceeded
	}
	if err := attemptCtx.Err(); err != nil {
		return nil, fallback.EvaluationResult{CanFallback: false}, err
	}

	origSelectedCandidate := ctx.VSRSelectedCandidate
	ctx.VSRSelectedCandidate = candidateRef
	origTraceContext := ctx.TraceContext
	ctx.TraceContext = attemptCtx
	origSemanticRequest := ctx.SemanticRequest
	origRateLimitCtx := ctx.RateLimitCtx
	origDiags := ctx.ProtocolDiagnostics
	origTargetFormat := ctx.TargetFormat
	var candidateSucceeded bool
	defer func() {
		ctx.TraceContext = origTraceContext
		if !candidateSucceeded {
			ctx.VSRSelectedCandidate = origSelectedCandidate
			ctx.SemanticRequest = origSemanticRequest
			ctx.RateLimitCtx = origRateLimitCtx
			ctx.ProtocolDiagnostics = origDiags
		}
	}()
	primaryModel := ctx.FallbackRecord.InitialModel
	primaryStatusCode := ctx.UpstreamStatusCode
	origPath := ctx.ResponsePath
	wasStreaming := ctx.IsStreamingResponse
	// The response-phase fallback answers with a buffered immediate response.
	candidate, prepared, err := r.prepareFallbackCandidate(callCtx, candidateRef, ctx, true)
	if err != nil {
		logSkippedFallbackCandidate(ctx, candidateRef, prepared, err)
		return nil, prepared, err
	}
	candidateModel, dispatch, reqCopy, requestBody := candidate.model, candidate.dispatch, candidate.request, candidate.body
	engine := candidate.engine
	var resBody []byte
	var statusCode int
	var callErr error
	attemptStart := time.Now()

	if r.fallbackCaller != nil {
		extraHeaders := fallbackRequestHeaders(ctx, dispatch.profile, orch)
		resBody, statusCode, callErr = r.fallbackCaller(attemptCtx, candidateModel, requestBody, extraHeaders)
	} else {
		resBody, statusCode, callErr = r.dispatchFallbackHTTP(attemptCtx, ctx, dispatch, requestBody, orch)
	}
	attemptDuration := time.Since(attemptStart)

	attemptOutcome := fallback.AttemptOutcome{
		Model:            candidateModel,
		Backend:          dispatch.backendName,
		StatusCode:       statusCode,
		Error:            callErr,
		Duration:         attemptDuration,
		Discarded:        callErr != nil || statusCode < 200 || statusCode >= 300,
		ForwardedHeaders: forwardedFallbackHeaders(ctx, orch),
	}

	if callErr != nil || statusCode < 200 || statusCode >= 300 {
		attemptOutcome.ErrorMessage = string(resBody)
		if callErr == nil {
			r.noteLengthRefusal(ctx, &attemptOutcome, dispatch.targetFormat, resBody)
		}
		ctx.fallbackLastFailure = failedCandidateAttempt(dispatch, statusCode, resBody, callErr)
		evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
		logging.ComponentWarnEvent("extproc", "fallback_candidate_attempt_failed", map[string]interface{}{
			"request_id":        ctx.RequestID,
			"candidate_model":   candidateModel,
			"status":            statusCode,
			"error":             callErr,
			"can_fallback":      evalResult.CanFallback,
			"final_status":      ctx.FallbackRecord.FinalStatus,
			"forwarded_headers": attemptOutcome.ForwardedHeaders,
		})
		return nil, evalResult, fmt.Errorf("candidate %s failed with status %d: %w", candidateModel, statusCode, callErr)
	}

	responseEngine, engineErr := r.protocolEngineForReply(resolveResponseVendor(dispatch.profile), reqCopy)
	if engineErr != nil {
		attemptOutcome.Error = engineErr
		attemptOutcome.Discarded = true
		evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
		return nil, evalResult, engineErr
	}

	mutation := clientResponseMutation(ctx, dispatch.targetFormat)
	translated, translateErr := responseEngine.TranslateResponse(dispatch.targetFormat, ctx.SourceFormat, resBody, mutation)
	if translateErr != nil {
		attemptOutcome.Error = translateErr
		attemptOutcome.Discarded = true
		evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
		logging.ComponentWarnEvent("extproc", "fallback_candidate_attempt_failed", map[string]interface{}{
			"request_id":      ctx.RequestID,
			"candidate_model": candidateModel,
			"status":          statusCode,
			"error":           translateErr.Error(),
			"can_fallback":    evalResult.CanFallback,
			"final_status":    ctx.FallbackRecord.FinalStatus,
		})
		return nil, evalResult, translateErr
	}

	if translated.Response.Usage.InputTotal.Value != nil {
		attemptOutcome.PromptTokens = int(*translated.Response.Usage.InputTotal.Value)
	}
	if translated.Response.Usage.OutputTotal.Value != nil {
		attemptOutcome.CompletionTokens = int(*translated.Response.Usage.OutputTotal.Value)
	}
	if translated.Response.Usage.Total.Value != nil {
		attemptOutcome.TotalTokens = int(*translated.Response.Usage.Total.Value)
	}

	candidateResponse := translated.Response
	attemptOutcome.ProviderRequestID = candidateResponse.ProviderRequestID
	if candidateResponse.Model == "" {
		candidateResponse.Model = candidateModel
	}
	if publicID := responseObjectPublicID(ctx); publicID != "" {
		candidateResponse.ID = publicID
	} else if candidateResponse.ID == "" {
		candidateResponse.ID = "chatcmpl-" + ctx.RequestID
	}
	if ctx.SourceFormat == llmprotocol.AnthropicMessagesV1 {
		// A streaming client's answer is rendered from this response.
		candidateResponse = protocolcodec.AnthropicReplyIdentifiers(dispatch.targetFormat, candidateResponse)
	}

	var responseBody []byte
	var streamDiags []llmprotocol.Diagnostic
	contentType := "application/json"
	isStreaming := isClientStreamingRequested(ctx)
	if isStreaming {
		format := ctx.SourceFormat
		if format == "" {
			format = llmprotocol.OpenAIChatV1
		}
		streamBody, diags, streamErr := engine.EncodeResponseStream(format, candidateResponse, llmprotocol.StreamContext{
			Context:        ctx.TraceContext,
			Source:         format,
			Target:         format,
			Options:        clientStreamOptions(ctx),
			PublicModel:    candidateModel,
			ResponseID:     candidateResponse.ID,
			ResponseRender: responseObjectRender(ctx),
		})
		if streamErr != nil {
			attemptOutcome.Error = streamErr
			attemptOutcome.Discarded = true
			evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
			return nil, evalResult, fmt.Errorf("fallback stream encoding failed: %w", streamErr)
		}
		if validateErr := engine.ValidateEncodedResponse(streamBody, true); validateErr != nil {
			attemptOutcome.Error = validateErr
			attemptOutcome.Discarded = true
			evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
			return nil, evalResult, fmt.Errorf("fallback stream wire validation failed: %w", validateErr)
		}
		streamDiags = diags
		responseBody = streamBody
		contentType = "text/event-stream"
	} else {
		if validateErr := engine.ValidateEncodedResponse(translated.Body, false); validateErr != nil {
			attemptOutcome.Error = validateErr
			attemptOutcome.Discarded = true
			evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
			return nil, evalResult, fmt.Errorf("fallback body wire validation failed: %w", validateErr)
		}
		responseBody = translated.Body
	}

	if ctx.InflightToken != 0 {
		inflight.End(ctx.InflightModel, ctx.InflightToken)
		ctx.InflightToken = 0
	}

	ctx.RequestModel = candidateModel
	ctx.VSRSelectedModel = candidateModel
	ctx.ResponsePath = headers.ResponsePathFallback
	ctx.UpstreamStatusCode = statusCode
	ctx.TargetFormat = dispatch.targetFormat
	ctx.ResponseVendor = resolveResponseVendor(dispatch.profile)
	ctx.SemanticResponse = &candidateResponse
	// The call was buffered, but the client still reads its own stream, which
	// is rendered with its own stream options.
	reqCopy.Stream, reqCopy.StreamOptions = origSemanticRequest.Stream, origSemanticRequest.StreamOptions
	ctx.SemanticRequest = reqCopy
	ctx.ResponseEnvelope = translated.Envelope
	ctx.ResponseUnknownFields = protocolcodec.DiagnosticsDroppedUnknownFields(translated.Diagnostics)

	if isStreaming {
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, streamDiags...)
		ctx.IsStreamingResponse = true
	} else {
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, translated.Diagnostics...)
	}

	// Evaluate response policy and body mutations first, but delay its persistence
	// side effects until the final public wire has passed validation.
	policyPlan := r.prepareResponsePolicy(ctx, ctx.SemanticResponse, translated.Body)
	if policyPlan.blocked != nil {
		blockedStatus := http.StatusBadRequest
		if imm := policyPlan.blocked.GetImmediateResponse(); imm != nil && imm.GetStatus() != nil {
			if code := int(imm.GetStatus().GetCode()); code != 0 {
				blockedStatus = code
			}
		}
		ctx.UpstreamStatusCode = blockedStatus
		evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
		if !evalResult.Succeeded {
			return nil, evalResult, fmt.Errorf("fallback attempt evaluation: %s", evalResult.Reason)
		}
		ctx.FallbackRecord.FinalStatus = "policy_blocked"
		r.settleFallbackCandidateUsage(ctx, attemptDuration)
		policyPlan.commit()
		metrics.RecordModelRouting(primaryModel, candidateModel)
		candidateSucceeded = true
		if isStreaming {
			r.updateRouterReplayStatus(ctx, http.StatusOK, true)
			return r.streamedFallbackRefusal(ctx, policyPlan.refusal), evalResult, nil
		}
		r.updateRouterReplayStatus(ctx, blockedStatus, false)
		return policyPlan.blocked, evalResult, nil
	}
	finalBody := policyPlan.finalBody
	if isStreaming {
		if !bytes.Equal(finalBody, translated.Body) {
			format := ctx.SourceFormat
			if format == "" {
				format = llmprotocol.OpenAIChatV1
			}
			streamResp := *ctx.SemanticResponse
			reencodedStream, _, streamErr := engine.EncodeResponseStream(format, streamResp, llmprotocol.StreamContext{
				Context:        ctx.TraceContext,
				Source:         format,
				Target:         format,
				Options:        clientStreamOptions(ctx),
				PublicModel:    candidateModel,
				ResponseID:     ctx.SemanticResponse.ID,
				ResponseRender: responseObjectRender(ctx),
			})
			if streamErr == nil {
				if validateErr := engine.ValidateEncodedResponse(reencodedStream, true); validateErr == nil {
					responseBody = reencodedStream
				} else {
					// The single SSE text frame exceeded Limits.SSEFrameBytes (e.g. policy warning pushed it over limit).
					// Produce a bounded stream containing the policy result by chunking text across frames within SSE limits.
					boundedResp := splitResponseTextContent(streamResp, 64*1024)
					boundedStream, _, boundedErr := engine.EncodeResponseStream(format, boundedResp, llmprotocol.StreamContext{
						Context:        ctx.TraceContext,
						Source:         format,
						Target:         format,
						Options:        clientStreamOptions(ctx),
						PublicModel:    candidateModel,
						ResponseID:     ctx.SemanticResponse.ID,
						ResponseRender: responseObjectRender(ctx),
					})
					if boundedErr == nil && engine.ValidateEncodedResponse(boundedStream, true) == nil {
						responseBody = boundedStream
					} else {
						attemptOutcome.Error = validateErr
						attemptOutcome.Discarded = true
						r.rollbackFallbackContext(ctx, primaryModel, primaryStatusCode, origPath, origDiags, wasStreaming, origTargetFormat)
						evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
						return nil, evalResult, fmt.Errorf("finalized fallback stream validation failed: %w", validateErr)
					}
				}
			} else {
				attemptOutcome.Error = streamErr
				attemptOutcome.Discarded = true
				r.rollbackFallbackContext(ctx, primaryModel, primaryStatusCode, origPath, origDiags, wasStreaming, origTargetFormat)
				evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
				return nil, evalResult, fmt.Errorf("finalized fallback stream encoding failed: %w", streamErr)
			}
		}
	} else {
		if validateErr := engine.ValidateEncodedResponse(finalBody, false); validateErr != nil {
			attemptOutcome.Error = validateErr
			attemptOutcome.Discarded = true
			r.rollbackFallbackContext(ctx, primaryModel, primaryStatusCode, origPath, origDiags, wasStreaming, origTargetFormat)
			evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
			return nil, evalResult, fmt.Errorf("finalized fallback body wire validation failed: %w", validateErr)
		}
		responseBody = finalBody
	}

	evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
	if !evalResult.Succeeded {
		return nil, evalResult, fmt.Errorf("fallback attempt evaluation: %s", evalResult.Reason)
	}

	// Settle usage, cost, session state, and response-policy persistence only
	// after the final public wire and the attempt itself are accepted.
	r.settleFallbackCandidateUsage(ctx, attemptDuration)
	policyPlan.commit()
	policyHeaders := policyPlan.headerOptions()

	metrics.RecordModelRouting(primaryModel, candidateModel)
	r.updateRouterReplayStatus(ctx, statusCode, isStreaming)
	r.recordServedFallback(ctx, candidate)

	// Write the finalized public response to cache only after response policy and wire validation succeed.
	// Blocked responses (jailbreak/hallucination) or rejected frames are never cached.
	r.updateResponseCache(ctx, r.cacheableClientResponse(finalBody, requiresClientResponseRewrite(ctx), *ctx.SemanticResponse, ctx))

	logging.ComponentEvent("extproc", "fallback_candidate_succeeded", map[string]interface{}{
		"request_id":      ctx.RequestID,
		"primary_model":   primaryModel,
		"candidate_model": candidateModel,
		"attempts":        len(ctx.FallbackRecord.Attempts),
	})

	immediateResp := r.buildFallbackImmediateResponse(responseBody, contentType, ctx, policyHeaders...)
	r.persistImmediateResponseObject(immediateResp, ctx)
	candidateSucceeded = true
	return immediateResp, evalResult, nil
}

// logSkippedFallbackCandidate says why a candidate was dropped before its
// call. The chain's record lists such a candidate as visited, with no attempt.
func logSkippedFallbackCandidate(
	ctx *RequestContext, candidateRef *config.ModelRef, prepared fallback.EvaluationResult, err error,
) {
	logging.ComponentWarnEvent("extproc", "fallback_candidate_skipped", map[string]interface{}{
		"request_id":      ctx.RequestID,
		"candidate_model": candidateModelIdentity(*candidateRef),
		"error":           err.Error(),
		"can_fallback":    prepared.CanFallback,
	})
}

// fallbackCandidate is a fallback candidate's request, prepared the way the
// primary dispatch was prepared.
type fallbackCandidate struct {
	model    string
	ref      *config.ModelRef
	dispatch *providerDispatch
	request  *llmprotocol.Request
	body     []byte
	engine   *protocolcodec.Engine
}

// prepareFallbackCandidate resolves a candidate's provider dispatch and
// encodes its request: reasoning, decision and request parameters,
// capability checks, codec encoding, provider adaptation and the rate limit,
// as for the primary. buffered asks for a non-streaming response, which the
// response-phase fallback needs to answer at once, so the call carries none of
// the client's stream options either; the native gateway keeps the client's
// streaming mode, and a streamed call asks a Chat backend for its usage as the
// primary's does.
func (r *OpenAIRouter) prepareFallbackCandidate(
	callCtx context.Context,
	candidateRef *config.ModelRef,
	ctx *RequestContext,
	buffered bool,
) (*fallbackCandidate, fallback.EvaluationResult, error) {
	candidateModel := candidateModelIdentity(*candidateRef)
	if err := r.fallbackWindowSkip(ctx, candidateRef); err != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, err
	}
	useReasoning := r.candidateReasoningChoice(ctx, candidateModel)
	dispatch, err := r.resolveProviderDispatchForCandidate(
		candidateModel, ctx.VSRSelectedDecisionName, useReasoning, ctx,
	)
	if err != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, err
	}
	if dispatch.profile.Delegated() {
		// A candidate the request cannot pay for ends the chain before its
		// call, and the request answers 401, as for the primary.
		if _, err := principal.Delegated(ctx.TraceContext); err != nil {
			return nil, fallback.EvaluationResult{CanFallback: false}, err
		}
	}

	requestBase := ctx.FallbackRequest
	requestAlreadyPrepared := requestBase != nil
	if requestBase == nil {
		requestBase = ctx.SemanticRequest
	}
	reqCopy, cloneErr := cloneSemanticRequestForReplay(requestBase)
	if cloneErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, cloneErr
	}
	reqCopy.Model = dispatch.upstreamModel
	if buffered {
		reqCopy.Stream = false
		reqCopy.StreamOptions = llmprotocol.StreamOptions{}
	}
	r.applyCandidateReasoning(ctx, reqCopy, candidateRef, dispatch, useReasoning)

	if requestAlreadyPrepared {
		if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
			r.applySemanticReasoningMode(
				reqCopy, dispatch.logicalModel, dispatch.targetFormat, dispatch.useReasoning,
				ctx.decisionForCandidate(dispatch.logicalModel),
			)
		}
	} else {
		if _, decisionErr := r.applyDispatchDecision(reqCopy, dispatch, ctx); decisionErr != nil {
			return nil, fallback.EvaluationResult{CanFallback: true}, decisionErr
		}
		if _, paramsErr := r.applyDispatchRequestParams(reqCopy, ctx); paramsErr != nil {
			return nil, fallback.EvaluationResult{CanFallback: true}, paramsErr
		}
	}

	engine, engineErr := r.protocolEngine()
	if engineErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, engineErr
	}

	requestBody, err := r.prepareFallbackCandidateBody(engine, reqCopy, dispatch, ctx)
	if err == nil {
		err = r.rejectDispatchCapabilityMismatch(reqCopy, dispatch, ctx)
	}
	if err != nil {
		return nil, fallback.EvaluationResult{CanFallback: !isContextExpired(callCtx)}, err
	}
	if rateLimitResponse := r.applyRateLimit(ctx, candidateModel); rateLimitResponse != nil {
		logging.ComponentEvent("extproc", "fallback_candidate_rate_limited", map[string]interface{}{
			"request_id":      ctx.RequestID,
			"candidate_model": candidateModel,
		})
		return nil, fallback.EvaluationResult{
			CanFallback: true,
			Reason:      "candidate_rate_limited",
		}, fmt.Errorf("fallback candidate %q rejected by rate limit", candidateModel)
	}
	// The primary's reservation ends here; a candidate the budget cannot
	// cover gives way to the next, which may cost less.
	if refusal := r.reserveDispatchSpend(ctx, candidateModel, reqCopy); refusal != nil {
		return nil, fallback.EvaluationResult{
			CanFallback: true,
			Reason:      "candidate_" + refusal.code,
		}, fmt.Errorf("fallback candidate %q refused by the spend budget: %s", candidateModel, refusal.message)
	}

	return &fallbackCandidate{
		model: candidateModel, ref: candidateRef, dispatch: dispatch, request: reqCopy, body: requestBody,
		engine: engine,
	}, fallback.EvaluationResult{CanFallback: true}, nil
}

func (r *OpenAIRouter) prepareFallbackCandidateBody(engine *protocolcodec.Engine, request *llmprotocol.Request, dispatch *providerDispatch, ctx *RequestContext) ([]byte, error) {
	diagnostics := len(ctx.ProtocolDiagnostics)
	previous := ctx.projectionObservations
	var observations []func()
	ctx.projectionObservations = &observations
	defer func() {
		ctx.projectionObservations = previous
		for _, record := range observations {
			ctx.observeProjection(record)
		}
	}()

	body, admission, err := r.encodeAndAdmitFallbackBody(engine, request, dispatch, ctx)
	if request.Sampling.AutomaticOutput && isFinalContextOverflow(err) {
		err = r.compressAutomaticOverflow(ctx, request, dispatch, err)
		if err == nil {
			ctx.ProtocolDiagnostics = ctx.ProtocolDiagnostics[:diagnostics]
			observations = nil
			body, admission, err = r.encodeAndAdmitFallbackBody(engine, request, dispatch, ctx)
		}
	}
	if admission.Source != "" {
		metrics.RecordFinalAdmission(admission.Source, admission.Verdict)
	}
	return body, err
}

func (r *OpenAIRouter) encodeAndAdmitFallbackBody(engine *protocolcodec.Engine, request *llmprotocol.Request, dispatch *providerDispatch, ctx *RequestContext) ([]byte, dispatchAdmission, error) {
	dispatchRequest := *request
	if err := r.applyPromptCachePolicy(&dispatchRequest, ctx, dispatch.targetFormat); err != nil {
		return nil, dispatchAdmission{}, err
	}
	projected, diagnostics, err := r.projectRequestForBackendWithDiagnostics(dispatchRequest, dispatch.effectiveBackendModel(), dispatch.targetFormat)
	if err != nil {
		return nil, dispatchAdmission{}, err
	}
	requestBackendStreamUsage(dispatch.targetFormat, &projected)
	r.requestUnstoredOnCallerLogin(dispatch.targetFormat, &projected)
	encoded, err := engine.EncodeRequest(dispatch.targetFormat, projected, ctx.ProtocolEnvelope)
	if err != nil {
		return nil, dispatchAdmission{}, err
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, encoded.Diagnostics...)

	body, err := r.adaptProviderRequest(encoded.Body, dispatch, ctx)
	if err != nil {
		return nil, dispatchAdmission{}, err
	}
	if len(body) == 0 {
		body = encoded.Body
	}
	return r.admitProviderBody(body, request, dispatch, ctx)
}

func (r *OpenAIRouter) settleFallbackCandidateUsage(ctx *RequestContext, attemptDuration time.Duration) {
	if r == nil || ctx == nil {
		return
	}
	completionLatency := attemptDuration
	if !ctx.StartTime.IsZero() {
		completionLatency = time.Since(ctx.StartTime)
	}
	// The answer replaces the failed primary's response headers, so the
	// candidate's own call time is its first response time.
	recordNonStreamingFirstResponse(ctx, attemptDuration)
	usage := r.takeNeutralResponseUsage(ctx)
	r.reportNonStreamingUsage(ctx, completionLatency, usage)
	r.calibrateTokenEstimator(ctx, usage.promptTokens)

	if ctx.FallbackRecord == nil || len(ctx.FallbackRecord.Attempts) == 0 {
		return
	}
	servedAttempt := &ctx.FallbackRecord.Attempts[len(ctx.FallbackRecord.Attempts)-1]
	if ctx.RequestCostPriced {
		servedAttempt.Cost = ctx.RequestCost
		servedAttempt.Currency = ctx.RequestCostCurrency
	}
	fallback.CorrelateExecution(ctx.FallbackRecord)
}

// fallbackProviderAuthorizer resolves provider credentials for user-request fallback attempts,
// preserving request-scoped headers (per-user/injected API keys) alongside static configuration.
func (r *OpenAIRouter) fallbackProviderAuthorizer(
	reqCtx *RequestContext,
	profile *config.ProviderProfile,
	model string,
) (func(context.Context, *http.Request) error, error) {
	provider, providerAuth, err := resolveProviderAuth(profile)
	if err != nil {
		return nil, err
	}
	if profile.Caller() {
		// The same contract as primary dispatch: only the client's own login,
		// and the attempt fails before the call without it.
		present, err := callerCredentialAuthorizer(reqCtx, profile)
		if err != nil {
			return nil, err
		}
		return func(_ context.Context, request *http.Request) error {
			present(request)
			return nil
		}, nil
	}
	if profile.Delegated() {
		// The same contract as primary dispatch: only the request's delegated
		// credential, and the attempt fails before the call without it.
		credential, ok := requestDelegatedCredential(reqCtx)
		if !ok {
			return nil, principal.ErrCredentialMissing
		}
		if providerAuth.Strategy == "none" || providerAuth.Header == "" {
			return nil, fmt.Errorf("delegated backend has no auth header to carry the credential")
		}
		return func(_ context.Context, request *http.Request) error {
			value := credential
			if providerAuth.Prefix != "" {
				value = providerAuth.Prefix + " " + credential
			}
			request.Header.Set(providerAuth.Header, value)
			return nil
		}, nil
	}
	if providerAuth.Strategy == "none" {
		return nil, nil
	}
	var headers map[string]string
	if reqCtx != nil {
		headers = reqCtx.Headers
	}
	return func(_ context.Context, request *http.Request) error {
		var accessKey string
		if r != nil && r.CredentialResolver != nil {
			var keyErr error
			accessKey, keyErr = r.CredentialResolver.KeyForProvider(provider, model, headers)
			if keyErr != nil {
				return fmt.Errorf("resolve credential for provider %s: %w", provider, keyErr)
			}
		} else if r != nil && r.Config != nil {
			accessKey = authz.NewStaticConfigProvider(r.Config).GetKey(provider, model, headers)
		}
		if accessKey == "" {
			return nil
		}
		if providerAuth.Prefix != "" {
			accessKey = providerAuth.Prefix + " " + accessKey
		}
		request.Header.Set(providerAuth.Header, accessKey)
		return nil
	}, nil
}

// dispatchFallbackHTTP issues a network request to the candidate backend via the connector.
func (r *OpenAIRouter) dispatchFallbackHTTP(
	ctx context.Context,
	reqCtx *RequestContext,
	dispatch *providerDispatch,
	body []byte,
	orch *fallback.Orchestrator,
) ([]byte, int, error) {
	timeout := 30 * time.Second
	if orch != nil && orch.Policy().PerAttemptTimeout > 0 {
		timeout = orch.Policy().PerAttemptTimeout
	}
	return r.dispatchProviderHTTP(ctx, reqCtx, dispatch, providerCall{
		name: "fallback_dispatch", body: body,
		headers: fallbackRequestHeaders(reqCtx, dispatch.profile, orch), timeout: timeout,
	})
}

// providerCall is a call the Router sends a backend itself, besides the
// credential the connector sets last.
type providerCall struct {
	name    string
	body    []byte
	headers map[string]string
	timeout time.Duration
}

// dispatchProviderHTTP sends call to dispatch's backend through the
// connector, with the request's credential for that backend, within
// call.timeout and ctx's deadline.
func (r *OpenAIRouter) dispatchProviderHTTP(
	ctx context.Context,
	reqCtx *RequestContext,
	dispatch *providerDispatch,
	call providerCall,
) ([]byte, int, error) {
	endpoint, query, err := splitProviderEndpoint(providerEndpointPath(dispatch.profile, dispatch.targetFormat))
	if err != nil {
		return nil, 0, err
	}
	authorize, err := r.fallbackProviderAuthorizer(reqCtx, dispatch.profile, dispatch.effectiveBackendModel())
	if err != nil {
		return nil, 0, err
	}
	base := providerEndpointScheme(r.Config, dispatch.backendName, dispatch.profile) + "://" + dispatch.backendAddress
	timeout := call.timeout
	if deadline, ok := ctx.Deadline(); ok {
		if remaining := time.Until(deadline); remaining > 0 && remaining < timeout {
			timeout = remaining
		}
	}
	tlsConfig, err := providerTLSConfig(dispatch.profile)
	if err != nil {
		return nil, 0, err
	}
	client, err := connector.New(base, authorize, connector.Options{
		AttemptTimeout:   timeout,
		MaxRequestBytes:  16 << 20,
		MaxResponseBytes: 32 << 20,
		MaxErrorBytes:    8 << 10,
		TLSConfig:        tlsConfig,
	})
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = client.Close() }()

	if ctx == nil {
		ctx = context.Background()
	}
	operation := connector.Operation{
		Name:              call.name,
		Method:            http.MethodPost,
		Path:              endpoint,
		Query:             query,
		SuccessStatusCode: http.StatusOK,
	}
	resp, err := client.DoRequest(ctx, operation, connector.Request{
		Body:    call.body,
		Headers: call.headers,
	})
	if err != nil {
		var connErr *connector.Error
		if errors.As(err, &connErr) && connErr.Kind == connector.KindStatus {
			errBody, _ := connErr.ResponseBody()
			return redactCallerCredential(reqCtx, errBody), connErr.StatusCode, err
		}
		return nil, 0, err
	}
	return resp.Body, resp.StatusCode, nil
}

// buildFallbackImmediateResponse constructs the terminal HTTP 200 response. It
// carries the public set of any routed response, as the native gateway's
// fallback answer does through the response phases; the trace counts the
// attempts it took.
func (r *OpenAIRouter) buildFallbackImmediateResponse(
	responseBody []byte,
	contentType string,
	ctx *RequestContext,
	extraHeaders ...*core.HeaderValueOption,
) *ext_proc.ProcessingResponse {
	if contentType == "" {
		contentType = "application/json"
	}
	builder := newResponseHeaderMutationBuilder()
	builder.addString("content-type", contentType)
	builder.addKeystone(ctx)
	builder.addPublicRoutingHeaders(ctx)
	builder.addWarnings(ctx)
	setHeaders := builder.setHeaders
	if len(extraHeaders) > 0 {
		setHeaders = append(setHeaders, extraHeaders...)
	}
	if ctx != nil && ctx.RequestCostPriced {
		hasCost := false
		for _, h := range extraHeaders {
			if h != nil && h.GetHeader() != nil && h.GetHeader().GetKey() == headers.VSRCost {
				hasCost = true
				break
			}
		}
		if !hasCost {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRCost,
					RawValue: []byte(strconv.FormatFloat(ctx.RequestCost, 'f', -1, 64)),
				},
			})
			if ctx.RequestCostCurrency != "" {
				setHeaders = append(setHeaders, &core.HeaderValueOption{
					Header: &core.HeaderValue{
						Key:      headers.VSRCostCurrency,
						RawValue: []byte(ctx.RequestCostCurrency),
					},
				})
			}
		}
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: typev3.StatusCode_OK,
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: setHeaders,
				},
				Body: responseBody,
			},
		},
	}
}

// fallbackOutcomeSource is the source of the outcome the fallback audit
// appends to a request's replay record.
const fallbackOutcomeSource = "fallback"

// recordFallbackExecution persists one bounded terminal audit record containing
// every correlated attempt. It runs for success, exhaustion, cancellation, and
// policy rejection so replay consumers never have to infer missing attempts.
// A record without attempts is a chain that never started, because the first
// model answered, so it leaves no outcome.
func (r *OpenAIRouter) recordFallbackExecution(ctx *RequestContext) {
	if ctx == nil || ctx.FallbackRecord == nil || len(ctx.FallbackRecord.Attempts) == 0 || ctx.FallbackAuditRecorded {
		return
	}

	record := *fallback.CorrelateExecution(ctx.FallbackRecord)
	record.Attempts = append([]fallback.AttemptOutcome(nil), record.Attempts...)
	for i := range record.Attempts {
		record.Attempts[i].ErrorMessage = boundedFallbackAuditText(record.Attempts[i].ErrorMessage, 1024)
	}
	executionJSON, err := json.Marshal(record)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "fallback_outcome_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      err.Error(),
		})
		return
	}

	selectedModel := record.SelectedModel
	if selectedModel == "" {
		selectedModel = record.InitialModel
	}
	verdict := "failed"
	switch record.FinalStatus {
	case "succeeded":
		verdict = "completed"
	case "policy_blocked":
		verdict = "blocked"
	}
	logging.ComponentEvent("extproc", "fallback_execution_complete", map[string]interface{}{
		"request_id":        record.RequestID,
		"session_id":        record.SessionID,
		"conversation_id":   record.ConversationID,
		"initial_model":     record.InitialModel,
		"selected_model":    record.SelectedModel,
		"attempts":          len(record.Attempts),
		"total_duration_ms": record.TotalDuration.Milliseconds(),
		"final_status":      record.FinalStatus,
		"usage":             record.UsageSummary,
		"execution_record":  string(executionJSON),
	})

	recorder := ctx.RouterReplayRecorder
	if recorder == nil && r != nil {
		recorder = r.ReplayRecorder
	}
	// Replay already records a primary response that ends the chain before
	// a candidate is tried.
	primaryResponseEndedChain := record.FinalStatus == "non_retryable" && len(record.Attempts) == 1 &&
		!record.Attempts[0].LocalReply && record.Attempts[0].StatusCode >= http.StatusMultipleChoices
	if recorder == nil || ctx.RouterReplayID == "" || primaryResponseEndedChain {
		ctx.FallbackAuditRecorded = true
		return
	}
	metadata := map[string]string{
		"initial_model":          record.InitialModel,
		"fallback_model":         record.SelectedModel,
		"attempts":               strconv.Itoa(len(record.Attempts)),
		"final_status":           record.FinalStatus,
		"billable_total_tokens":  strconv.Itoa(record.UsageSummary.BillableTotalTokens),
		"discarded_total_tokens": strconv.Itoa(record.UsageSummary.DiscardedTotalTokens),
		"execution_record":       string(executionJSON),
	}
	outcome := routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    fallbackOutcomeSource,
		Target:    "request",
		TargetRef: selectedModel,
		Verdict:   verdict,
		Reason:    record.FinalStatus,
		Metadata:  metadata,
	}
	if err := recorder.AppendOutcome(ctx.RouterReplayID, outcome); err != nil {
		logging.ComponentErrorEvent("extproc", "fallback_outcome_persist_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
		return
	}
	ctx.FallbackAuditRecorded = true
}

func boundedFallbackAuditText(value string, maxRunes int) string {
	if maxRunes <= 0 {
		return ""
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[:maxRunes])
}

// rollbackFallbackContext restores RequestContext to the primary attempt's state
// when a candidate fails finalized stream encoding or wire validation.
func (r *OpenAIRouter) rollbackFallbackContext(
	ctx *RequestContext,
	primaryModel string,
	primaryStatusCode int,
	origPath string,
	origDiags []llmprotocol.Diagnostic,
	wasStreaming bool,
	origTargetFormat llmprotocol.WireFormat,
) {
	if ctx == nil {
		return
	}
	ctx.RequestModel = primaryModel
	ctx.VSRSelectedModel = primaryModel
	ctx.ResponsePath = origPath
	ctx.UpstreamStatusCode = primaryStatusCode
	ctx.TargetFormat = origTargetFormat
	ctx.ProtocolDiagnostics = origDiags
	ctx.IsStreamingResponse = wasStreaming
	ctx.ResponseVendor = ""
	ctx.SemanticResponse = nil
	ctx.ResponseEnvelope = llmprotocol.Envelope{}
	ctx.ResponseUnknownFields = false
	ctx.RequestCostPriced = false
	ctx.RequestCost = 0
	ctx.RequestCostCurrency = ""
	ctx.ResponseGuard = nil
	ctx.PrimaryOutputDigest = ""
	ctx.PrimaryOutputChars = 0
	r.updateRouterReplayStatus(ctx, primaryStatusCode, wasStreaming)
}

// splitResponseTextContent slices large text contents in response output items so that each
// content item produces a stream event that fits comfortably within SSE frame limits.
func splitResponseTextContent(resp llmprotocol.Response, maxChunkSize int) llmprotocol.Response {
	if maxChunkSize <= 0 {
		maxChunkSize = 64 * 1024
	}
	cloned := resp
	cloned.Output = splitOutputItemsTextContent(resp.Output, maxChunkSize)
	if len(resp.Alternatives) > 0 {
		cloned.Alternatives = make([][]llmprotocol.OutputItem, len(resp.Alternatives))
		for i, alt := range resp.Alternatives {
			cloned.Alternatives[i] = splitOutputItemsTextContent(alt, maxChunkSize)
		}
	}
	return cloned
}

func splitOutputItemsTextContent(items []llmprotocol.OutputItem, maxChunkSize int) []llmprotocol.OutputItem {
	newItems := make([]llmprotocol.OutputItem, len(items))
	for i, item := range items {
		itemCopy := item
		var newContents []llmprotocol.Content
		for _, c := range item.Content {
			if (c.Kind == llmprotocol.ContentText || c.Kind == llmprotocol.ContentRefusal) && len(c.Text) > maxChunkSize {
				chunks := splitStringToChunks(c.Text, maxChunkSize)
				for _, chunk := range chunks {
					chunkContent := c
					chunkContent.Text = chunk
					newContents = append(newContents, chunkContent)
				}
			} else {
				newContents = append(newContents, c)
			}
		}
		itemCopy.Content = newContents
		newItems[i] = itemCopy
	}
	return newItems
}

func splitStringToChunks(s string, maxChunkSize int) []string {
	if maxChunkSize <= 0 || len(s) <= maxChunkSize {
		return []string{s}
	}
	var chunks []string
	for len(s) > maxChunkSize {
		cut := maxChunkSize
		for cut > 0 && !utf8.RuneStart(s[cut]) {
			cut--
		}
		if cut == 0 {
			_, size := utf8.DecodeRuneInString(s)
			cut = size
		}
		chunks = append(chunks, s[:cut])
		s = s[cut:]
	}
	if len(s) > 0 {
		chunks = append(chunks, s)
	}
	return chunks
}

func isContextExpired(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	if ctx.Err() != nil {
		return true
	}
	if deadline, ok := ctx.Deadline(); ok && !deadline.After(time.Now()) {
		return true
	}
	return false
}
