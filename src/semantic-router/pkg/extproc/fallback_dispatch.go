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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

type fallbackTransportCaller func(ctx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error)

// hasNonIdempotentToolExecution reports whether the request contains executed tool side-effects.
func (ctx *RequestContext) hasNonIdempotentToolExecution() bool {
	if ctx == nil || ctx.SemanticRequest == nil {
		return false
	}
	for _, msg := range ctx.SemanticRequest.Messages {
		if msg.Role == llmprotocol.RoleTool {
			return true
		}
		for _, content := range msg.Content {
			if content.Kind == llmprotocol.ContentToolResult {
				return true
			}
		}
	}
	return false
}

// isClientStreamingRequested reports whether the original downstream client requested a streaming response.
func isClientStreamingRequested(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	return ctx.ExpectStreamingResponse || (ctx.SemanticRequest != nil && ctx.SemanticRequest.Stream)
}

// candidateReasoningChoice resolves the effective reasoning policy for a fallback candidate.
func (r *OpenAIRouter) candidateReasoningChoice(ctx *RequestContext, candidateModel string) bool {
	if ctx == nil {
		return false
	}
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
	if r == nil || ctx == nil {
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
	backendName := primaryModel
	if dispatch, err := r.resolveProviderDispatch(primaryModel, ctx.VSRSelectedDecisionName, false); err == nil && dispatch != nil {
		backendName = dispatch.backendName
	}
	orch.CircuitBreaker().RecordSuccess(backendName)
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
	if ctx.LooperRequest || ctx.ResponseHeadersContinued {
		return false
	}
	return len(ctx.VSREligibleModelRefs) > 1
}

// maybeExecuteFallback evaluates the upstream failure and, if eligible, coordinates
// bounded fallback across candidate models from the decision's hard-eligible list.
func (r *OpenAIRouter) maybeExecuteFallback(body []byte, ctx *RequestContext) *ext_proc.ProcessingResponse {
	orch := r.fallbackOrchestratorForContext(ctx)
	if !r.shouldAttemptFallback(ctx) || orch == nil {
		return nil
	}

	commit := fallback.CommitState{
		ResponseCommitted:           ctx.ResponseHeadersContinued || ctx.StreamingComplete || (ctx.IsStreamingResponse && ctx.TTFTRecorded),
		HasNonIdempotentSideEffects: ctx.hasNonIdempotentToolExecution(),
		BodyReplayable:              ctx.SemanticRequest != nil,
	}

	primaryModel := ctx.VSRSelectedModel
	if primaryModel == "" {
		primaryModel = ctx.RequestModel
	}
	backendName := primaryModel
	if dispatch, err := r.resolveProviderDispatch(primaryModel, ctx.VSRSelectedDecisionName, false); err == nil && dispatch != nil {
		backendName = dispatch.backendName
	}

	if ctx.FallbackRecord == nil {
		ctx.FallbackRecord = orch.NewExecutionRecord(ctx.RequestID, ctx.VSRSelectedDecisionName, primaryModel)
		ctx.FallbackRecord.SessionID = ctx.SessionID
		if ctx.SemanticRequest != nil {
			ctx.FallbackRecord.ConversationID = ctx.SemanticRequest.ConversationID
		}
		if ctx.FallbackRecord.ConversationID == "" && ctx.ResponseObjectState != nil {
			ctx.FallbackRecord.ConversationID = ctx.ResponseObjectState.ConversationID
		}
	}
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
			return nil
		}
		nextCandidate := &ctx.VSREligibleModelRefs[candIdx]

		resp, candidateEval, attemptErr := r.executeFallbackCandidate(callCtx, nextCandidate, commit, ctx, orch)
		if attemptErr == nil && resp != nil {
			return resp
		}

		if !candidateEval.CanFallback {
			if ctx.FallbackRecord.FinalStatus == "" {
				switch {
				case errors.Is(attemptErr, context.DeadlineExceeded):
					ctx.FallbackRecord.FinalStatus = "total_deadline_exceeded"
				case errors.Is(attemptErr, context.Canceled):
					ctx.FallbackRecord.FinalStatus = "context_canceled"
				default:
					ctx.FallbackRecord.FinalStatus = "fallback_halted"
				}
			}
			logging.ComponentDebugEvent("extproc", "fallback_loop_halted", map[string]interface{}{
				"request_id":   ctx.RequestID,
				"final_status": ctx.FallbackRecord.FinalStatus,
				"reason":       candidateEval.Reason,
			})
			break
		}
	}

	return nil
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
	var candidateSucceeded bool
	defer func() {
		ctx.TraceContext = origTraceContext
		if !candidateSucceeded {
			ctx.VSRSelectedCandidate = origSelectedCandidate
			ctx.SemanticRequest = origSemanticRequest
			ctx.RateLimitCtx = origRateLimitCtx
		}
	}()
	candidateModel := candidateModelIdentity(*candidateRef)
	primaryModel := ctx.FallbackRecord.InitialModel
	primaryStatusCode := ctx.UpstreamStatusCode
	origPath := ctx.ResponsePath
	origDiags := ctx.ProtocolDiagnostics
	wasStreaming := ctx.IsStreamingResponse
	useReasoning := r.candidateReasoningChoice(ctx, candidateModel)
	dispatch, err := r.resolveProviderDispatch(candidateModel, ctx.VSRSelectedDecisionName, useReasoning)
	if err != nil && candidateRef.LoRAName != "" && candidateRef.Model != "" {
		if baseDispatch, baseErr := r.resolveProviderDispatch(candidateRef.Model, ctx.VSRSelectedDecisionName, useReasoning); baseErr == nil {
			dispatch = baseDispatch
			dispatch.logicalModel = candidateModel
			dispatch.upstreamModel = r.Config.ResolveExternalModelID(candidateModel, baseDispatch.backendName)
			err = nil
		}
	}
	if err != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, err
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
	reqCopy.Stream = false // Fallback response is buffered
	if useReasoning {
		effort := candidateRef.ReasoningEffort
		if effort == "" {
			effort = r.getReasoningEffort(ctx.decisionForCandidate(candidateModel), candidateModel)
		}
		reqCopy.ReasoningEffort = effort
		mode := candidateRef.ReasoningMode
		if mode == "" {
			mode = r.getReasoningMode(ctx.decisionForCandidate(candidateModel), candidateModel, true)
		}
		reqCopy.ReasoningMode = llmprotocol.ReasoningMode(mode)
	} else {
		reqCopy.ReasoningEffort = ""
		reqCopy.ReasoningMode = ""
		reqCopy.ReasoningBudgetTokens = nil
	}

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

	if err := r.prepareAutomaticDispatch(ctx, reqCopy, dispatch); err != nil {
		if isContextExpired(callCtx) {
			return nil, fallback.EvaluationResult{CanFallback: false}, err
		}
		return nil, fallback.EvaluationResult{CanFallback: true}, err
	}
	if protocolErr := r.rejectDispatchCapabilityMismatch(reqCopy, dispatch, ctx); protocolErr != nil {
		if isContextExpired(callCtx) {
			return nil, fallback.EvaluationResult{CanFallback: false}, protocolErr
		}
		return nil, fallback.EvaluationResult{CanFallback: true}, protocolErr
	}

	engine, engineErr := r.protocolEngine()
	if engineErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, engineErr
	}

	encoded, encodeErr := engine.EncodeRequest(dispatch.targetFormat, *reqCopy, ctx.ProtocolEnvelope)
	if encodeErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, encodeErr
	}

	requestBody := encoded.Body
	adaptedBody, adaptErr := r.adaptProviderRequest(encoded.Body, dispatch, ctx)
	if adaptErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, adaptErr
	}
	if len(adaptedBody) > 0 {
		requestBody = adaptedBody
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

	var resBody []byte
	var statusCode int
	var callErr error
	attemptStart := time.Now()

	if r.fallbackCaller != nil {
		extraHeaders := make(map[string]string)
		if dispatch.profile != nil {
			for k, v := range dispatch.profile.ExtraHeaders {
				extraHeaders[k] = v
			}
		}
		resBody, statusCode, callErr = r.fallbackCaller(attemptCtx, candidateModel, requestBody, extraHeaders)
	} else {
		resBody, statusCode, callErr = r.dispatchFallbackHTTP(attemptCtx, ctx, dispatch, requestBody, orch)
	}
	attemptDuration := time.Since(attemptStart)

	attemptOutcome := fallback.AttemptOutcome{
		Model:      candidateModel,
		Backend:    dispatch.backendName,
		StatusCode: statusCode,
		Error:      callErr,
		Duration:   attemptDuration,
		Discarded:  callErr != nil || statusCode < 200 || statusCode >= 300,
	}

	if callErr != nil || statusCode < 200 || statusCode >= 300 {
		attemptOutcome.ErrorMessage = string(resBody)
		evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
		logging.ComponentWarnEvent("extproc", "fallback_candidate_attempt_failed", map[string]interface{}{
			"request_id":      ctx.RequestID,
			"candidate_model": candidateModel,
			"status":          statusCode,
			"error":           callErr,
			"can_fallback":    evalResult.CanFallback,
			"final_status":    ctx.FallbackRecord.FinalStatus,
		})
		return nil, evalResult, fmt.Errorf("candidate %s failed with status %d: %w", candidateModel, statusCode, callErr)
	}

	responseEngine, engineErr := r.protocolEngineForVendor(resolveResponseVendor(dispatch.profile))
	if engineErr != nil {
		attemptOutcome.Error = engineErr
		attemptOutcome.Discarded = true
		evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
		return nil, evalResult, engineErr
	}

	var mutation protocolcodec.ResponseMutation
	if responseID := responseObjectPublicID(ctx); responseID != "" {
		mutation = func(response *llmprotocol.Response) error {
			response.ID = responseID
			return nil
		}
	}

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
			Context:            ctx.TraceContext,
			Source:             format,
			Target:             format,
			Options:            clientStreamOptions(ctx),
			PublicModel:        candidateModel,
			ResponseID:         candidateResponse.ID,
			PreviousResponseID: responseObjectPreviousID(ctx),
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
		inflight.End(primaryModel, ctx.InflightToken)
		ctx.InflightToken = 0
	}

	ctx.RequestModel = candidateModel
	ctx.VSRSelectedModel = candidateModel
	ctx.ResponsePath = headers.ResponsePathFallback
	ctx.UpstreamStatusCode = statusCode
	ctx.ResponseVendor = resolveResponseVendor(dispatch.profile)
	ctx.SemanticResponse = &candidateResponse
	ctx.SemanticRequest = reqCopy
	ctx.ResponseEnvelope = translated.Envelope
	ctx.ResponseVendorExtensions = protocolcodec.DiagnosticsDroppedVendorExtensions(translated.Diagnostics)

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
		blockedStatus := http.StatusForbidden
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
		r.updateRouterReplayStatus(ctx, blockedStatus, false)
		candidateSucceeded = true
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
				Context:            ctx.TraceContext,
				Source:             format,
				Target:             format,
				Options:            clientStreamOptions(ctx),
				PublicModel:        candidateModel,
				ResponseID:         ctx.SemanticResponse.ID,
				PreviousResponseID: responseObjectPreviousID(ctx),
			})
			if streamErr == nil {
				if validateErr := engine.ValidateEncodedResponse(reencodedStream, true); validateErr == nil {
					responseBody = reencodedStream
				} else {
					// The single SSE text frame exceeded Limits.SSEFrameBytes (e.g. policy warning pushed it over limit).
					// Produce a bounded stream containing the policy result by chunking text across frames within SSE limits.
					boundedResp := splitResponseTextContent(streamResp, 64*1024)
					boundedStream, _, boundedErr := engine.EncodeResponseStream(format, boundedResp, llmprotocol.StreamContext{
						Context:            ctx.TraceContext,
						Source:             format,
						Target:             format,
						Options:            clientStreamOptions(ctx),
						PublicModel:        candidateModel,
						ResponseID:         ctx.SemanticResponse.ID,
						PreviousResponseID: responseObjectPreviousID(ctx),
					})
					if boundedErr == nil && engine.ValidateEncodedResponse(boundedStream, true) == nil {
						responseBody = boundedStream
					} else {
						attemptOutcome.Error = validateErr
						attemptOutcome.Discarded = true
						r.rollbackFallbackContext(ctx, primaryModel, primaryStatusCode, origPath, origDiags, wasStreaming)
						evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
						return nil, evalResult, fmt.Errorf("finalized fallback stream validation failed: %w", validateErr)
					}
				}
			} else {
				attemptOutcome.Error = streamErr
				attemptOutcome.Discarded = true
				r.rollbackFallbackContext(ctx, primaryModel, primaryStatusCode, origPath, origDiags, wasStreaming)
				evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
				return nil, evalResult, fmt.Errorf("finalized fallback stream encoding failed: %w", streamErr)
			}
		}
	} else {
		if validateErr := engine.ValidateEncodedResponse(finalBody, false); validateErr != nil {
			attemptOutcome.Error = validateErr
			attemptOutcome.Discarded = true
			r.rollbackFallbackContext(ctx, primaryModel, primaryStatusCode, origPath, origDiags, wasStreaming)
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

func (r *OpenAIRouter) settleFallbackCandidateUsage(ctx *RequestContext, attemptDuration time.Duration) {
	if r == nil || ctx == nil {
		return
	}
	completionLatency := attemptDuration
	if !ctx.StartTime.IsZero() {
		completionLatency = time.Since(ctx.StartTime)
	}
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
	endpoint, query, err := splitProviderEndpoint(providerEndpointPath(dispatch.profile, dispatch.targetFormat))
	if err != nil {
		return nil, 0, err
	}
	authorize, err := r.fallbackProviderAuthorizer(reqCtx, dispatch.profile, dispatch.logicalModel)
	if err != nil {
		return nil, 0, err
	}
	base := providerEndpointScheme(r.Config, dispatch.backendName, dispatch.profile) + "://" + dispatch.backendAddress
	timeout := 30 * time.Second
	if orch != nil && orch.Policy().PerAttemptTimeout > 0 {
		timeout = orch.Policy().PerAttemptTimeout
	}
	if deadline, ok := ctx.Deadline(); ok {
		if remaining := time.Until(deadline); remaining > 0 && remaining < timeout {
			timeout = remaining
		}
	}
	client, err := connector.New(base, authorize, connector.Options{
		AttemptTimeout:   timeout,
		MaxRequestBytes:  16 << 20,
		MaxResponseBytes: 32 << 20,
		MaxErrorBytes:    8 << 10,
	})
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = client.Close() }()

	if ctx == nil {
		ctx = context.Background()
	}
	operation := connector.Operation{
		Name:              "fallback_dispatch",
		Method:            http.MethodPost,
		Path:              endpoint,
		Query:             query,
		SuccessStatusCode: http.StatusOK,
	}
	reqHeaders := make(map[string]string)
	if dispatch.profile != nil {
		for k, v := range dispatch.profile.ExtraHeaders {
			reqHeaders[k] = v
		}
	}
	resp, err := client.DoRequest(ctx, operation, connector.Request{
		Body:    body,
		Headers: reqHeaders,
	})
	if err != nil {
		var connErr *connector.Error
		if errors.As(err, &connErr) && connErr.Kind == connector.KindStatus {
			errBody, _ := connErr.ResponseBody()
			return errBody, connErr.StatusCode, err
		}
		return nil, 0, err
	}
	return resp.Body, resp.StatusCode, nil
}

// buildFallbackImmediateResponse constructs the terminal HTTP 200 response with observability headers.
func (r *OpenAIRouter) buildFallbackImmediateResponse(
	responseBody []byte,
	contentType string,
	ctx *RequestContext,
	extraHeaders ...*core.HeaderValueOption,
) *ext_proc.ProcessingResponse {
	attemptsCount := "1"
	if ctx.FallbackRecord != nil {
		attemptsCount = strconv.Itoa(len(ctx.FallbackRecord.Attempts))
	}
	if contentType == "" {
		contentType = "application/json"
	}
	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      "content-type",
				RawValue: []byte(contentType),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      headers.VSRSelectedModel,
				RawValue: []byte(ctx.VSRSelectedModel),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      headers.VSRFallbackAttempts,
				RawValue: []byte(attemptsCount),
			},
		},
	}
	setHeaders = append(setHeaders, httputil.KeystoneHeaderOptions(ctx.ResponsePath)...)
	if ctx.RouterReplayID != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.RouterReplayID,
				RawValue: []byte(ctx.RouterReplayID),
			},
		})
	}
	if ctx != nil && !ctx.SkipProcessing && !ctx.LooperRequest && !ctx.VSRCacheHit && ctx.SemanticRequest != nil {
		sampling := ctx.SemanticRequest.Sampling
		if sampling.AutomaticOutput && sampling.AutomaticInputTokens != nil && sampling.MaxOutputTokens != nil &&
			*sampling.AutomaticInputTokens > 0 && *sampling.MaxOutputTokens > 0 {
			setHeaders = append(setHeaders,
				&core.HeaderValueOption{
					Header: &core.HeaderValue{
						Key:      headers.VSREffectiveInputTokens,
						RawValue: []byte(strconv.FormatInt(*sampling.AutomaticInputTokens, 10)),
					},
				},
				&core.HeaderValueOption{
					Header: &core.HeaderValue{
						Key:      headers.VSREffectiveMaxOutputTokens,
						RawValue: []byte(strconv.FormatInt(*sampling.MaxOutputTokens, 10)),
					},
				},
			)
		}
	}
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

// recordFallbackExecution persists one bounded terminal audit record containing
// every correlated attempt. It runs for success, exhaustion, cancellation, and
// policy rejection so replay consumers never have to infer missing attempts.
func (r *OpenAIRouter) recordFallbackExecution(ctx *RequestContext) {
	if ctx == nil || ctx.FallbackRecord == nil || ctx.FallbackAuditRecorded {
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
	if recorder == nil || ctx.RouterReplayID == "" {
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
		Source:    "fallback",
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
) {
	if ctx == nil {
		return
	}
	ctx.RequestModel = primaryModel
	ctx.VSRSelectedModel = primaryModel
	ctx.ResponsePath = origPath
	ctx.UpstreamStatusCode = primaryStatusCode
	ctx.ProtocolDiagnostics = origDiags
	ctx.IsStreamingResponse = wasStreaming
	ctx.ResponseVendor = ""
	ctx.SemanticResponse = nil
	ctx.ResponseEnvelope = llmprotocol.Envelope{}
	ctx.ResponseVendorExtensions = false
	ctx.RequestCostPriced = false
	ctx.RequestCost = 0
	ctx.RequestCostCurrency = ""
	ctx.HallucinationDetected = false
	ctx.HallucinationConfidence = 0
	ctx.HallucinationScoreAvailable = false
	ctx.HallucinationScoreKind = ""
	ctx.HallucinationSpans = nil
	ctx.EnhancedHallucinationInfo = nil
	ctx.UnverifiedFactualResponse = false
	ctx.ResponseJailbreakDetected = false
	ctx.ResponseJailbreakType = ""
	ctx.ResponseJailbreakDecision = nil
	ctx.ResponseJailbreakConfidence = 0
	ctx.ResponseJailbreakScoreAvailable = false
	ctx.VSRMatchedResponseJailbreak = nil
	ctx.VSRResponseJailbreakType = ""
	ctx.VSRResponseJailbreakRisk = 0
	ctx.VSRResponseJailbreakScoreAvailable = false
	ctx.VSRResponseJailbreakDecision = nil
	ctx.VSRMatchedHallucination = nil
	ctx.VSRHallucinationEvidence = nil
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
