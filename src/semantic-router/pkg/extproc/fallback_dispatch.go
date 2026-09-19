package extproc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
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

// recordPrimarySuccess resets consecutive failures for the primary backend when an attempt succeeds.
func (r *OpenAIRouter) recordPrimarySuccess(ctx *RequestContext) {
	if r == nil || ctx == nil {
		return
	}
	orch := r.fallbackOrchestratorForContext(ctx)
	if orch == nil || orch.CircuitBreaker() == nil {
		return
	}
	primaryModel := ctx.RequestModel
	if primaryModel == "" {
		primaryModel = ctx.VSRSelectedModel
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

	primaryModel := ctx.RequestModel
	if primaryModel == "" {
		primaryModel = ctx.VSRSelectedModel
	}
	backendName := primaryModel
	if dispatch, err := r.resolveProviderDispatch(primaryModel, ctx.VSRSelectedDecisionName, false); err == nil && dispatch != nil {
		backendName = dispatch.backendName
	}

	if ctx.FallbackRecord == nil {
		ctx.FallbackRecord = orch.NewExecutionRecord(ctx.RequestID, ctx.VSRSelectedDecisionName, primaryModel)
	}

	var primaryDuration time.Duration
	if !ctx.ProcessingStartTime.IsZero() {
		primaryDuration = time.Since(ctx.ProcessingStartTime)
	}

	primaryErrorMsg := fmt.Sprintf("upstream returned status %d", ctx.UpstreamStatusCode)
	if len(body) > 0 {
		primaryErrorMsg = fmt.Sprintf("upstream returned status %d: %s", ctx.UpstreamStatusCode, string(body))
	}

	primaryOutcome := fallback.AttemptOutcome{
		Model:        primaryModel,
		Backend:      backendName,
		StatusCode:   ctx.UpstreamStatusCode,
		Error:        errors.New(primaryErrorMsg),
		ErrorMessage: string(body),
		Duration:     primaryDuration,
		Discarded:    true,
	}

	callCtx := ctx.TraceContext
	if callCtx == nil {
		callCtx = context.Background()
	}
	if orch.Policy().TotalTimeout > 0 {
		var elapsed time.Duration
		if !ctx.ProcessingStartTime.IsZero() {
			elapsed = time.Since(ctx.ProcessingStartTime)
		}
		remainingTotal := orch.Policy().TotalTimeout - elapsed
		if remainingTotal <= 0 {
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

	evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, primaryOutcome, commit)
	if !evalResult.CanFallback {
		logging.ComponentDebugEvent("extproc", "fallback_not_allowed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"reason":     evalResult.Reason,
			"error":      evalResult.Err,
		})
		return nil
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
			func(i int) string { return ctx.VSREligibleModelRefs[i].Model },
			func(model string) (string, error) {
				dispatch, err := r.resolveProviderDispatch(model, ctx.VSRSelectedDecisionName, false)
				if err != nil {
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

		resp, candidateEval, attemptErr := r.executeFallbackCandidate(callCtx, nextCandidate.Model, commit, ctx, orch)
		if attemptErr == nil && resp != nil {
			return resp
		}

		if !candidateEval.CanFallback {
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
	candidateModel string,
	commit fallback.CommitState,
	ctx *RequestContext,
	orch *fallback.Orchestrator,
) (*ext_proc.ProcessingResponse, fallback.EvaluationResult, error) {
	if orch == nil {
		orch = r.fallbackOrchestratorForContext(ctx)
	}
	primaryModel := ctx.FallbackRecord.InitialModel
	dispatch, err := r.resolveProviderDispatch(candidateModel, ctx.VSRSelectedDecisionName, false)
	if err != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, err
	}

	reqCopy := *ctx.SemanticRequest
	if ctx.SemanticRequest.Instructions != nil {
		reqCopy.Instructions = make([]llmprotocol.InstructionBlock, len(ctx.SemanticRequest.Instructions))
		copy(reqCopy.Instructions, ctx.SemanticRequest.Instructions)
	}
	if ctx.SemanticRequest.Messages != nil {
		reqCopy.Messages = make([]llmprotocol.Message, len(ctx.SemanticRequest.Messages))
		copy(reqCopy.Messages, ctx.SemanticRequest.Messages)
	}
	if ctx.SemanticRequest.Tools != nil {
		reqCopy.Tools = make([]llmprotocol.Tool, len(ctx.SemanticRequest.Tools))
		copy(reqCopy.Tools, ctx.SemanticRequest.Tools)
	}
	reqCopy.Model = dispatch.upstreamModel
	reqCopy.Stream = false // Fallback response is buffered

	if _, decisionErr := r.applyDispatchDecision(&reqCopy, dispatch, ctx); decisionErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, decisionErr
	}
	if _, paramsErr := r.applyDispatchRequestParams(&reqCopy, ctx); paramsErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, paramsErr
	}

	engine, engineErr := r.protocolEngine()
	if engineErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, engineErr
	}

	encoded, encodeErr := engine.EncodeRequest(dispatch.targetFormat, reqCopy, ctx.ProtocolEnvelope)
	if encodeErr != nil {
		return nil, fallback.EvaluationResult{CanFallback: true}, encodeErr
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
		resBody, statusCode, callErr = r.fallbackCaller(attemptCtx, candidateModel, encoded.Body, extraHeaders)
	} else {
		resBody, statusCode, callErr = r.dispatchFallbackHTTP(attemptCtx, ctx, dispatch, encoded.Body, orch)
	}
	attemptDuration := time.Since(attemptStart)

	attemptOutcome := fallback.AttemptOutcome{
		Model:        candidateModel,
		Backend:      dispatch.backendName,
		StatusCode:   statusCode,
		Error:        callErr,
		ErrorMessage: string(resBody),
		Duration:     attemptDuration,
		Discarded:    callErr != nil || statusCode < 200 || statusCode >= 300,
	}

	if callErr != nil || statusCode < 200 || statusCode >= 300 {
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

	ctx.RequestModel = candidateModel
	ctx.VSRSelectedModel = candidateModel
	ctx.ResponsePath = headers.ResponsePathFallback
	ctx.UpstreamStatusCode = statusCode
	ctx.ResponseVendor = resolveResponseVendor(dispatch.profile)
	ctx.SemanticResponse = &translated.Response
	ctx.ResponseEnvelope = translated.Envelope
	ctx.ResponseVendorExtensions = protocolcodec.DiagnosticsDroppedVendorExtensions(translated.Diagnostics)

	// Finalize through shared response policy: response signals, safety plugins (jailbreak,
	// hallucination), memory suppression, warnings, and replay/persistence.
	blocked, finalBody, policyHeaders := r.finalizeResponsePolicy(ctx, ctx.SemanticResponse, translated.Body)
	if blocked != nil {
		evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
		return blocked, evalResult, nil
	}

	var responseBody []byte
	contentType := "application/json"
	if isClientStreamingRequested(ctx) {
		format := ctx.SourceFormat
		if format == "" {
			format = llmprotocol.OpenAIChatV1
		}
		if ctx.SemanticResponse.Model == "" {
			ctx.SemanticResponse.Model = candidateModel
		}
		if publicID := responseObjectPublicID(ctx); publicID != "" {
			ctx.SemanticResponse.ID = publicID
		} else if ctx.SemanticResponse.ID == "" {
			ctx.SemanticResponse.ID = "chatcmpl-" + ctx.RequestID
		}
		streamBody, streamDiags, streamErr := engine.EncodeResponseStream(format, *ctx.SemanticResponse, llmprotocol.StreamContext{
			Context:            ctx.TraceContext,
			Source:             format,
			Target:             format,
			Options:            clientStreamOptions(ctx),
			PublicModel:        candidateModel,
			ResponseID:         ctx.SemanticResponse.ID,
			PreviousResponseID: responseObjectPreviousID(ctx),
		})
		if streamErr != nil {
			attemptOutcome.Error = streamErr
			attemptOutcome.Discarded = true
			evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
			return nil, evalResult, fmt.Errorf("fallback stream encoding failed: %w", streamErr)
		}
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, streamDiags...)
		responseBody = streamBody
		contentType = "text/event-stream"
		ctx.IsStreamingResponse = true
	} else {
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, translated.Diagnostics...)
		responseBody = finalBody
	}

	evalResult := orch.EvaluateAttempt(callCtx, ctx.FallbackRecord, attemptOutcome, commit)
	if !evalResult.Succeeded {
		return nil, evalResult, fmt.Errorf("fallback attempt evaluation: %s", evalResult.Reason)
	}

	metrics.RecordModelRouting(primaryModel, candidateModel)
	r.recordFallbackReplayOutcome(ctx, primaryModel, candidateModel)

	logging.ComponentEvent("extproc", "fallback_candidate_succeeded", map[string]interface{}{
		"request_id":      ctx.RequestID,
		"primary_model":   primaryModel,
		"candidate_model": candidateModel,
		"attempts":        len(ctx.FallbackRecord.Attempts),
	})

	immediateResp := r.buildFallbackImmediateResponse(responseBody, contentType, ctx, policyHeaders...)
	r.persistImmediateResponseObject(immediateResp, ctx)
	return immediateResp, evalResult, nil
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
	if len(extraHeaders) > 0 {
		setHeaders = append(setHeaders, extraHeaders...)
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

// recordFallbackReplayOutcome attaches an audit trail outcome to the replay record.
func (r *OpenAIRouter) recordFallbackReplayOutcome(
	ctx *RequestContext,
	initialModel, fallbackModel string,
) {
	if ctx == nil || ctx.RouterReplayRecorder == nil || ctx.RouterReplayID == "" {
		return
	}
	attempts := 0
	if ctx.FallbackRecord != nil {
		attempts = len(ctx.FallbackRecord.Attempts)
	}
	metadata := map[string]string{
		"initial_model":  initialModel,
		"fallback_model": fallbackModel,
		"attempts":       strconv.Itoa(attempts),
		"reason":         "upstream_error_fallback",
	}
	outcome := routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    "fallback",
		Target:    "model",
		TargetRef: fallbackModel,
		Verdict:   "completed",
		Reason:    "fallback_succeeded",
		Metadata:  metadata,
	}
	if err := ctx.RouterReplayRecorder.AppendOutcome(ctx.RouterReplayID, outcome); err != nil {
		logging.ComponentErrorEvent("extproc", "fallback_outcome_persist_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}
