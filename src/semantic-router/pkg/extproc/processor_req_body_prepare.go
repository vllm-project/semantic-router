package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (r *OpenAIRouter) translateResponseAPIRequest(
	requestBody []byte,
	ctx *RequestContext,
) ([]byte, *ext_proc.ProcessingResponse) {
	if ctx.ResponseAPICtx == nil || !ctx.ResponseAPICtx.IsResponseAPIRequest || r.ResponseAPIFilter == nil {
		return requestBody, nil
	}

	respCtx, translatedBody, err := r.ResponseAPIFilter.TranslateRequest(ctx.TraceContext, requestBody)
	if err != nil {
		logging.Errorf("Response API translation error: %v", err)
		return nil, r.createErrorResponse(400, "Invalid Response API request: "+err.Error())
	}
	if respCtx == nil || translatedBody == nil {
		logging.Errorf("Response API: Request to /v1/responses missing required 'input' field")
		return nil, r.createErrorResponse(
			400,
			"Invalid Response API request: 'input' field is required. Use 'input' instead of 'messages' for Response API.",
		)
	}

	ctx.ResponseAPICtx = respCtx
	logging.Infof("Response API: Translated to Chat Completions format")
	return translatedBody, nil
}

func (r *OpenAIRouter) extractFastRequestState(
	requestBody []byte,
	ctx *RequestContext,
) (*FastExtractResult, error) {
	fast, err := extractContentFast(requestBody)
	if err != nil {
		logging.Errorf("Error extracting request fields: %v", err)
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
		metrics.RecordModelRequest(ctx.RequestModel)
		return nil, status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
	}
	if fast.Stream {
		logging.Infof("Original request contains stream parameter: true")
		ctx.ExpectStreamingResponse = true
	}
	return fast, nil
}

func (r *OpenAIRouter) runRequestPreRoutingStages(
	originalModel string,
	fast *FastExtractResult,
	ctx *RequestContext,
) (requestDecisionState, *ext_proc.ProcessingResponse) {
	decisionName, _, reasoningDecision, selectedModel, authzErr := r.performDecisionEvaluation(
		originalModel,
		fast.UserContent,
		fast.NonUserMessages,
		ctx,
	)
	if authzErr != nil {
		logging.Errorf("[Request Body] Authz evaluation failed: %v", authzErr)
		return requestDecisionState{}, r.createErrorResponse(403, authzErr.Error())
	}

	metrics.RecordModelRequest(selectedModel)
	if resp := r.handleFastResponse(ctx, decisionName); resp != nil {
		r.startRouterReplay(ctx, originalModel, selectedModel, decisionName)
		r.updateRouterReplayStatus(ctx, 200, false)
		return requestDecisionState{}, resp
	}
	if resp := r.applyRateLimitAndCacheChecks(ctx, selectedModel, decisionName); resp != nil {
		return requestDecisionState{}, resp
	}
	if ragErr := r.executeRAGPlugin(ctx, decisionName); ragErr != nil {
		return requestDecisionState{}, r.createErrorResponse(503, fmt.Sprintf("RAG retrieval failed: %v", ragErr))
	}

	return requestDecisionState{
		decisionName:      decisionName,
		reasoningDecision: reasoningDecision,
		selectedModel:     selectedModel,
	}, nil
}

func (r *OpenAIRouter) applyRateLimitAndCacheChecks(
	ctx *RequestContext,
	selectedModel string,
	decisionName string,
) *ext_proc.ProcessingResponse {
	if r.RateLimiter != nil {
		rlCtx := r.buildRateLimitContext(ctx, selectedModel)
		decision, err := r.RateLimiter.Check(rlCtx)
		if err != nil {
			logging.Errorf("[Request Body] Rate limit check error: %v", err)
			return r.createRateLimitResponse(decision)
		}
		if decision != nil && !decision.Allowed {
			logging.Infof("[Request Body] Rate limited: user=%s model=%s provider=%s remaining=%d",
				rlCtx.UserID, rlCtx.Model, decision.Provider, decision.Remaining)
			return r.createRateLimitResponse(decision)
		}
		ctx.RateLimitCtx = &rlCtx
	}

	if response, shouldReturn := r.handleCaching(ctx, decisionName); shouldReturn {
		logging.Infof("handleCaching returned a response, returning immediately")
		return response
	}
	logging.Infof("handleCaching returned no cached response, continuing to model routing")
	return nil
}

func (r *OpenAIRouter) prepareRequestForModelRouting(
	requestBody []byte,
	userContent string,
	ctx *RequestContext,
) (*openai.ChatCompletionNewParams, *ext_proc.ProcessingResponse, error) {
	r.maybeForceImageGenerationModality(userContent, ctx)

	openAIRequest, err := parseOpenAIRequest(requestBody)
	if err != nil {
		logging.Errorf("Error parsing OpenAI request for routing: %v", err)
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
		return nil, nil, status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
	}
	if resp, err := r.handleModalityFromDecision(ctx, openAIRequest); err != nil {
		logging.Errorf("[ModalityRouter] Error: %v", err)
		return nil, r.createErrorResponse(503, fmt.Sprintf("Modality routing failed: %v", err)), nil
	} else if resp != nil {
		return nil, resp, nil
	}

	requestBody, memErr := r.handleMemoryRetrieval(ctx, userContent, requestBody, openAIRequest)
	if memErr != nil {
		logging.Warnf("Memory retrieval failed: %v, continuing without memory", memErr)
	}
	if ctx.MemoryContext != "" {
		ctx.OriginalRequestBody = requestBody
	}
	r.refreshResponseAPITranslatedBody(ctx, requestBody)
	openAIRequest = r.reparseRequestWithMemory(requestBody, openAIRequest, ctx)

	return openAIRequest, nil, nil
}

func (r *OpenAIRouter) maybeForceImageGenerationModality(userContent string, ctx *RequestContext) {
	if ctx.ResponseAPICtx == nil || !ctx.ResponseAPICtx.HasImageGenerationTool {
		return
	}
	if ctx.ModalityClassification != nil && ctx.ModalityClassification.Modality != "" && ctx.ModalityClassification.Modality != ModalityAR {
		return
	}

	modality := ModalityDiffusion
	if userContent != "" {
		modality = ModalityBoth
	}
	ctx.ModalityClassification = &ModalityClassificationResult{
		Modality:   modality,
		Confidence: 1.0,
		Method:     "image_generation_tool",
	}
	logging.Infof("[ModalityRouter] Explicit image_generation tool detected — forcing modality=%s", modality)
}

func (r *OpenAIRouter) refreshResponseAPITranslatedBody(ctx *RequestContext, requestBody []byte) {
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest && len(requestBody) > 0 {
		ctx.ResponseAPICtx.TranslatedBody = requestBody
	}
}

func (r *OpenAIRouter) reparseRequestWithMemory(
	requestBody []byte,
	openAIRequest *openai.ChatCompletionNewParams,
	ctx *RequestContext,
) *openai.ChatCompletionNewParams {
	if ctx.MemoryContext == "" {
		return openAIRequest
	}

	updatedReq, err := parseOpenAIRequest(requestBody)
	if err != nil {
		logging.Errorf("[MemoryPatch] Failed to re-parse memory-augmented body: %v", err)
		return openAIRequest
	}

	logging.Infof("[MemoryPatch] Re-parsed request with memory, body_len=%d", len(requestBody))
	return updatedReq
}
