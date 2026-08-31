/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"context"
	"fmt"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// isLooperRequest checks if the incoming request is from looper (internal request)
// If so, extproc should skip plugin processing to avoid recursion
func (r *OpenAIRouter) isLooperRequest(ctx *RequestContext) bool {
	return ctx.LooperRequest
}

// shouldUseLooper checks if the decision requires looper execution
// Returns true if:
// - Decision has an Algorithm configured AND
// - Decision has at least one ModelRef (ReMoM supports single model) AND
// - Looper endpoint is configured in router config
func (r *OpenAIRouter) shouldUseLooper(decision *config.Decision) bool {
	if decision == nil || r.Config == nil {
		return false
	}
	if decision.Algorithm == nil {
		return false
	}
	if !config.IsLooperAlgorithmType(decision.Algorithm.Type) {
		return false
	}

	if !hasLooperModelInputs(decision) {
		return false
	}

	if !r.Config.Looper.IsEnabled() {
		logging.Warnf("Decision %s has algorithm configured but looper endpoint is not set", decision.Name)
		return false
	}
	return true
}

func hasLooperModelInputs(decision *config.Decision) bool {
	switch decision.Algorithm.Type {
	case config.DecisionAlgorithmReMoM:
		return len(decision.ModelRefs) >= 1
	case config.DecisionAlgorithmFusion:
		return len(decision.ModelRefs) >= 1 || hasFusionAnalysisModels(decision)
	case config.DecisionAlgorithmWorkflows:
		return len(decision.ModelRefs) >= 1
	default:
		return len(decision.ModelRefs) > 1
	}
}

func hasFusionAnalysisModels(decision *config.Decision) bool {
	return decision.Algorithm.Fusion != nil &&
		len(decision.Algorithm.Fusion.AnalysisModels) > 0
}

func (r *OpenAIRouter) createLooper(
	decision *config.Decision,
	reqCtx *RequestContext,
) (looper.Looper, error) {
	l, err := looper.Factory(&r.Config.Looper, decision.Algorithm.Type)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_construction_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"decision":   decision.Name,
			"algorithm":  decision.Algorithm.Type,
			"error":      err.Error(),
		})
	}
	return l, err
}

// handleLooperExecution executes the looper for multi-model decisions
// Returns an ImmediateResponse with the aggregated result
func (r *OpenAIRouter) handleLooperExecution(
	ctx context.Context,
	request *llmprotocol.Request,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	// Create looper based on algorithm type
	l, err := r.createLooper(decision, reqCtx)
	if err != nil {
		return r.createErrorResponse(500, "Looper construction failed: "+err.Error()), nil
	}
	looperReq, errorResponse := r.buildLooperRequest(request, decision, reqCtx)
	if errorResponse != nil {
		return errorResponse, nil
	}
	resp, errorResponse := r.executeLooperRequest(ctx, l, looperReq, request.Model, decision, reqCtx)
	if errorResponse != nil {
		return errorResponse, nil
	}

	response, semanticResponse, clientBody, err := r.prepareLooperResponse(resp, reqCtx)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_response_encode_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"format":     reqCtx.SourceFormat,
			"error":      err.Error(),
		})
		return r.createErrorResponse(502, "Looper returned an invalid response"), nil
	}
	r.recordSuccessfulLooperExecution(
		resp, request.Model, decision, reqCtx, semanticResponse, clientBody,
	)
	return response, nil
}

func (r *OpenAIRouter) buildLooperRequest(
	request *llmprotocol.Request,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*looper.Request, *ext_proc.ProcessingResponse) {
	// Build looper request.
	// Looper currently aggregates a buffered semantic result for non-Chat
	// clients. The common immediate-response codec encodes that result into the
	// inbound wire format after execution.
	streaming := reqCtx.ExpectStreamingResponse
	if isResponseAPIRequest(reqCtx) {
		streaming = false
	}
	logging.ComponentEvent("extproc", "looper_execution_started", map[string]interface{}{
		"request_id":       reqCtx.RequestID,
		"decision":         decision.Name,
		"algorithm":        decision.Algorithm.Type,
		"candidate_models": len(decision.ModelRefs),
		"streaming":        streaming,
		"response_api":     isResponseAPIRequest(reqCtx),
	})
	engine, err := r.protocolEngine()
	if err == nil {
		var encoded protocolcodec.RequestResult
		encoded, err = engine.EncodeRequest(llmprotocol.OpenAIChatV1, *request, llmprotocol.Envelope{})
		if err == nil {
			var openAIRequest *openai.ChatCompletionNewParams
			openAIRequest, err = parseOpenAIRequest(encoded.Body)
			if err == nil {
				looperReq := &looper.Request{
					OriginalRequest:    openAIRequest,
					ModelRefs:          decision.ModelRefs,
					ModelParams:        r.getModelParams(),
					Algorithm:          decision.Algorithm,
					IsStreaming:        streaming,
					DecisionName:       decision.Name,
					RecipeName:         reqCtx.Routing.RecipeName(),
					OutputContract:     decision.OutputContract,
					OutputContractSpec: decision.OutputContractSpec,
				}
				return looperReq, nil
			}
		}
	}
	if err != nil {
		return nil, r.recordLooperFailure(
			reqCtx, request.Model, decision, 400,
			"Looper cannot represent the request semantics", "looper_request_unsupported",
		)
	}
	return nil, r.recordLooperFailure(
		reqCtx, request.Model, decision, 400,
		"Looper cannot represent the request semantics", "looper_request_unsupported",
	)
}

func (r *OpenAIRouter) executeLooperRequest(
	ctx context.Context,
	l looper.Looper,
	req *looper.Request,
	originalModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*looper.Response, *ext_proc.ProcessingResponse) {
	// Execute looper, recording the wall-clock latency of the full execution
	// (all model calls plus algorithm overhead) on the response for the
	// x-vsr-looper-latency-ms debug header (#2694).
	resp, err := looper.ExecuteWithLatency(ctx, l, req)
	if err != nil {
		return nil, r.looperExecutionErrorResponse(err, originalModel, decision, reqCtx)
	}
	logging.ComponentEvent("extproc", "looper_execution_completed", map[string]interface{}{
		"request_id":     reqCtx.RequestID,
		"decision":       decision.Name,
		"algorithm":      resp.AlgorithmType,
		"models_used":    resp.ModelsUsed,
		"iterations":     resp.Iterations,
		"selected_model": resp.Model,
	})
	return resp, nil
}

func (r *OpenAIRouter) looperExecutionErrorResponse(
	err error,
	originalModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
) *ext_proc.ProcessingResponse {
	failureFields := map[string]interface{}{
		"request_id": reqCtx.RequestID,
		"decision":   decision.Name,
		"algorithm":  decision.Algorithm.Type,
		"error":      err.Error(),
	}
	executionEvidence, hasExecutionEvidence := looper.ConfidenceEvidenceFromError(err)
	var evidenceArgs []looper.ConfidenceExecutionEvidence
	if hasExecutionEvidence {
		failureFields["models_used"] = executionEvidence.ModelsUsed
		failureFields["iterations"] = executionEvidence.Iterations
		failureFields["prompt_tokens"] = executionEvidence.Usage.PromptTokens
		failureFields["completion_tokens"] = executionEvidence.Usage.CompletionTokens
		failureFields["total_tokens"] = executionEvidence.Usage.TotalTokens
		evidenceArgs = append(evidenceArgs, executionEvidence)
	}
	logging.ComponentErrorEvent("extproc", "looper_execution_failed", failureFields)
	return r.recordLooperFailure(
		reqCtx,
		originalModel,
		decision,
		500,
		"Looper execution failed: "+err.Error(),
		"looper_execution_failed",
		evidenceArgs...,
	)
}

func (r *OpenAIRouter) recordSuccessfulLooperExecution(
	resp *looper.Response,
	originalModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
	semanticResponse *llmprotocol.Response,
	clientBody []byte,
) {
	// Update context with looper results
	reqCtx.RequestModel = resp.Model
	reqCtx.VSRSelectedModel = resp.Model
	reqCtx.VSRSelectionMethod = resp.AlgorithmType

	// Capture router replay information if enabled
	// ModelsUsed is the execution trace; resp.Model is the final response model.
	r.startRouterReplay(reqCtx, originalModel, resp.Model, decision.Name)
	r.updateLooperReplayUsage(reqCtx, resp.Usage)

	// Update router replay with success status (looper returns immediate response with 200)
	r.updateRouterReplayStatus(reqCtx, 200, false)

	// Attach response body to router replay record
	r.attachRouterReplayResponse(reqCtx, clientBody, true)

	// Memory auto_store: the normal response body pipeline (where
	// scheduleResponseMemoryStore runs) is bypassed by ImmediateResponse.
	// A buffered looper result is decoded once into the same neutral response
	// contract used by direct inference. Streaming memory storage is deferred
	// until a terminal semantic accumulator is available.
	if semanticResponse != nil {
		r.scheduleSemanticResponseMemoryStore(reqCtx, semanticResponse)
	}
}

func (r *OpenAIRouter) recordLooperFailure(
	ctx *RequestContext,
	originalModel string,
	decision *config.Decision,
	statusCode int,
	message string,
	reason string,
	executionEvidence ...looper.ConfidenceExecutionEvidence,
) *ext_proc.ProcessingResponse {
	response := r.createErrorResponse(statusCode, message)
	decisionName := ""
	if decision != nil {
		decisionName = decision.Name
	}
	if len(executionEvidence) > 0 && executionEvidence[0].Iterations > 0 {
		evidence := executionEvidence[0]
		algorithm := "confidence"
		ctx.VSRSelectionMethod = algorithm
		ctx.VSRSelectionReasoning = boundedSelectionReasoning(fmt.Sprintf(
			"%s execution failed after %d call attempts; completed models: %s",
			algorithm,
			evidence.Iterations,
			strings.Join(evidence.ModelsUsed, ","),
		))
	}
	r.startRouterReplay(ctx, originalModel, "", decisionName)
	if len(executionEvidence) > 0 {
		r.updateLooperReplayUsage(ctx, executionEvidence[0].Usage)
	}
	r.updateRouterReplayStatus(ctx, statusCode, false)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		r.attachRouterReplayResponse(ctx, immediate.Body, false)
	}
	r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, reason)
	addRouterReplayHeaderToImmediateResponse(response, ctx.RouterReplayID)
	return response
}

func (r *OpenAIRouter) updateLooperReplayUsage(ctx *RequestContext, usage looper.TokenUsage) {
	promptTokens := int(usage.PromptTokens)
	completionTokens := int(usage.CompletionTokens)
	totalTokens := int(usage.TotalTokens)
	if promptTokens == 0 && completionTokens == 0 && totalTokens == 0 {
		return
	}

	// A looper result can aggregate differently priced model calls. Persist its
	// exact reported token counts, but leave pricing fields unset rather than
	// attributing the whole cascade to the selected model's price.
	r.updateRouterReplayUsageCost(ctx, routerreplay.UsageCost{
		PromptTokens:     replayIntPtr(promptTokens),
		CompletionTokens: replayIntPtr(completionTokens),
		TotalTokens:      replayIntPtr(totalTokens),
	})
}
