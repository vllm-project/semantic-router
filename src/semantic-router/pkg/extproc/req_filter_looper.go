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

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func isLooperAlgorithmType(algorithmType string) bool {
	switch algorithmType {
	case "confidence", "ratings", "remom":
		return true
	default:
		return false
	}
}

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
	if !isLooperAlgorithmType(decision.Algorithm.Type) {
		return false
	}

	// ReMoM algorithm can work with single model (first_only strategy)
	// Other algorithms (confidence, ratings) require multiple models
	if decision.Algorithm.Type == "remom" {
		if len(decision.ModelRefs) < 1 {
			return false
		}
	} else {
		if len(decision.ModelRefs) <= 1 {
			return false
		}
	}

	if !r.Config.Looper.IsEnabled() {
		logging.Warnf("Decision %s has algorithm configured but looper endpoint is not set", decision.Name)
		return false
	}
	return true
}

// handleLooperExecution executes the looper for multi-model decisions
// Returns an ImmediateResponse with the aggregated result
func (r *OpenAIRouter) handleLooperExecution(
	ctx context.Context,
	openAIRequest *openai.ChatCompletionNewParams,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("[Looper] Starting looper execution for decision: %s, algorithm: %s",
		decision.Name, decision.Algorithm.Type)

	// Create looper based on algorithm type
	l := looper.Factory(&r.Config.Looper, decision.Algorithm.Type)

	// Build looper request.
	// Response API requests always return JSON, so force non-streaming in the
	// looper to get a JSON body that TranslateResponse can parse. The
	// Response API layer handles its own streaming format separately.
	streaming := reqCtx.ExpectStreamingResponse
	if isResponseAPIRequest(reqCtx) {
		streaming = false
	}
	looperReq := &looper.Request{
		OriginalRequest: openAIRequest,
		ModelRefs:       decision.ModelRefs,
		ModelParams:     r.getModelParams(),
		Algorithm:       decision.Algorithm,
		IsStreaming:     streaming,
		DecisionName:    decision.Name,
	}

	// Execute looper
	resp, err := l.Execute(ctx, looperReq)
	if err != nil {
		logging.Errorf("[Looper] Execution failed: %v", err)
		return r.createErrorResponse(500, "Looper execution failed: "+err.Error()), nil
	}

	logging.Infof("[Looper] Execution completed, models_used=%v, iterations=%d, algorithm=%s",
		resp.ModelsUsed, resp.Iterations, resp.AlgorithmType)

	// Update context with looper results
	reqCtx.RequestModel = resp.Model
	reqCtx.VSRSelectedModel = resp.Model
	reqCtx.VSRSelectionMethod = resp.AlgorithmType

	// Capture router replay information if enabled
	// Use first model from ModelsUsed as the "selected" model for replay
	selectedModel := resp.Model
	if len(resp.ModelsUsed) > 0 {
		selectedModel = resp.ModelsUsed[0]
	}
	r.startRouterReplay(reqCtx, openAIRequest.Model, selectedModel, decision.Name)

	// Update router replay with success status (looper returns immediate response with 200)
	r.updateRouterReplayStatus(reqCtx, 200, false)

	// Attach response body to router replay record
	r.attachRouterReplayResponse(reqCtx, resp.Body, true)

	// Memory auto_store: the normal response body pipeline (where
	// scheduleResponseMemoryStore runs) is bypassed by ImmediateResponse.
	// Trigger it here while resp.Body is still in Chat Completions format
	// (extractAssistantResponseText parses Chat Completions). The
	// ResponseAPICtx on reqCtx provides the ConversationID and user message
	// needed by extractMemoryInfo / extractCurrentUserMessage.
	r.scheduleResponseMemoryStore(reqCtx, resp.Body)

	// Response API back-translation: the looper executes against Chat
	// Completions endpoints directly, so resp.Body is in Chat Completions
	// format. If the original client request was a Response API request,
	// translate the response back before returning to the client.
	if isResponseAPIRequest(reqCtx) && r.ResponseAPIFilter != nil {
		translated, err := r.ResponseAPIFilter.TranslateResponse(ctx, reqCtx.ResponseAPICtx, resp.Body)
		if err != nil {
			logging.Errorf("[Looper] Response API translation failed: %v", err)
		} else {
			resp.Body = translated
			logging.Infof("[Looper] Translated looper response to Response API format")
		}
	}

	// Create immediate response with detailed headers
	return r.createLooperResponse(resp, reqCtx), nil
}
