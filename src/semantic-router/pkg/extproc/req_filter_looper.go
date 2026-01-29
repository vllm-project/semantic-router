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

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// isLooperRequest checks if the incoming request is from looper (internal request)
// If so, extproc should skip plugin processing to avoid recursion
func (r *OpenAIRouter) isLooperRequest(ctx *RequestContext) bool {
	return ctx.LooperRequest
}

// shouldUseLooper checks if the decision requires looper execution
// Returns true if:
// - Decision has multiple ModelRefs AND
// - Decision has an Algorithm configured AND
// - Looper endpoint is configured in router config
func (r *OpenAIRouter) shouldUseLooper(decision *config.Decision) bool {
	if decision == nil {
		return false
	}
	if len(decision.ModelRefs) <= 1 {
		return false
	}
	if decision.Algorithm == nil {
		return false
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

	// Build looper request
	looperReq := &looper.Request{
		OriginalRequest: openAIRequest,
		ModelRefs:       decision.ModelRefs,
		ModelParams:     r.getModelParams(),
		Algorithm:       decision.Algorithm,
		IsStreaming:     reqCtx.ExpectStreamingResponse,
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

	// Create immediate response with detailed headers
	return r.createLooperResponse(resp), nil
}

// createLooperResponse creates an ImmediateResponse from looper output
// Includes headers for: model used, all models called, iteration count, algorithm type
func (r *OpenAIRouter) createLooperResponse(resp *looper.Response) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: typev3.StatusCode_OK,
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:      "content-type",
								RawValue: []byte(resp.ContentType),
							},
						},
						{
							Header: &core.HeaderValue{
								Key:      headers.VSRLooperModel,
								RawValue: []byte(resp.Model),
							},
						},
						{
							Header: &core.HeaderValue{
								Key:      headers.VSRLooperModelsUsed,
								RawValue: []byte(strings.Join(resp.ModelsUsed, ",")),
							},
						},
						{
							Header: &core.HeaderValue{
								Key:      headers.VSRLooperIterations,
								RawValue: []byte(fmt.Sprintf("%d", resp.Iterations)),
							},
						},
						{
							Header: &core.HeaderValue{
								Key:      headers.VSRLooperAlgorithm,
								RawValue: []byte(resp.AlgorithmType),
							},
						},
					},
				},
				Body: resp.Body,
			},
		},
	}
}

// handleLooperInternalRequest handles requests from looper to extproc
// This bypasses all plugin processing and routes directly to the specified model
func (r *OpenAIRouter) handleLooperInternalRequest(
	modelName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("[Looper] Handling internal request for model: %s", modelName)

	// Rewrite request body with the target model
	modifiedBody, err := rewriteRequestModel(ctx.OriginalRequestBody, modelName)
	if err != nil {
		logging.Errorf("[Looper] Failed to rewrite request body: %v", err)
		return r.createErrorResponse(500, "Failed to process looper request: "+err.Error()), nil
	}

	// Build header mutations - just set the model header
	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      headers.VSRSelectedModel,
				RawValue: []byte(modelName),
			},
		},
	}

	// Return response that continues to upstream with modified body
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:          ext_proc.CommonResponse_CONTINUE,
					HeaderMutation:  &ext_proc.HeaderMutation{SetHeaders: setHeaders},
					BodyMutation:    &ext_proc.BodyMutation{Mutation: &ext_proc.BodyMutation_Body{Body: modifiedBody}},
					ClearRouteCache: true,
				},
			},
		},
	}, nil
}
