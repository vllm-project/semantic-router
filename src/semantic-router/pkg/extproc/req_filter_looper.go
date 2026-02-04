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
// - Decision has an Algorithm configured AND
// - Decision has at least one ModelRef (ReMoM supports single model) AND
// - Looper endpoint is configured in router config
func (r *OpenAIRouter) shouldUseLooper(decision *config.Decision) bool {
	if decision == nil {
		return false
	}
	if decision.Algorithm == nil {
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

	// Build looper request
	looperReq := &looper.Request{
		OriginalRequest: openAIRequest,
		ModelRefs:       decision.ModelRefs,
		ModelParams:     r.getModelParams(),
		Algorithm:       decision.Algorithm,
		IsStreaming:     reqCtx.ExpectStreamingResponse,
		DecisionName:    decision.Name, // Pass decision name for extproc lookup
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
	return r.createLooperResponse(resp, reqCtx), nil
}

// createLooperResponse creates an ImmediateResponse from looper output
// Includes headers for: model used, all models called, iteration count, algorithm type, and signal headers
func (r *OpenAIRouter) createLooperResponse(resp *looper.Response, reqCtx *RequestContext) *ext_proc.ProcessingResponse {
	// Build header list starting with looper-specific headers
	setHeaders := []*core.HeaderValueOption{
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
	}

	// Add signal tracking headers from RequestContext
	if len(reqCtx.VSRMatchedKeywords) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedKeywords,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedKeywords, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedEmbeddings) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedEmbeddings,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedEmbeddings, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedDomains) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedDomains,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedDomains, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedFactCheck) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedFactCheck,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedFactCheck, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedUserFeedback) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedUserFeedback,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedUserFeedback, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedPreference) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedPreference,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedPreference, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedLanguage) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedLanguage,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedLanguage, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedLatency) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedLatency,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedLatency, ",")),
			},
		})
	}

	if len(reqCtx.VSRMatchedContext) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedContext,
				RawValue: []byte(strings.Join(reqCtx.VSRMatchedContext, ",")),
			},
		})
	}

	if reqCtx.VSRContextTokenCount > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRContextTokenCount,
				RawValue: []byte(fmt.Sprintf("%d", reqCtx.VSRContextTokenCount)),
			},
		})
	}

	// Add decision-related headers
	if reqCtx.VSRSelectedDecisionName != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRSelectedDecision,
				RawValue: []byte(reqCtx.VSRSelectedDecisionName),
			},
		})
	}

	if reqCtx.VSRSelectedCategory != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRSelectedCategory,
				RawValue: []byte(reqCtx.VSRSelectedCategory),
			},
		})
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
				Body: resp.Body,
			},
		},
	}
}

// findDecisionByName finds a decision by name in the router configuration
func (r *OpenAIRouter) findDecisionByName(name string) *config.Decision {
	if r.Config == nil || r.Config.Decisions == nil {
		return nil
	}

	for i := range r.Config.Decisions {
		if r.Config.Decisions[i].Name == name {
			return &r.Config.Decisions[i]
		}
	}

	return nil
}

// getReasoningInfoFromDecision extracts reasoning configuration from a decision for a specific model
// Returns (useReasoning, reasoningEffort)
func (r *OpenAIRouter) getReasoningInfoFromDecision(
	decision *config.Decision,
	modelName string,
) (bool, string) {
	// 1. Check ModelRefs for model-specific reasoning configuration
	for _, ref := range decision.ModelRefs {
		// Match by model name or LoRA name
		if ref.Model == modelName || ref.LoRAName == modelName {
			// If ModelRef has UseReasoning configured, use it
			if ref.UseReasoning != nil {
				useReasoning := *ref.UseReasoning
				reasoningEffort := ref.ReasoningEffort
				if reasoningEffort == "" {
					reasoningEffort = "medium" // Default effort
				}
				logging.Infof("[Looper] Found reasoning config in ModelRef: use=%v, effort=%s", useReasoning, reasoningEffort)
				return useReasoning, reasoningEffort
			}
			break
		}
	}

	// 2. Check ModelParams for reasoning_family
	if r.Config != nil && r.Config.ModelConfig != nil {
		if params, ok := r.Config.ModelConfig[modelName]; ok {
			if params.ReasoningFamily != "" {
				// Model has reasoning_family, so it supports reasoning
				// Use default effort from global config or "medium"
				reasoningEffort := "medium"
				if r.Config.DefaultReasoningEffort != "" {
					reasoningEffort = r.Config.DefaultReasoningEffort
				}
				logging.Infof("[Looper] Found reasoning_family in ModelParams: %s, effort=%s", params.ReasoningFamily, reasoningEffort)
				return true, reasoningEffort
			}
		}
	}

	// 3. No reasoning configuration found
	return false, ""
}

// modifyRequestBodyForLooper modifies the request body for looper internal requests
// Similar to modifyRequestBodyForAutoRouting but for looper context
func (r *OpenAIRouter) modifyRequestBodyForLooper(
	openAIRequest *openai.ChatCompletionNewParams,
	modelName string,
	decisionName string,
	useReasoning bool,
	ctx *RequestContext,
) ([]byte, error) {
	// 1. Set model name in request
	openAIRequest.Model = modelName

	// 2. Serialize the modified request
	modifiedBody, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
	if err != nil {
		logging.Errorf("[Looper] Error serializing modified request: %v", err)
		return nil, fmt.Errorf("error serializing modified request: %w", err)
	}

	// 3. Apply reasoning mode if decision name is provided
	if decisionName != "" {
		modifiedBody, err = r.setReasoningModeToRequestBody(modifiedBody, useReasoning, decisionName)
		if err != nil {
			logging.Errorf("[Looper] Error setting reasoning mode %v to request: %v", useReasoning, err)
			return nil, fmt.Errorf("error setting reasoning mode: %w", err)
		}
	}

	// 4. Add decision-specific system prompt if configured
	if decisionName != "" {
		modifiedBody, err = r.addSystemPromptIfConfigured(modifiedBody, decisionName, modelName, ctx)
		if err != nil {
			logging.Errorf("[Looper] Error adding system prompt: %v", err)
			return nil, fmt.Errorf("error adding system prompt: %w", err)
		}
	}

	// Note: Memory context injection is skipped for looper requests
	// Memory plugin should skip looper requests internally

	return modifiedBody, nil
}

// buildHeaderMutationsForLooper builds header mutations for looper internal requests
func (r *OpenAIRouter) buildHeaderMutationsForLooper(
	decision *config.Decision,
	modelName string,
) ([]*core.HeaderValueOption, []string) {
	setHeaders := []*core.HeaderValueOption{}
	removeHeaders := []string{"content-length"} // Always remove old content-length when body is modified

	// 1. Add standard routing headers
	setHeaders = append(setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      headers.VSRSelectedModel,
			RawValue: []byte(modelName),
		},
	})

	// 2. Add Authorization header if model has access_key configured
	if accessKey := r.getModelAccessKey(modelName); accessKey != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "Authorization",
				RawValue: []byte(fmt.Sprintf("Bearer %s", accessKey)),
			},
		})
		logging.Infof("[Looper] Added Authorization header for model %s", modelName)
	}

	// 3. Apply header mutations from decision's header_mutation plugin
	if decision != nil {
		pluginSetHeaders, pluginRemoveHeaders := r.buildHeaderMutations(decision)
		if len(pluginSetHeaders) > 0 {
			setHeaders = append(setHeaders, pluginSetHeaders...)
			logging.Infof("[Looper] Applied %d header mutations from decision %s", len(pluginSetHeaders), decision.Name)
		}
		if len(pluginRemoveHeaders) > 0 {
			removeHeaders = append(removeHeaders, pluginRemoveHeaders...)
			logging.Infof("[Looper] Applied %d header deletions from decision %s", len(pluginRemoveHeaders), decision.Name)
		}
	}

	return setHeaders, removeHeaders
}

// handleLooperInternalRequest handles requests from looper to extproc
// This bypasses all plugin processing and routes directly to the specified model
// Deprecated: Use handleLooperInternalRequestWithPlugins instead
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

// handleLooperInternalRequestWithPlugins handles looper internal requests with plugin execution
// It looks up the decision by name and executes all configured plugins
func (r *OpenAIRouter) handleLooperInternalRequestWithPlugins(
	modelName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	// 1. Extract decision name from header
	decisionName := ctx.Headers[headers.VSRLooperDecision]
	if decisionName == "" {
		logging.Warnf("[Looper] No decision name in looper request, falling back to simple routing")
		return r.handleLooperInternalRequest(modelName, ctx)
	}

	logging.Infof("[Looper] Processing internal request for model: %s, decision: %s", modelName, decisionName)

	// 2. Find decision by name
	decision := r.findDecisionByName(decisionName)
	if decision == nil {
		logging.Warnf("[Looper] Decision %s not found, falling back to simple routing", decisionName)
		return r.handleLooperInternalRequest(modelName, ctx)
	}

	// 3. Set context fields
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decisionName
	ctx.VSRSelectedModel = modelName
	ctx.RequestModel = modelName

	// 3.1 Set router replay config from decision (required for startRouterReplay)
	if replayCfg := decision.GetRouterReplayConfig(); replayCfg != nil && replayCfg.Enabled {
		cfgCopy := *replayCfg
		ctx.RouterReplayPluginConfig = &cfgCopy
		logging.Debugf("[Looper] Router replay enabled for decision %s", decisionName)
	}

	// Note: Signals are NOT re-evaluated for looper internal requests to avoid latency
	// The signals from the original external request are already in the context
	// Router replay will use those signals for all looper internal calls

	// 4. Get reasoning info from decision
	useReasoning, reasoningEffort := r.getReasoningInfoFromDecision(decision, modelName)
	if useReasoning {
		ctx.VSRReasoningMode = "on"
		logging.Infof("[Looper] Reasoning enabled for model %s, effort: %s", modelName, reasoningEffort)
	} else {
		ctx.VSRReasoningMode = "off"
	}

	// 5. Parse OpenAI request
	openAIRequest, err := parseOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("[Looper] Failed to parse request body: %v", err)
		return r.createErrorResponse(400, "Invalid request body"), nil
	}

	// 6. Get user content for plugin processing
	userContent, nonUserMessages := extractUserAndNonUserContent(openAIRequest)
	ctx.UserContent = userContent

	// 7. Execute plugins according to decision configuration
	// Note: Plugins will check ctx.LooperRequest internally for special handling

	// 7.1 Jailbreak detection
	if response, shouldReturn := r.performJailbreaks(ctx, userContent, nonUserMessages, decisionName); shouldReturn {
		return response, nil
	}

	// 7.2 PII detection
	if piiResponse := r.performPIIDetection(ctx, userContent, nonUserMessages, decisionName); piiResponse != nil {
		return piiResponse, nil
	}

	// 7.3 Semantic Cache (plugin will skip read for looper requests)
	if response, shouldReturn := r.handleCaching(ctx, decisionName); shouldReturn {
		return response, nil
	}

	// 7.4 RAG (plugin will skip for looper requests)
	if err = r.executeRAGPlugin(ctx, decisionName); err != nil {
		return r.createErrorResponse(503, fmt.Sprintf("RAG failed: %v", err)), nil
	}

	// Note: Memory plugin is skipped for looper requests
	// Memory plugin should check ctx.LooperRequest internally and skip retrieval

	// 8. Modify request body with model, reasoning, and system prompt
	modifiedBody, err := r.modifyRequestBodyForLooper(openAIRequest, modelName, decisionName, useReasoning, ctx)
	if err != nil {
		logging.Errorf("[Looper] Failed to modify request body: %v", err)
		return r.createErrorResponse(500, "Failed to process looper request"), nil
	}

	// 9. Build header mutations
	setHeaders, removeHeaders := r.buildHeaderMutationsForLooper(decision, modelName)

	// 10. Start router replay if enabled (for looper internal requests)
	// Note: This captures each individual model call within the looper
	// Clear RouterReplayID to allow creating a new record for each looper internal call
	// (startRouterReplay skips if RouterReplayID is already set)
	ctx.RouterReplayID = "" // Clear to allow new record creation
	// For looper internal requests, both originalModel and selectedModel are the same (the actual model being called)
	r.startRouterReplay(ctx, "ReMoM", modelName, decisionName)

	// 11. Return response with mutations
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders:    setHeaders,
						RemoveHeaders: removeHeaders,
					},
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{Body: modifiedBody},
					},
					ClearRouteCache: true,
				},
			},
		},
	}, nil
}
