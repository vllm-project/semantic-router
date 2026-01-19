package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

const (
	candidatePoolMultiplier = 5
	candidatePoolMinSize    = 20
)

// handleToolSelectionForRequest handles tool selection for the request
func (r *OpenAIRouter) handleToolSelectionForRequest(openAIRequest *openai.ChatCompletionNewParams, response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	userContent, nonUserMessages := extractUserAndNonUserContent(openAIRequest)

	if err := r.handleToolSelection(openAIRequest, userContent, nonUserMessages, &response, ctx); err != nil {
		logging.Errorf("Error in tool selection: %v", err)
		// Continue without failing the request
	}
}

// handleToolSelection handles automatic tool selection based on semantic similarity
func (r *OpenAIRouter) handleToolSelection(openAIRequest *openai.ChatCompletionNewParams, userContent string, nonUserMessages []string, response **ext_proc.ProcessingResponse, ctx *RequestContext) error {
	// Check if tool_choice is set to "auto"
	if openAIRequest.ToolChoice.OfAuto.Value == "auto" {
		// Continue with tool selection logic
	} else {
		return nil // Not auto tool selection
	}

	// Get text for tools classification
	var classificationText string
	if len(userContent) > 0 {
		classificationText = userContent
	} else if len(nonUserMessages) > 0 {
		classificationText = strings.Join(nonUserMessages, " ")
	}

	if classificationText == "" {
		logging.Infof("No content available for tool classification")
		return nil
	}

	if !r.ToolsDatabase.IsEnabled() {
		logging.Infof("Tools database is disabled")
		return nil
	}

	// Get configuration for tool selection
	topK := r.Config.Tools.TopK
	if topK <= 0 {
		topK = 3 // Default to 3 tools
	}

	// Find similar tools based on the query
	var selectedTools []openai.ChatCompletionToolParam
	var err error

	advanced := r.Config.Tools.AdvancedFiltering
	if advanced != nil && advanced.Enabled {
		candidatePoolSize := topK
		if advanced.CandidatePoolSize != nil && *advanced.CandidatePoolSize > 0 {
			candidatePoolSize = *advanced.CandidatePoolSize
		} else if advanced.CandidatePoolSize == nil {
			candidatePoolSize = max(topK*candidatePoolMultiplier, candidatePoolMinSize)
		}
		if candidatePoolSize < topK {
			candidatePoolSize = topK
		}

		candidates, findErr := r.ToolsDatabase.FindSimilarToolsWithScores(classificationText, candidatePoolSize)
		if findErr != nil {
			err = findErr
		} else {
			selectedCategory := ctx.VSRSelectedCategory
			if advanced.UseCategoryFilter != nil && *advanced.UseCategoryFilter && selectedCategory != "" {
				if advanced.CategoryConfidenceThreshold != nil &&
					ctx.VSRSelectedDecisionConfidence < float64(*advanced.CategoryConfidenceThreshold) {
					selectedCategory = ""
				}
			}
			selectedTools = tools.FilterAndRankTools(classificationText, candidates, topK, advanced, selectedCategory)
		}
	} else {
		selectedTools, err = r.ToolsDatabase.FindSimilarTools(classificationText, topK)
	}

	if err != nil {
		if r.Config.Tools.FallbackToEmpty {
			logging.Warnf("Tool selection failed, falling back to no tools: %v", err)
			openAIRequest.Tools = nil
			return r.updateRequestWithTools(openAIRequest, response, ctx)
		}
		metrics.RecordRequestError(getModelFromCtx(ctx), "classification_failed")
		return err
	}

	if len(selectedTools) == 0 {
		if r.Config.Tools.FallbackToEmpty {
			logging.Infof("No suitable tools found, falling back to no tools")
			openAIRequest.Tools = nil
		} else {
			logging.Infof("No suitable tools found above threshold")
			openAIRequest.Tools = []openai.ChatCompletionToolParam{} // Empty array
		}
	} else {
		// Convert selected tools to OpenAI SDK tool format
		tools := make([]openai.ChatCompletionToolParam, len(selectedTools))
		for i, tool := range selectedTools {
			// Convert the tool to OpenAI SDK format
			toolBytes, err := json.Marshal(tool)
			if err != nil {
				metrics.RecordRequestError(getModelFromCtx(ctx), "serialization_error")
				return err
			}
			var sdkTool openai.ChatCompletionToolParam
			if err := json.Unmarshal(toolBytes, &sdkTool); err != nil {
				return err
			}
			tools[i] = sdkTool
		}

		openAIRequest.Tools = tools
		logging.Infof("Auto-selected %d tools for query: %s", len(selectedTools), classificationText)
	}

	return r.updateRequestWithTools(openAIRequest, response, ctx)
}

// updateRequestWithTools updates the request body with the selected tools
func (r *OpenAIRouter) updateRequestWithTools(openAIRequest *openai.ChatCompletionNewParams, response **ext_proc.ProcessingResponse, ctx *RequestContext) error {
	// Re-serialize the request with modified tools and preserved stream parameter
	modifiedBody, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
	if err != nil {
		return err
	}

	// Create body mutation with the modified body
	bodyMutation := &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: modifiedBody,
		},
	}

	// Create header mutation with content-length removal AND all necessary routing headers
	// (body phase HeaderMutation replaces header phase completely)

	// Get the headers that should have been set in the main routing
	var selectedEndpoint, actualModel string

	// These should be available from the existing response
	if (*response).GetRequestBody() != nil && (*response).GetRequestBody().GetResponse() != nil &&
		(*response).GetRequestBody().GetResponse().GetHeaderMutation() != nil &&
		(*response).GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() != nil {
		for _, header := range (*response).GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
			switch header.Header.Key {
			case headers.GatewayDestinationEndpoint:
				selectedEndpoint = header.Header.Value
			case headers.SelectedModel:
				actualModel = header.Header.Value
			}
		}
	}

	setHeaders := []*core.HeaderValueOption{}

	// Add new content-length for the modified body
	if len(modifiedBody) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "content-length",
				RawValue: []byte(fmt.Sprintf("%d", len(modifiedBody))),
			},
		})
	}

	if selectedEndpoint != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.GatewayDestinationEndpoint,
				RawValue: []byte(selectedEndpoint),
			},
		})
	}
	if actualModel != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.SelectedModel,
				RawValue: []byte(actualModel),
			},
		})
	}

	// Intentionally do not mutate Authorization header here

	headerMutation := &ext_proc.HeaderMutation{
		RemoveHeaders: []string{"content-length"},
		SetHeaders:    setHeaders,
	}

	// Create CommonResponse
	commonResponse := &ext_proc.CommonResponse{
		Status:         ext_proc.CommonResponse_CONTINUE,
		HeaderMutation: headerMutation,
		BodyMutation:   bodyMutation,
	}

	// Check if route cache should be cleared
	if r.shouldClearRouteCache() {
		commonResponse.ClearRouteCache = true
		logging.Debugf("Setting ClearRouteCache=true (feature enabled) in updateRequestWithTools")
	}

	// Update the response with body mutation and content-length removal
	*response = &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: commonResponse,
			},
		},
	}

	return nil
}
