package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	handled, err := r.applyToolScope(openAIRequest, response, ctx)
	if err != nil {
		return err
	}
	if handled {
		return nil
	}

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

	selectedTools, toolErr := r.findToolsForQuery(classificationText, ctx)
	if toolErr != nil {
		if r.Config.Tools.FallbackToEmpty {
			logging.Warnf("Tool selection failed, falling back to no tools: %v", toolErr)
			openAIRequest.Tools = nil
			return r.updateRequestWithTools(openAIRequest, response, ctx)
		}
		metrics.RecordRequestError(getModelFromCtx(ctx), "classification_failed")
		return toolErr
	}

	if len(selectedTools) == 0 {
		if r.Config.Tools.FallbackToEmpty {
			logging.Infof("No suitable tools found, falling back to no tools")
			openAIRequest.Tools = nil
		} else {
			logging.Infof("No suitable tools found above threshold")
			openAIRequest.Tools = []openai.ChatCompletionToolParam{}
		}
	} else {
		sdkTools, err := convertToSDKTools(selectedTools)
		if err != nil {
			metrics.RecordRequestError(getModelFromCtx(ctx), "serialization_error")
			return err
		}
		openAIRequest.Tools = sdkTools
		logging.Infof("Auto-selected %d tools for query: %s", len(sdkTools), classificationText)
	}

	return r.updateRequestWithTools(openAIRequest, response, ctx)
}

// findToolsForQuery discovers candidate tools via semantic similarity, applying
// advanced filtering when configured.
func (r *OpenAIRouter) findToolsForQuery(query string, ctx *RequestContext) ([]openai.ChatCompletionToolParam, error) {
	topK := r.Config.Tools.TopK
	if topK <= 0 {
		topK = 3
	}

	advanced := r.Config.Tools.AdvancedFiltering
	if advanced == nil || !advanced.Enabled {
		return r.ToolsDatabase.FindSimilarTools(query, topK)
	}

	candidates, err := r.ToolsDatabase.FindSimilarToolsWithScores(query, resolveCandidatePoolSize(advanced, topK))
	if err != nil {
		return nil, err
	}

	return tools.FilterAndRankTools(query, candidates, topK, advanced, resolveCategory(advanced, ctx)), nil
}

func resolveCandidatePoolSize(advanced *config.AdvancedToolFilteringConfig, topK int) int {
	var size int
	switch {
	case advanced.CandidatePoolSize != nil && *advanced.CandidatePoolSize > 0:
		size = *advanced.CandidatePoolSize
	case advanced.CandidatePoolSize == nil:
		size = max(topK*candidatePoolMultiplier, candidatePoolMinSize)
	default:
		size = topK
	}
	if size < topK {
		size = topK
	}
	return size
}

func resolveCategory(advanced *config.AdvancedToolFilteringConfig, ctx *RequestContext) string {
	cat := ctx.VSRSelectedCategory
	if advanced.UseCategoryFilter == nil || !*advanced.UseCategoryFilter || cat == "" {
		return cat
	}
	if advanced.CategoryConfidenceThreshold != nil &&
		ctx.VSRSelectedDecisionConfidence < float64(*advanced.CategoryConfidenceThreshold) {
		return ""
	}
	return cat
}

// convertToSDKTools re-serializes tool params through JSON to ensure they
// conform to the OpenAI SDK type.
func convertToSDKTools(selected []openai.ChatCompletionToolParam) ([]openai.ChatCompletionToolParam, error) {
	out := make([]openai.ChatCompletionToolParam, len(selected))
	for i, tool := range selected {
		b, err := json.Marshal(tool)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, &out[i]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// applyToolScope enforces per-decision tool scope restrictions before semantic
// tool selection runs. Returns (true, nil) when the scope has been fully
// handled and the caller should skip further tool selection. Returns
// (false, nil) when tool selection should proceed normally.
func (r *OpenAIRouter) applyToolScope(openAIRequest *openai.ChatCompletionNewParams, response **ext_proc.ProcessingResponse, ctx *RequestContext) (bool, error) {
	dec := ctx.VSRSelectedDecision
	if dec == nil || dec.ToolScope == "" {
		return false, nil
	}

	switch dec.ToolScope {
	case config.ToolScopeNone:
		logging.Infof("[ToolScope] Decision %q has tool_scope=none, stripping all tools", dec.Name)
		openAIRequest.Tools = nil
		if err := r.updateRequestWithTools(openAIRequest, response, ctx); err != nil {
			return false, err
		}
		return true, nil

	case config.ToolScopeLocalOnly, config.ToolScopeStandard:
		if len(dec.AllowTools) > 0 || len(dec.BlockTools) > 0 {
			openAIRequest.Tools = filterToolsByDecisionPolicy(openAIRequest.Tools, dec.AllowTools, dec.BlockTools)
			logging.Infof("[ToolScope] Decision %q tool_scope=%s, filtered to %d tools via allow/block lists",
				dec.Name, dec.ToolScope, len(openAIRequest.Tools))
			if err := r.updateRequestWithTools(openAIRequest, response, ctx); err != nil {
				return false, err
			}
			return true, nil
		}
		return false, nil

	case config.ToolScopeFull:
		return false, nil

	default:
		logging.Warnf("[ToolScope] Decision %q has unrecognized tool_scope=%q, stripping all tools as a safety measure",
			dec.Name, dec.ToolScope)
		openAIRequest.Tools = nil
		if err := r.updateRequestWithTools(openAIRequest, response, ctx); err != nil {
			return false, err
		}
		return true, nil
	}
}

// filterToolsByDecisionPolicy applies per-decision allow/block lists to the
// request tool set. If allowTools is non-empty, only tools whose function name
// appears in the allow list are kept. Any tool whose name appears in blockTools
// is removed regardless.
func filterToolsByDecisionPolicy(tools []openai.ChatCompletionToolParam, allowTools, blockTools []string) []openai.ChatCompletionToolParam {
	allowSet := make(map[string]bool, len(allowTools))
	for _, t := range allowTools {
		allowSet[t] = true
	}
	blockSet := make(map[string]bool, len(blockTools))
	for _, t := range blockTools {
		blockSet[t] = true
	}

	filtered := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, tool := range tools {
		name := tool.Function.Name
		if blockSet[name] {
			continue
		}
		if len(allowSet) > 0 && !allowSet[name] {
			continue
		}
		filtered = append(filtered, tool)
	}
	return filtered
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
