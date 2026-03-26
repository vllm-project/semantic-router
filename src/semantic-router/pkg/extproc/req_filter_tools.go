package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

	if clearToolChoiceWhenNoTools(openAIRequest) {
		if err := r.updateRequestWithTools(openAIRequest, &response, ctx); err != nil {
			logging.Errorf("Error clearing invalid tool_choice without tools: %v", err)
		}
	}
}

// handleToolSelection handles automatic tool selection based on semantic similarity
func (r *OpenAIRouter) handleToolSelection(openAIRequest *openai.ChatCompletionNewParams, userContent string, nonUserMessages []string, response **ext_proc.ProcessingResponse, ctx *RequestContext) error {
	toolsCfg := resolveDecisionToolsConfig(ctx)
	if toolsCfg == nil || !toolsCfg.Enabled {
		return nil
	}

	switch toolsCfg.EffectiveMode() {
	case config.ToolsPluginModeNone:
		logging.Infof("[ToolsPlugin] Decision %q has mode=none, stripping all tools", ctx.VSRSelectedDecision.Name)
		openAIRequest.Tools = nil
		return r.updateRequestWithTools(openAIRequest, response, ctx)
	case config.ToolsPluginModeFiltered:
		openAIRequest.Tools = filterToolsByDecisionPolicy(openAIRequest.Tools, toolsCfg.AllowTools, toolsCfg.BlockTools)
		if openAIRequest.ToolChoice.OfAuto.Value != "auto" {
			logging.Infof("[ToolsPlugin] Decision %q filtered explicit tools to %d entries", ctx.VSRSelectedDecision.Name, len(openAIRequest.Tools))
			return r.updateRequestWithTools(openAIRequest, response, ctx)
		}
	case config.ToolsPluginModePassthrough:
		// No explicit-tool filtering; semantic selection may still run below.
	default:
		return fmt.Errorf("tools plugin: unsupported mode %q", toolsCfg.Mode)
	}

	if openAIRequest.ToolChoice.OfAuto.Value != "auto" {
		return nil
	}
	if !toolsCfg.SelectionEnabled() {
		return r.updateRequestWithTools(openAIRequest, response, ctx)
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

	selectedTools, toolErr := r.findToolsForQuery(classificationText, ctx, toolsCfg)
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
func (r *OpenAIRouter) findToolsForQuery(query string, ctx *RequestContext, toolsCfg *config.ToolsPluginConfig) ([]openai.ChatCompletionToolParam, error) {
	topK := r.Config.Tools.TopK
	if topK <= 0 {
		topK = 3
	}

	advanced := mergeAdvancedToolFiltering(r.Config.Tools.AdvancedFiltering, toolsCfg)
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
	if advanced == nil {
		return max(topK*candidatePoolMultiplier, candidatePoolMinSize)
	}
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

func resolveDecisionToolsConfig(ctx *RequestContext) *config.ToolsPluginConfig {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	return ctx.VSRSelectedDecision.GetToolsConfig()
}

func mergeAdvancedToolFiltering(base *config.AdvancedToolFilteringConfig, toolsCfg *config.ToolsPluginConfig) *config.AdvancedToolFilteringConfig {
	if toolsCfg == nil || toolsCfg.EffectiveMode() != config.ToolsPluginModeFiltered {
		return base
	}

	allowTools, blockTools := mergeToolFilters(base, toolsCfg)
	if base == nil {
		return &config.AdvancedToolFilteringConfig{
			Enabled:    true,
			AllowTools: allowTools,
			BlockTools: blockTools,
		}
	}

	merged := *base
	merged.AllowTools = allowTools
	merged.BlockTools = blockTools
	return &merged
}

func mergeToolFilters(base *config.AdvancedToolFilteringConfig, toolsCfg *config.ToolsPluginConfig) ([]string, []string) {
	globalAllow := []string(nil)
	globalBlock := []string(nil)
	if base != nil {
		globalAllow = append(globalAllow, base.AllowTools...)
		globalBlock = append(globalBlock, base.BlockTools...)
	}
	pluginAllow := append([]string(nil), toolsCfg.AllowTools...)
	pluginBlock := append([]string(nil), toolsCfg.BlockTools...)

	return intersectToolAllowLists(globalAllow, pluginAllow), unionToolBlockLists(globalBlock, pluginBlock)
}

func intersectToolAllowLists(left, right []string) []string {
	switch {
	case len(left) == 0:
		return append([]string(nil), right...)
	case len(right) == 0:
		return append([]string(nil), left...)
	}

	rightSet := make(map[string]struct{}, len(right))
	for _, value := range right {
		rightSet[strings.ToLower(strings.TrimSpace(value))] = struct{}{}
	}

	result := make([]string, 0, min(len(left), len(right)))
	seen := make(map[string]struct{})
	for _, value := range left {
		key := strings.ToLower(strings.TrimSpace(value))
		if key == "" {
			continue
		}
		if _, ok := rightSet[key]; !ok {
			continue
		}
		if _, duplicated := seen[key]; duplicated {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, value)
	}
	return result
}

func unionToolBlockLists(left, right []string) []string {
	result := make([]string, 0, len(left)+len(right))
	seen := make(map[string]struct{}, len(left)+len(right))
	for _, value := range append(append([]string(nil), left...), right...) {
		key := strings.ToLower(strings.TrimSpace(value))
		if key == "" {
			continue
		}
		if _, duplicated := seen[key]; duplicated {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, value)
	}
	return result
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

	commonResponse := ensureRequestBodyCommonResponse(response)
	commonResponse.Status = ext_proc.CommonResponse_CONTINUE
	commonResponse.BodyMutation = bodyMutation
	if commonResponse.HeaderMutation == nil {
		commonResponse.HeaderMutation = &ext_proc.HeaderMutation{}
	}

	ensureHeaderRemoved(commonResponse.HeaderMutation, "content-length")
	setHeaderValue(commonResponse.HeaderMutation, "content-length", fmt.Sprintf("%d", len(modifiedBody)))

	// Check if route cache should be cleared
	if r.shouldClearRouteCache() {
		commonResponse.ClearRouteCache = true
		logging.Debugf("Setting ClearRouteCache=true (feature enabled) in updateRequestWithTools")
	}

	return nil
}

func ensureRequestBodyCommonResponse(response **ext_proc.ProcessingResponse) *ext_proc.CommonResponse {
	if *response == nil {
		*response = &ext_proc.ProcessingResponse{}
	}
	if (*response).GetRequestBody() == nil {
		(*response).Response = &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{},
		}
	}
	if (*response).GetRequestBody().GetResponse() == nil {
		(*response).GetRequestBody().Response = &ext_proc.CommonResponse{}
	}
	return (*response).GetRequestBody().GetResponse()
}

func ensureHeaderRemoved(mutation *ext_proc.HeaderMutation, key string) {
	for _, existing := range mutation.RemoveHeaders {
		if existing == key {
			return
		}
	}
	mutation.RemoveHeaders = append(mutation.RemoveHeaders, key)
}

func clearToolChoiceWhenNoTools(openAIRequest *openai.ChatCompletionNewParams) bool {
	if openAIRequest == nil || len(openAIRequest.Tools) > 0 || !hasToolChoice(openAIRequest) {
		return false
	}

	logging.Infof("[ToolsPlugin] Clearing tool_choice because no tools are present in the upstream request")
	openAIRequest.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{}
	return true
}

func hasToolChoice(openAIRequest *openai.ChatCompletionNewParams) bool {
	if openAIRequest == nil {
		return false
	}

	return !param.IsOmitted(openAIRequest.ToolChoice.OfAuto) ||
		openAIRequest.ToolChoice.OfChatCompletionNamedToolChoice != nil
}

func setHeaderValue(mutation *ext_proc.HeaderMutation, key, value string) {
	for _, option := range mutation.SetHeaders {
		if option.Header != nil && option.Header.Key == key {
			option.Header.RawValue = []byte(value)
			option.Header.Value = value
			return
		}
	}
	mutation.SetHeaders = append(mutation.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			Value:    value,
			RawValue: []byte(value),
		},
	})
}
