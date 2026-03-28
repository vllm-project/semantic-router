package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// handleMemoryRetrieval retrieves relevant memories and injects them into the request.
// Per-decision plugin config takes precedence over global config.
func (r *OpenAIRouter) handleMemoryRetrieval(
	ctx *RequestContext,
	userContent string,
	requestBody []byte,
	openAIRequest *openai.ChatCompletionNewParams,
) ([]byte, error) {
	memoryPluginConfig, shouldRetrieve := r.resolveMemoryPluginConfig(ctx)
	if !shouldRetrieve {
		return requestBody, nil
	}
	store := r.getMemoryStore()
	if store == nil || !store.IsEnabled() {
		logging.Debugf("Memory: Store not available or disabled, skipping retrieval")
		return requestBody, nil
	}

	logging.Debugf("Memory: retrieval flow query=%q", truncateForLog(userContent, 80))
	searchQuery, userID, shouldSearch := r.prepareMemorySearchQuery(ctx, userContent, openAIRequest)
	if !shouldSearch {
		return requestBody, nil
	}
	retrieveOpts := r.buildMemoryRetrieveOptions(memoryPluginConfig, searchQuery, userID)
	memories, err := store.Retrieve(ctx.TraceContext, retrieveOpts)
	if err != nil {
		metrics.RecordPluginExecution("memory", ctx.VSRSelectedDecisionName, "retrieval_error", 0)
		logging.Errorf("Memory: retrieval failed for user=%s decision=%s query=%q: %v",
			userID, ctx.VSRSelectedDecisionName, truncateForLog(searchQuery, 60), err)
		return requestBody, fmt.Errorf("memory retrieval failed: %w", err)
	}
	memories = r.filterRetrievedMemories(memoryPluginConfig, memories, userID)
	if len(memories) == 0 {
		return requestBody, nil
	}
	return r.injectRetrievedMemories(ctx, requestBody, memories), nil
}

func (r *OpenAIRouter) resolveMemoryPluginConfig(
	ctx *RequestContext,
) (*config.MemoryPluginConfig, bool) {
	var memoryPluginConfig *config.MemoryPluginConfig
	if ctx.VSRSelectedDecision != nil {
		memoryPluginConfig = ctx.VSRSelectedDecision.GetMemoryConfig()
	}

	memoryEnabled := r.Config.Memory.Enabled
	if memoryPluginConfig != nil {
		memoryEnabled = memoryPluginConfig.Enabled
		if !memoryEnabled {
			logging.Debugf("Memory: Disabled by per-decision plugin config for decision '%s'", ctx.VSRSelectedDecisionName)
			return memoryPluginConfig, false
		}
	} else if !memoryEnabled {
		logging.Debugf("Memory: Disabled in global config, skipping retrieval")
		return nil, false
	}

	// Check opt-out header (x-disable-router-memory: true)
	if ctx.Headers["x-disable-router-memory"] == "true" {
		logging.Debugf("Memory: Disabled via x-disable-router-memory header (SDK-managed memory opt-out)")
		return memoryPluginConfig, false
	}

	// Check config-based per-route disable
	requestPath := ctx.Headers[":path"]
	if r.isMemoryDisabledForRoute(requestPath) {
		logging.Debugf("Memory: Disabled for route %s via config (SDK-managed memory opt-out)", requestPath)
		return memoryPluginConfig, false
	}

	// Check config-based per-model disable
	if ctx.RequestModel != "" && r.isMemoryDisabledForModel(ctx.RequestModel) {
		logging.Debugf("Memory: Disabled for model %s via config (SDK-managed memory opt-out)", ctx.RequestModel)
		return memoryPluginConfig, false
	}

	return memoryPluginConfig, true
}

func (r *OpenAIRouter) prepareMemorySearchQuery(
	ctx *RequestContext,
	userContent string,
	openAIRequest *openai.ChatCompletionNewParams,
) (string, string, bool) {
	if !ShouldSearchMemory(ctx, userContent) {
		logging.Debugf("Memory: skipping search (query type not suitable)")
		return "", "", false
	}

	history := r.extractConversationHistory(openAIRequest)
	searchQuery, err := BuildSearchQuery(ctx.TraceContext, history, userContent, r.Config)
	if err != nil {
		logging.Warnf("Memory: Query rewriting failed, using original query: %v", err)
		searchQuery = userContent
	}

	userID := r.getUserIDFromContext(ctx)
	if userID == "" {
		logging.Debugf("Memory: no user ID, skipping search")
		return "", "", false
	}

	return searchQuery, userID, true
}

func (r *OpenAIRouter) extractConversationHistory(
	openAIRequest *openai.ChatCompletionNewParams,
) []ConversationMessage {
	var messagesJSON []byte
	if openAIRequest.Messages != nil {
		messagesJSON, _ = json.Marshal(openAIRequest.Messages)
	}

	history, err := ExtractConversationHistory(messagesJSON)
	if err != nil {
		logging.Warnf("Memory: Failed to extract conversation history: %v", err)
		return []ConversationMessage{}
	}

	return history
}

func (r *OpenAIRouter) buildMemoryRetrieveOptions(
	memoryPluginConfig *config.MemoryPluginConfig,
	searchQuery string,
	userID string,
) memory.RetrieveOptions {
	retrieveLimit := r.Config.Memory.DefaultRetrievalLimit
	retrieveThreshold := r.Config.Memory.DefaultSimilarityThreshold

	if memoryPluginConfig != nil {
		if memoryPluginConfig.RetrievalLimit != nil {
			retrieveLimit = *memoryPluginConfig.RetrievalLimit
		}
		if memoryPluginConfig.SimilarityThreshold != nil {
			retrieveThreshold = *memoryPluginConfig.SimilarityThreshold
		}
	}

	retrieveOpts := memory.RetrieveOptions{
		Query:             searchQuery,
		UserID:            userID,
		Limit:             retrieveLimit,
		Threshold:         retrieveThreshold,
		AdaptiveThreshold: r.Config.Memory.AdaptiveThreshold,
	}
	if memoryPluginConfig != nil && memoryPluginConfig.HybridSearch {
		retrieveOpts.HybridSearch = true
		retrieveOpts.HybridMode = memoryPluginConfig.HybridMode
	} else if r.Config.Memory.HybridSearch {
		retrieveOpts.HybridSearch = true
		retrieveOpts.HybridMode = r.Config.Memory.HybridMode
	}
	if retrieveOpts.Limit <= 0 {
		retrieveOpts.Limit = 5
	}
	if retrieveOpts.Threshold <= 0 {
		retrieveOpts.Threshold = 0.6
	}

	return retrieveOpts
}

func (r *OpenAIRouter) filterRetrievedMemories(
	memoryPluginConfig *config.MemoryPluginConfig,
	memories []*memory.RetrieveResult,
	userID string,
) []*memory.RetrieveResult {
	if len(memories) == 0 {
		logging.Debugf("Memory: no memories found above threshold for user=%s", userID)
		return nil
	}
	logging.Infof("Memory: found %d memories for user=%s", len(memories), userID)

	var perDecisionReflection *config.MemoryReflectionConfig
	if memoryPluginConfig != nil && memoryPluginConfig.Reflection != nil {
		perDecisionReflection = memoryPluginConfig.Reflection
	}
	filter := memory.NewMemoryFilter(r.Config.Memory.Reflection, perDecisionReflection)
	filtered := filter.Filter(memories)
	if len(filtered) == 0 {
		logging.Debugf("Memory: all memories filtered by memory filter for user=%s", userID)
	}
	return filtered
}

func (r *OpenAIRouter) injectRetrievedMemories(
	ctx *RequestContext,
	requestBody []byte,
	memories []*memory.RetrieveResult,
) []byte {
	ctx.MemoryContext = FormatMemoriesAsContext(memories)
	if ctx.MemoryContext == "" {
		return requestBody
	}

	injectedBody, err := injectMemoryMessages(requestBody, ctx.MemoryContext)
	if err != nil {
		logging.Warnf("Memory: Failed to inject memory context: %v", err)
		metrics.RecordPluginExecution("memory", ctx.VSRSelectedDecisionName, "injection_error", 0)
		ctx.MemoryContext = ""
		return requestBody
	}

	metrics.RecordPluginExecution("memory", ctx.VSRSelectedDecisionName, "injected", 0)
	logging.Infof("Memory: Injected %d memories into request (decision=%s, context_len=%d)",
		len(memories), ctx.VSRSelectedDecisionName, len(ctx.MemoryContext))
	return injectedBody
}

func (r *OpenAIRouter) getMemoryStore() memory.Store {
	return r.MemoryStore
}

// getUserIDFromContext extracts user ID from the trusted auth header.
func (r *OpenAIRouter) getUserIDFromContext(ctx *RequestContext) string {
	return extractUserID(ctx)
}

// buildRateLimitContext constructs a ratelimit.Context from the request context.
func (r *OpenAIRouter) buildRateLimitContext(ctx *RequestContext, selectedModel string) ratelimit.Context {
	userID := ctx.Headers[r.Config.Authz.Identity.GetUserIDHeader()]
	groupsStr := ctx.Headers[r.Config.Authz.Identity.GetUserGroupsHeader()]
	var groups []string
	if groupsStr != "" {
		for _, g := range strings.Split(groupsStr, ",") {
			g = strings.TrimSpace(g)
			if g != "" {
				groups = append(groups, g)
			}
		}
	}

	return ratelimit.Context{
		UserID:     userID,
		Groups:     groups,
		Model:      selectedModel,
		Headers:    ctx.Headers,
		TokenCount: ctx.VSRContextTokenCount,
	}
}

// createRateLimitResponse builds a 429 response with standard rate limit headers.
func (r *OpenAIRouter) createRateLimitResponse(decision *ratelimit.Decision) *ext_proc.ProcessingResponse {
	retryAfterSec := "60"
	if decision != nil && decision.RetryAfter > 0 {
		retryAfterSec = fmt.Sprintf("%d", int(decision.RetryAfter.Seconds()))
	}

	body := []byte(fmt.Sprintf(`{"error":{"message":"Rate limit exceeded. Retry after %s seconds.","type":"rate_limit_error","code":429}}`, retryAfterSec))

	respHeaders := []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: "content-type", RawValue: []byte("application/json")}},
		{Header: &core.HeaderValue{Key: "retry-after", RawValue: []byte(retryAfterSec)}},
	}

	if decision != nil {
		respHeaders = append(respHeaders,
			&core.HeaderValueOption{Header: &core.HeaderValue{
				Key: "x-ratelimit-limit", RawValue: []byte(fmt.Sprintf("%d", decision.Limit)),
			}},
			&core.HeaderValueOption{Header: &core.HeaderValue{
				Key: "x-ratelimit-remaining", RawValue: []byte(fmt.Sprintf("%d", decision.Remaining)),
			}},
		)
		if !decision.ResetAt.IsZero() {
			respHeaders = append(respHeaders, &core.HeaderValueOption{Header: &core.HeaderValue{
				Key: "x-ratelimit-reset", RawValue: []byte(fmt.Sprintf("%d", decision.ResetAt.Unix())),
			}})
		}
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{Code: typev3.StatusCode_TooManyRequests},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: respHeaders,
				},
				Body: body,
			},
		},
	}
}

// handleFastResponse returns an immediate response for the fast_response plugin.
func (r *OpenAIRouter) handleFastResponse(ctx *RequestContext, decisionName string) *ext_proc.ProcessingResponse {
	if ctx.VSRSelectedDecision == nil {
		return nil
	}

	fastCfg := ctx.VSRSelectedDecision.GetFastResponseConfig()
	if fastCfg == nil {
		return nil
	}

	logging.Infof("[FastResponse] Decision '%s' has fast_response plugin, returning immediate response", decisionName)
	metrics.RecordPluginExecution("fast_response", decisionName, "executed", 0)

	return httputil.CreateFastResponse(fastCfg.Message, ctx.ExpectStreamingResponse, decisionName)
}

// isMemoryDisabledForRoute checks if memory is disabled for the given route path.
// Returns true if the route is in the Config.Memory.DisabledRoutes list.
func (r *OpenAIRouter) isMemoryDisabledForRoute(requestPath string) bool {
	if len(r.Config.Memory.DisabledRoutes) == 0 {
		return false
	}

	for _, disabledRoute := range r.Config.Memory.DisabledRoutes {
		if requestPath == disabledRoute {
			return true
		}
	}

	return false
}

// isMemoryDisabledForModel checks if memory is disabled for the given model name.
// Returns true if the model is in the Config.Memory.DisabledModels list.
func (r *OpenAIRouter) isMemoryDisabledForModel(modelName string) bool {
	if len(r.Config.Memory.DisabledModels) == 0 {
		return false
	}

	for _, disabledModel := range r.Config.Memory.DisabledModels {
		if modelName == disabledModel {
			return true
		}
	}

	return false
}
