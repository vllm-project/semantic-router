package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// handleMemoryRetrieval retrieves relevant memories and injects them into the request.
// Per-decision plugin config takes precedence over global config.
func (r *OpenAIRouter) handleMemoryRetrieval(
	ctx *RequestContext,
	userContent string,
	request *llmprotocol.Request,
) error {
	if requestBypassesRouting(ctx) {
		return nil
	}
	ctx.MemoryBackend = r.Config.Memory.Backend
	if ctx.MemoryBackend == "" {
		ctx.MemoryBackend = "milvus"
	}
	memoryPluginConfig, shouldRetrieve := r.resolveMemoryPluginConfig(ctx)
	if !shouldRetrieve {
		return nil
	}
	store := r.getMemoryStore()
	if store == nil || !store.IsEnabled() {
		logging.Debugf("Memory: Store not available or disabled, skipping retrieval")
		recordMemoryOutcome(ctx, "unavailable", "store_unavailable", true)
		return nil
	}

	logging.Debugf("Memory: retrieval flow query=%s", logging.ContentDescriptor(userContent))
	searchQuery, userID, shouldSearch := r.prepareMemorySearchQuery(ctx, userContent, request)
	if !shouldSearch {
		return nil
	}
	retrieveOpts := r.buildMemoryRetrieveOptions(memoryPluginConfig, searchQuery, userID)
	memories, err := store.Retrieve(ctx.TraceContext, retrieveOpts)
	if err != nil {
		recordMemoryOutcome(ctx, "unavailable", "retrieval_error", true)
		logging.Errorf("Memory: retrieval failed for user=%s decision=%s query=%s error_class=%T",
			userID, ctx.VSRSelectedDecisionName, logging.ContentDescriptor(searchQuery), err)
		// The caller records this error in the stack log. Do not propagate the
		// provider's free-form text because it may echo tenant content or secrets.
		return fmt.Errorf("memory retrieval failed (error_class=%T)", err)
	}
	retrievedCount := len(memories)
	memories = r.filterRetrievedMemories(memoryPluginConfig, memories, userID)
	if len(memories) == 0 {
		reason := "no_results"
		if retrievedCount > 0 {
			reason = "filtered"
		}
		recordMemoryOutcome(ctx, "missing", reason, false)
		return nil
	}
	r.injectRetrievedMemories(ctx, request, memories)
	return nil
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
			recordMemoryOutcome(ctx, "disabled", "decision_config", false)
			return memoryPluginConfig, false
		}
	} else if !memoryEnabled {
		logging.Debugf("Memory: Disabled in global config, skipping retrieval")
		recordMemoryOutcome(ctx, "disabled", "global_config", false)
		return nil, false
	}

	// Check opt-out header (x-disable-router-memory: true)
	if ctx.Headers[headers.DisableRouterMemory] == "true" {
		logging.Debugf("Memory: Disabled via x-disable-router-memory header (SDK-managed memory opt-out)")
		recordMemoryOutcome(ctx, "policy_blocked", "request_opt_out", false)
		return memoryPluginConfig, false
	}

	// Check config-based per-route disable
	requestPath := ctx.Headers[":path"]
	if r.isMemoryDisabledForRoute(requestPath) {
		logging.Debugf("Memory: Disabled for route %s via config (SDK-managed memory opt-out)", requestPath)
		recordMemoryOutcome(ctx, "policy_blocked", "route_config", false)
		return memoryPluginConfig, false
	}

	// Check config-based per-model disable
	if ctx.RequestModel != "" && r.isMemoryDisabledForModel(ctx.RequestModel) {
		logging.Debugf("Memory: Disabled for model %s via config (SDK-managed memory opt-out)", ctx.RequestModel)
		recordMemoryOutcome(ctx, "policy_blocked", "model_config", false)
		return memoryPluginConfig, false
	}

	return memoryPluginConfig, true
}

func (r *OpenAIRouter) prepareMemorySearchQuery(
	ctx *RequestContext,
	userContent string,
	request *llmprotocol.Request,
) (string, string, bool) {
	if !ShouldSearchMemory(ctx, userContent) {
		logging.Debugf("Memory: skipping search (query type not suitable)")
		recordMemoryOutcome(ctx, "ignored", "query_ineligible", false)
		return "", "", false
	}

	history := r.extractConversationHistory(request)
	searchQuery, err := BuildSearchQuery(ctx.TraceContext, history, userContent, r.Config)
	if err != nil {
		logging.Warnf("Memory: Query rewriting failed, using original query (error_class=%T)", err)
		ctx.MemoryFailOpen = true
		ctx.MemoryFallbackReason = "query_rewrite_error"
		searchQuery = userContent
	}

	userID := r.getUserIDFromContext(ctx)
	if userID == "" {
		logging.Debugf("Memory: no user ID, skipping search")
		recordMemoryOutcome(ctx, "missing", "user_id_missing", false)
		return "", "", false
	}

	return searchQuery, userID, true
}

func (r *OpenAIRouter) extractConversationHistory(
	request *llmprotocol.Request,
) []ConversationMessage {
	if request == nil {
		return nil
	}
	history := make([]ConversationMessage, 0, len(request.Messages))
	for _, message := range request.Messages {
		text := semanticText(message.Content)
		if text == "" {
			continue
		}
		history = append(history, ConversationMessage{Role: string(message.Role), Content: text})
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
	logging.Debugf("Memory: found %d memories for user=%s", len(memories), userID)

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
	request *llmprotocol.Request,
	memories []*memory.RetrieveResult,
) {
	ctx.MemoryContext = FormatMemoriesAsContext(memories)
	if ctx.MemoryContext == "" {
		recordMemoryOutcome(ctx, "missing", "empty_context", false)
		return
	}
	if request == nil {
		recordMemoryOutcome(ctx, "unavailable", "injection_error", true)
		ctx.MemoryContext = ""
		return
	}
	memoryMessage := llmprotocol.Message{
		Role:    llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: ctx.MemoryContext}},
	}
	request.Messages = append([]llmprotocol.Message{memoryMessage}, request.Messages...)
	request.Generation++
	messageIndex := 0
	ctx.MemoryResultCount = len(memories)
	if ctx.MemoryMessageIndexes == nil {
		ctx.MemoryMessageIndexes = make(map[int]struct{})
	}
	ctx.MemoryMessageIndexes[messageIndex] = struct{}{}
	recordMemoryOutcome(ctx, "used", "injected", false)
	logging.Debugf("Memory: Injected %d memories (decision=%s, context_len=%d)",
		len(memories), ctx.VSRSelectedDecisionName, len(ctx.MemoryContext))
}

func recordMemoryOutcome(ctx *RequestContext, status, reason string, failOpen bool) {
	ctx.MemoryStatus = status
	ctx.MemoryReason = reason
	ctx.MemoryFailOpen = ctx.MemoryFailOpen || failOpen
	if failOpen {
		ctx.MemoryFallbackReason = reason
	}
	metrics.RecordPluginExecution("memory", requestDecisionStateKey(ctx), status, 0)
}

func (r *OpenAIRouter) getMemoryStore() memory.Store {
	return r.MemoryStore
}

// getUserIDFromContext returns the Router-authenticated tenant user.
func (r *OpenAIRouter) getUserIDFromContext(ctx *RequestContext) string {
	return extractUserID(ctx)
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
	metrics.RecordPluginExecution("fast_response", requestDecisionStateKey(ctx), "executed", 0)

	body, contentType, _, err := r.encodeSyntheticTextResponse(
		ctx,
		fastCfg.Message,
		ctx.ExpectStreamingResponse,
	)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "fast_response_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return r.createErrorResponse(500, "Fast response could not be encoded")
	}
	response := httputil.CreateFastResponseWithBody(body, contentType, decisionName)
	ctx.ImmediateResponseEncoded = true
	appendRecipeHeaderToImmediateResponse(response, ctx)
	return response
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
