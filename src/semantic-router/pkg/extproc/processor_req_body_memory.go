package extproc

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// retrieveMemorySource recalls the user's memories for a memory source, or
// lists the newest for a judge without recall, keeps them by the source's
// match block after its reflection filter, and injects them. A failed
// retrieval goes on without memories.
func (r *OpenAIRouter) retrieveMemorySource(
	ctx *RequestContext,
	plugin *config.ContextPluginConfig,
	source *config.ContextSource,
	userContent string,
	request *llmprotocol.Request,
	eligible bool,
) error {
	if status, reason, optedOut := r.memoryOptOut(ctx); optedOut {
		logging.Debugf("Memory: skipped (%s)", reason)
		recordMemoryOutcome(ctx, status, reason, false)
		return nil
	}
	store := r.getMemoryStore()
	if store == nil || !store.IsEnabled() {
		logging.Debugf("Memory: Store not available or disabled, skipping retrieval")
		recordMemoryOutcome(ctx, "unavailable", "store_unavailable", true)
		return nil
	}
	logging.Debugf("Memory: retrieval flow query=%s", logging.ContentDescriptor(userContent))
	searchQuery, userID, shouldSearch := r.prepareMemorySearchQuery(ctx, userContent, request, eligible)
	if !shouldSearch {
		return nil
	}
	index := &memoryIndex{router: r, source: source, store: store, userID: userID}
	block := r.Config.MemoryMatch(source)
	result, err := r.matcher().Match(ctx.TraceContext, config.MatchMemory, &block, match.Input{Request: searchQuery, Index: index})
	if err != nil {
		recordMemoryOutcome(ctx, "unavailable", "retrieval_error", true)
		logging.Errorf("Memory: retrieval failed for user=%s decision=%s query=%s error_class=%T",
			userID, ctx.VSRSelectedDecisionName, logging.ContentDescriptor(searchQuery), err)
		// The caller records this error in the stack log. Do not propagate the
		// provider's free-form text because it may echo tenant content or secrets.
		return fmt.Errorf("memory retrieval failed (error_class=%T)", err)
	}
	memories := index.kept(result.Kept)
	if len(memories) == 0 {
		reason := "no_results"
		if index.retrieved > 0 {
			reason = "filtered"
		}
		recordMemoryOutcome(ctx, "missing", reason, false)
		return nil
	}
	r.injectRetrievedMemories(ctx, plugin, memories)
	return nil
}

// memoryOptOut reports a request whose memory the caller or the store's
// routes and models turned off.
func (r *OpenAIRouter) memoryOptOut(ctx *RequestContext) (string, string, bool) {
	switch {
	case strings.EqualFold(strings.TrimSpace(headerValueCI(ctx, headers.VSRMemory)), headers.MemoryOff):
		return "policy_blocked", "request_opt_out", true
	case r.isMemoryDisabledForRoute(ctx.Headers[":path"]):
		return "policy_blocked", "route_config", true
	case ctx.RequestModel != "" && r.isMemoryDisabledForModel(ctx.RequestModel):
		return "policy_blocked", "model_config", true
	}
	return "", "", false
}

// memoryIndex recalls a user's memories from the memory store, bounded by
// recall and filtered by the source's reflection, for a memory source's
// match; a judge without recall reads them listed.
type memoryIndex struct {
	router    *OpenAIRouter
	source    *config.ContextSource
	store     memory.Store
	userID    string
	recalled  []*memory.RetrieveResult
	retrieved int
}

func (x *memoryIndex) Recall(ctx context.Context, query string, recall config.MatchRecall) ([]match.Scored, error) {
	cfg := x.router.Config
	options := memory.RetrieveOptions{
		Query:             query,
		UserID:            x.userID,
		Limit:             recall.TopK,
		AdaptiveThreshold: cfg.Memory.AdaptiveThreshold,
	}
	// Without a bound, the store recalls above its default_similarity_threshold.
	if recall.MinSimilarity != nil {
		options.Threshold, options.ThresholdSet = float32(*recall.MinSimilarity), true
	}
	switch {
	case x.source.HybridSearch:
		options.HybridSearch, options.HybridMode = true, x.source.HybridMode
	case cfg.Memory.HybridSearch:
		options.HybridSearch, options.HybridMode = true, cfg.Memory.HybridMode
	}
	results, err := x.store.Retrieve(ctx, options)
	if err != nil {
		return nil, err
	}
	x.retrieved = len(results)
	x.recalled = memory.NewMemoryFilter(cfg.Memory.Reflection, x.source.Reflection).Filter(results)
	scored := make([]match.Scored, 0, len(x.recalled))
	for index, result := range x.recalled {
		if result == nil || result.Memory == nil {
			continue
		}
		scored = append(scored, match.Scored{
			Candidate:  match.Candidate{ID: strconv.Itoa(index), Text: result.Memory.Content},
			Similarity: float64(result.Score),
		})
	}
	return scored, nil
}

// List lists the user's memories, newest first and at most limit of them,
// for a memory source whose judge reads them without recall, filtered by
// the source's reflection. held counts the user's memories, but the ones
// the reflection dropped.
func (x *memoryIndex) List(ctx context.Context, limit int) ([]match.Candidate, int, error) {
	var listed []*memory.RetrieveResult
	total := 0
	for len(listed) < limit {
		page, err := x.store.List(ctx, memory.ListOptions{UserID: x.userID, Limit: limit - len(listed), Offset: len(listed)})
		if err != nil {
			return nil, 0, err
		}
		total = page.Total
		for _, item := range page.Memories {
			listed = append(listed, &memory.RetrieveResult{Memory: item})
		}
		if len(page.Memories) == 0 || len(listed) >= total {
			break
		}
	}
	x.retrieved = len(listed)
	x.recalled = memory.NewMemoryFilter(x.router.Config.Memory.Reflection, x.source.Reflection).Filter(listed)
	candidates := make([]match.Candidate, 0, len(x.recalled))
	for index, result := range x.recalled {
		if result != nil && result.Memory != nil {
			candidates = append(candidates, match.Candidate{ID: strconv.Itoa(index), Text: result.Memory.Content})
		}
	}
	return candidates, max(total, len(listed)) - (len(listed) - len(candidates)), nil
}

// kept are the recalled memories a match kept, best first.
func (x *memoryIndex) kept(scored []match.Scored) []*memory.RetrieveResult {
	memories := make([]*memory.RetrieveResult, 0, len(scored))
	for _, candidate := range scored {
		if index, err := strconv.Atoi(candidate.ID); err == nil && index < len(x.recalled) {
			memories = append(memories, x.recalled[index])
		}
	}
	return memories
}

func (r *OpenAIRouter) prepareMemorySearchQuery(
	ctx *RequestContext,
	userContent string,
	request *llmprotocol.Request,
	eligible bool,
) (string, string, bool) {
	if !eligible {
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

// extractConversationHistory returns the routing view of the conversation that
// memory query rewriting reads.
func (r *OpenAIRouter) extractConversationHistory(
	request *llmprotocol.Request,
) []ConversationMessage {
	if request == nil {
		return nil
	}
	selector := r.routingInputSelector()
	history := make([]ConversationMessage, 0, len(request.Messages))
	for _, message := range request.Messages {
		text := semanticText(message.Content)
		if message.Role == llmprotocol.RoleUser {
			if textOnlyMessage(message) && selector.SkipsMessage(text) {
				continue
			}
			text, _ = selector.Strip(text)
		}
		if text == "" {
			continue
		}
		history = append(history, ConversationMessage{Role: string(message.Role), Content: text})
	}
	return history
}

func (r *OpenAIRouter) injectRetrievedMemories(
	ctx *RequestContext,
	plugin *config.ContextPluginConfig,
	memories []*memory.RetrieveResult,
) {
	ctx.MemoryContext = FormatMemoriesAsContext(memories)
	if ctx.MemoryContext == "" {
		recordMemoryOutcome(ctx, "missing", "empty_context", false)
		return
	}
	if err := r.injectContext(ctx, plugin, config.ContextSourceMemory, ctx.MemoryContext); err != nil {
		recordMemoryOutcome(ctx, "unavailable", "injection_error", true)
		ctx.MemoryContext = ""
		return
	}
	ctx.MemoryResultCount = len(memories)
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
	fastCfg := config.FastResponsePluginOf(requestPlugins(r.Config, ctx))
	if fastCfg == nil {
		return nil
	}

	logging.Infof("[FastResponse] Decision '%s' has fast_response plugin, returning immediate response", decisionName)
	metrics.RecordPluginExecution("fast_response", requestDecisionStateKey(ctx), "executed", 0)
	ctx.VSRSelectedDecisionName = decisionName

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
	ctx.ImmediateResponseEncoded = true
	return httputil.CreateFastResponseWithBody(body, contentType)
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
