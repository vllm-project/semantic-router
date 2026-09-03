package extproc

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// handleToolSelectionForRequest handles tool selection for the request.
func (r *OpenAIRouter) handleToolSelectionForRequest(request *llmprotocol.Request, response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	fast := extractSemanticRequestSignals(request)
	if err := r.handleToolSelection(request, fast.UserContent, fast.NonUserMessages, &response, ctx); err != nil {
		logging.Errorf("Error in tool selection: %v", err)
		// Continue without failing the request
	}
	if clearSemanticToolChoiceWhenNoTools(request) {
		if err := commitToolSelection(request, ctx); err != nil {
			logging.Errorf("Error clearing invalid tool_choice without tools: %v", err)
		}
	}
}

func (r *OpenAIRouter) applySelectedTools(
	request *llmprotocol.Request,
	selectedTools []llmprotocol.Tool,
	strategyID string,
	confidence float32,
	latency time.Duration,
	classificationText string,
	fallbackToEmptyOverride *bool,
) error {
	fallbackEmpty := r.Config.Tools.FallbackToEmpty
	if fallbackToEmptyOverride != nil {
		fallbackEmpty = *fallbackToEmptyOverride
	}

	if len(selectedTools) == 0 {
		changed := len(request.Tools) > 0
		if fallbackEmpty {
			logging.Infof("No suitable tools found, falling back to no tools")
			request.Tools = nil
		} else {
			logging.Infof("No suitable tools found above threshold")
			request.Tools = []llmprotocol.Tool{}
		}
		if changed {
			request.Generation++
		}
		return nil
	}
	request.Tools = append([]llmprotocol.Tool(nil), selectedTools...)
	request.Generation++
	logging.Infof("Auto-selected %d tools via strategy %q (confidence=%.3f, latency=%s) for query: %s",
		len(selectedTools), strategyID, confidence, latency.Round(time.Millisecond),
		logging.ContentDescriptor(classificationText))
	return nil
}

// handleEarlyToolModes applies the tool plugin mode (none, filtered, passthrough)
// and returns true if the caller should proceed to semantic tool selection.
// If it returns false, the request has already been updated and the caller must return nil.
func (r *OpenAIRouter) handleEarlyToolModes(
	request *llmprotocol.Request,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	toolsCfg *config.ToolsPluginConfig,
) (bool, error) {
	switch toolsCfg.EffectiveMode() {
	case config.ToolsPluginModeNone:
		logging.Infof("[ToolsPlugin] Decision %q has mode=none, stripping all tools", ctx.VSRSelectedDecision.Name)
		changed, removed := stripSemanticToolPolicy(request, toolsCfg.StripToolHistory)
		if changed {
			request.Generation++
		}
		if removed > 0 {
			logging.Infof("[ToolsPlugin] Decision %q stripped %d prior tool-history messages", ctx.VSRSelectedDecision.Name, removed)
		}
		if err := commitToolSelection(request, ctx); err != nil {
			return false, err
		}
		return false, nil

	case config.ToolsPluginModeFiltered:
		filtered := filterToolsByDecisionPolicy(request.Tools, toolsCfg.AllowTools, toolsCfg.BlockTools)
		if len(filtered) != len(request.Tools) {
			request.Generation++
		}
		request.Tools = filtered
		if request.ToolChoice.Mode != llmprotocol.ToolChoiceAuto {
			logging.Infof("[ToolsPlugin] Decision %q filtered explicit tools to %d entries", ctx.VSRSelectedDecision.Name, len(request.Tools))
			if err := commitToolSelection(request, ctx); err != nil {
				return false, err
			}
			return false, nil
		}
		return true, nil

	case config.ToolsPluginModePassthrough:
		return request.ToolChoice.Mode == llmprotocol.ToolChoiceAuto, nil

	default:
		return false, fmt.Errorf("tools plugin: unsupported mode %q", toolsCfg.Mode)
	}
}

// runSemanticToolSelection performs the retrieval, observability reporting,
// fallback handling and final tool application after the earlier mode checks
// have determined that semantic selection should run.
func (r *OpenAIRouter) runSemanticToolSelection(
	request *llmprotocol.Request,
	classificationText string,
	historySummary string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	toolsCfg *config.ToolsPluginConfig,
) error {
	selectedTools, strategyID, confidence, latency, toolErr := r.findToolsForQuery(request, classificationText, historySummary, ctx, toolsCfg)

	emitToolObservability(response, ctx, strategyID, confidence, latency)
	metrics.RecordToolsRetrieval(strategyID, latency.Seconds())

	if toolErr != nil {
		return r.handleToolSelectionError(request, response, ctx, toolErr, r.Config.Tools.FallbackToEmpty)
	}

	if err := r.applySelectedTools(request, selectedTools, strategyID, confidence, latency, classificationText, nil); err != nil {
		return err
	}
	return commitToolSelection(request, ctx)
}

func (r *OpenAIRouter) handleToolSelectionError(
	request *llmprotocol.Request,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	toolErr error,
	fallbackToEmpty bool,
) error {
	if fallbackToEmpty {
		logging.Warnf("Tool selection failed, falling back to no tools: %v", toolErr)
		request.Tools = nil
		request.Generation++
		return commitToolSelection(request, ctx)
	}
	metrics.RecordRequestError(getModelFromCtx(ctx), "classification_failed")
	return toolErr
}

// handleToolSelection handles automatic tool selection based on semantic similarity.
func (r *OpenAIRouter) handleToolSelection(
	request *llmprotocol.Request,
	userContent string,
	nonUserMessages []string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
) error {
	if ctx.VSRSelectedDecision == nil {
		return nil
	}

	tsPlugin := ctx.VSRSelectedDecision.GetToolSelectionConfig()
	toolsCfg := resolveDecisionToolsConfig(ctx)

	handled, err := r.handleToolSelectionDecisionPlugin(request, userContent, nonUserMessages, response, ctx, tsPlugin, toolsCfg)
	if err != nil {
		return err
	}
	if handled {
		return nil
	}

	if toolsCfg == nil || !toolsCfg.Enabled {
		return nil
	}

	shouldContinue, err := r.handleEarlyToolModes(request, response, ctx, toolsCfg)
	if err != nil {
		return err
	}
	if !shouldContinue {
		return nil
	}

	if !toolsCfg.SelectionEnabled() {
		return commitToolSelection(request, ctx)
	}

	classificationText, historySummary, ok := buildToolClassificationText(userContent, nonUserMessages)
	if !ok {
		logging.Infof("No content available for tool classification")
		return nil
	}

	if !r.ToolsDatabase.IsEnabled() {
		logging.Infof("Tools database is disabled")
		return nil
	}

	return r.runSemanticToolSelection(request, classificationText, historySummary, response, ctx, toolsCfg)
}

func (r *OpenAIRouter) handleToolSelectionDecisionPlugin(
	request *llmprotocol.Request,
	userContent string,
	nonUserMessages []string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	tsPlugin *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
) (bool, error) {
	if tsPlugin == nil || !tsPlugin.Enabled {
		return false, nil
	}
	earlyCfg := toolsCfg
	if earlyCfg == nil || !earlyCfg.Enabled {
		earlyCfg = &config.ToolsPluginConfig{Enabled: true, Mode: config.ToolsPluginModePassthrough}
	}

	shouldContinue, err := r.handleEarlyToolModes(request, response, ctx, earlyCfg)
	if err != nil {
		return false, err
	}
	if !shouldContinue {
		return true, nil
	}

	classificationText, historySummary, ok := buildToolClassificationText(userContent, nonUserMessages)
	if !ok {
		logging.Infof("No content available for tool classification")
		return true, nil
	}

	mode := strings.TrimSpace(tsPlugin.Mode)
	if mode == "" {
		mode = config.ToolSelectionModeAdd
	}

	switch mode {
	case config.ToolSelectionModeFilter:
		return true, r.runToolSelectionPluginFilter(request, classificationText, response, ctx, tsPlugin)
	case config.ToolSelectionModeAdd:
		return true, r.runToolSelectionPluginAdd(request, classificationText, historySummary, response, ctx, tsPlugin, toolsCfg)
	default:
		return false, fmt.Errorf("tool_selection plugin: unsupported mode %q", mode)
	}
}

func (r *OpenAIRouter) retrievalViaEmbeddingDatabase(strategyID string, in tools.RetrievalInput, db *tools.ToolsDatabase) (tools.RetrievalResult, error) {
	pool := in.EffectivePoolSize()
	results, err := db.FindSimilarToolsWithScoresMinSimilarity(in.Query, pool, in.MinSimilarity)
	if err != nil {
		return tools.RetrievalResult{StrategyID: strategyID}, err
	}
	confidence := float32(0)
	if len(results) > 0 {
		confidence = results[0].Similarity
	}
	return tools.RetrievalResult{
		Tools:      results,
		Confidence: confidence,
		StrategyID: strategyID,
	}, nil
}

// executeRetrieval resolves the retriever for strategyID from the registry and
// runs it with in. If the registry is nil or the strategy is not found, it falls
// back to ToolsDatabase directly and marks the returned StrategyID with "-fallback".
//
// scopedEmbDB, when non-nil, forces retrieval against that database via embedding
// similarity (registry strategies are skipped).
func (r *OpenAIRouter) executeRetrieval(ctx context.Context, strategyID string, in tools.RetrievalInput, scopedEmbDB *tools.ToolsDatabase) (tools.RetrievalResult, error) {
	if scopedEmbDB != nil {
		return r.retrievalViaEmbeddingDatabase(strategyID+"-scoped", in, scopedEmbDB)
	}
	if r.ToolsRegistry != nil {
		if retriever, ok := r.ToolsRegistry.Get(strategyID); ok {
			return retriever.Retrieve(ctx, in)
		}

		// Named strategy not found — try "default" before falling back to DB
		if retriever, ok := r.ToolsRegistry.Get(tools.StrategyDefault); ok {
			logging.Warnf("[ToolsPlugin] strategy %q not found, falling back to default", strategyID)
			result, err := retriever.Retrieve(ctx, in)
			result.StrategyID = strategyID + "-fallback"
			return result, err
		}

		logging.Warnf("[ToolsPlugin] strategy %q not found in registry, using ToolsDatabase directly", strategyID)
	} else {
		logging.Warnf("[ToolsPlugin] registry not initialized for strategy %q, using ToolsDatabase directly", strategyID)
	}

	pool := in.EffectivePoolSize()
	results, err := r.ToolsDatabase.FindSimilarToolsWithScoresMinSimilarity(in.Query, pool, in.MinSimilarity)
	if err != nil {
		return tools.RetrievalResult{StrategyID: strategyID + "-fallback"}, err
	}

	confidence := float32(0)
	if len(results) > 0 {
		confidence = results[0].Similarity
	}

	return tools.RetrievalResult{
		Tools:      results,
		Confidence: confidence,
		StrategyID: strategyID + "-fallback",
	}, nil
}

// findToolsForQuery resolves the retriever strategy from the registry, executes
// retrieval, applies advanced filtering, and returns the final tool list together
// with the strategy name, top-1 confidence score, and retrieval wall-clock time.
func (r *OpenAIRouter) findToolsForQuery(
	request *llmprotocol.Request,
	query, historySummary string,
	ctx *RequestContext,
	toolsCfg *config.ToolsPluginConfig,
) ([]llmprotocol.Tool, string, float32, time.Duration, error) {
	topK := r.Config.Tools.TopK
	if topK <= 0 {
		topK = 3
	}
	advanced := mergeAdvancedToolFiltering(r.Config.Tools.AdvancedFiltering, toolsCfg)
	return r.findToolsForQueryExt(request, query, historySummary, ctx, toolsCfg, topK, advanced, toolsCfg.EffectiveStrategy(), nil, r.Config.Tools.SimilarityThreshold)
}

func (r *OpenAIRouter) findToolsForQueryExt(
	request *llmprotocol.Request,
	query, historySummary string,
	ctx *RequestContext,
	toolsCfg *config.ToolsPluginConfig,
	topK int,
	advanced *config.AdvancedToolFilteringConfig,
	strategyID string,
	scopedEmbDB *tools.ToolsDatabase,
	minSimilarity *float32,
) ([]llmprotocol.Tool, string, float32, time.Duration, error) {
	if topK <= 0 {
		topK = 3
	}

	poolSize := resolveCandidatePoolSize(r.Config.Tools.AdvancedFiltering, topK)
	if advanced != nil && advanced.Enabled {
		poolSize = resolveCandidatePoolSize(advanced, topK)
	}

	retrievalIn := newToolRetrievalInput(query, historySummary, topK, poolSize, ctx, minSimilarity)

	retrievalCtx := context.Background()
	if ctx != nil && ctx.TraceContext != nil {
		retrievalCtx = ctx.TraceContext
	}

	start := time.Now()
	retrieved, err := r.executeRetrieval(retrievalCtx, strategyID, retrievalIn, scopedEmbDB)
	latency := time.Since(start)

	if err != nil {
		return nil, retrieved.StrategyID, 0, latency, err
	}

	if advanced == nil || !advanced.Enabled {
		selected, convertErr := selectTopKTools(retrieved.Tools, topK)
		return selected, retrieved.StrategyID, retrieved.Confidence, latency, convertErr
	}

	if config.IsHybridHistoryRetrieval(advanced) {
		transition := extractToolTransitionContextFromRequest(request, config.ResolveHybridHistoryHorizon(advanced), ctx)
		decisionConfidence := float64(0)
		if ctx != nil {
			decisionConfidence = float64(ctx.VSRSelectedDecisionConfidence)
		}
		selected := tools.FilterAndRankToolsWithConversation(
			query,
			retrieved.Tools,
			topK,
			advanced,
			resolveCategory(advanced, ctx),
			transition.RecentToolNames,
			decisionConfidence,
		)
		semantic, convertErr := tools.SemanticTools(selected)
		return semantic, retrieved.StrategyID, retrieved.Confidence, latency, convertErr
	}

	selected := tools.FilterAndRankTools(query, retrieved.Tools, topK, advanced, resolveCategory(advanced, ctx))
	semantic, convertErr := tools.SemanticTools(selected)
	return semantic, retrieved.StrategyID, retrieved.Confidence, latency, convertErr
}

// emitToolObservability writes the three x-vsr-tools-* headers into the
// in-flight ProcessingResponse so they reach the client as response headers.
// Per the v0.4 contract (#2205) these are intermediate observability details
// demoted off the default surface: they are emitted only when the request opted
// into debug headers via x-vsr-debug. The Prometheus tools-retrieval metric is
// recorded by the caller and stays unconditional.
func emitToolObservability(response **ext_proc.ProcessingResponse, ctx *RequestContext, strategyID string, confidence float32, latency time.Duration) {
	if response == nil || *response == nil {
		return
	}
	if !debugHeadersRequested(ctx) {
		return
	}
	commonResponse := ensureRequestBodyCommonResponse(response)
	if commonResponse.HeaderMutation == nil {
		commonResponse.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	setHeaderValue(commonResponse.HeaderMutation, headers.VSRToolsStrategy, strategyID)
	setHeaderValue(commonResponse.HeaderMutation, headers.VSRToolsConfidence, strconv.FormatFloat(float64(confidence), 'f', 4, 32))
	setHeaderValue(commonResponse.HeaderMutation, headers.VSRToolsLatencyMs, strconv.FormatInt(latency.Milliseconds(), 10))
}

// selectTopKTools returns the top-K entries from candidates by similarity
// without applying advanced filtering.
func selectTopKTools(candidates []tools.ToolSimilarity, topK int) ([]llmprotocol.Tool, error) {
	return tools.SemanticToolsFromCandidates(candidates, topK)
}

func resolveCandidatePoolSize(advanced *config.AdvancedToolFilteringConfig, topK int) int {
	if advanced == nil {
		return max(topK*tools.DefaultCandidatePoolMultiplier, tools.DefaultCandidatePoolMin)
	}
	var size int
	switch {
	case advanced.CandidatePoolSize != nil && *advanced.CandidatePoolSize > 0:
		size = *advanced.CandidatePoolSize
	case advanced.CandidatePoolSize == nil:
		size = max(topK*tools.DefaultCandidatePoolMultiplier, tools.DefaultCandidatePoolMin)
	default:
		size = topK
	}
	if size < topK {
		size = topK
	}
	return size
}

func newToolRetrievalInput(
	query, historySummary string,
	topK, poolSize int,
	ctx *RequestContext,
	minSimilarity *float32,
) tools.RetrievalInput {
	in := tools.RetrievalInput{
		Query:          query,
		HistorySummary: historySummary,
		TopK:           topK,
		PoolSize:       poolSize,
		MinSimilarity:  minSimilarity,
	}
	if ctx == nil {
		return in
	}
	in.Category = ctx.VSRSelectedCategory
	in.DecisionName = ctx.VSRSelectedDecisionName
	if ctx.VSRSelectedDecision != nil && in.DecisionName == "" {
		in.DecisionName = ctx.VSRSelectedDecision.Name
	}
	in.DecisionConfidence = ctx.VSRSelectedDecisionConfidence
	return in
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

// filterToolsByDecisionPolicy applies per-decision allow/block lists to the
// request tool set. If allowTools is non-empty, only tools whose function name
// appears in the allow list are kept. Any tool whose name appears in blockTools
// is removed regardless.
func filterToolsByDecisionPolicy(tools []llmprotocol.Tool, allowTools, blockTools []string) []llmprotocol.Tool {
	allowSet := make(map[string]bool, len(allowTools))
	for _, t := range allowTools {
		allowSet[t] = true
	}
	blockSet := make(map[string]bool, len(blockTools))
	for _, t := range blockTools {
		blockSet[t] = true
	}
	filtered := make([]llmprotocol.Tool, 0, len(tools))
	for _, tool := range tools {
		name := tool.Name
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

// commitToolSelection publishes the mutated neutral request to the dispatch
// context. Provider serialization deliberately remains in
// finalizeProviderDispatchResponse, after every semantic plugin has run.
func commitToolSelection(request *llmprotocol.Request, ctx *RequestContext) error {
	if request == nil || ctx == nil {
		return fmt.Errorf("neutral inference request is unavailable")
	}
	ctx.SemanticRequest = request
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

func setHeaderValue(mutation *ext_proc.HeaderMutation, key, value string) {
	for _, option := range mutation.SetHeaders {
		if option.Header != nil && option.Header.Key == key {
			option.Header.RawValue = []byte(value)
			option.Header.Value = ""
			return
		}
	}
	mutation.SetHeaders = append(mutation.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	})
}
