package extproc

import (
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func buildToolClassificationText(userContent string, nonUserMessages []string) (classificationText string, historySummary string, ok bool) {
	if len(userContent) > 0 {
		classificationText = userContent
	} else if len(nonUserMessages) > 0 {
		classificationText = strings.Join(nonUserMessages, " ")
	}
	if len(nonUserMessages) > 0 {
		historySummary = strings.Join(nonUserMessages, " ")
	}
	if historySummary == classificationText {
		historySummary = ""
	}
	if strings.TrimSpace(classificationText) == "" {
		return "", "", false
	}
	return classificationText, historySummary, true
}

func (r *OpenAIRouter) effectiveToolSelectionFallback(plugin *config.ToolSelectionPluginConfig) bool {
	if plugin != nil && plugin.FallbackToEmpty != nil {
		return *plugin.FallbackToEmpty
	}
	return r.Config.Tools.FallbackToEmpty
}

func mergeToolSelectionAdvanced(plugin *config.ToolSelectionPluginConfig, global *config.AdvancedToolFilteringConfig, toolsCfg *config.ToolsPluginConfig) *config.AdvancedToolFilteringConfig {
	base := global
	if plugin != nil && plugin.AdvancedFiltering != nil {
		base = plugin.AdvancedFiltering
	}
	return mergeAdvancedToolFiltering(base, toolsCfg)
}

func effectivePluginToolTopK(plugin *config.ToolSelectionPluginConfig, global int) int {
	if plugin != nil && plugin.TopK > 0 {
		return plugin.TopK
	}
	if global > 0 {
		return global
	}
	return 3
}

func (r *OpenAIRouter) runToolSelectionPluginAdd(
	request *llmprotocol.Request,
	classificationText, historySummary string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	ts *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
) error {
	db, forceDirectEmbedding, err := r.toolDatabaseForSelectionPlugin(ts)
	if err != nil {
		return r.handleToolSelectionError(request, response, ctx, err, r.effectiveToolSelectionFallback(ts))
	}
	if db == nil || !db.IsEnabled() {
		logging.Infof("[tool_selection] add mode skipped: database disabled or unloaded")
		return nil
	}

	topK := effectivePluginToolTopK(ts, r.Config.Tools.TopK)
	strategyID := ts.EffectiveStrategy()
	minSim := ts.SimilarityThreshold
	if minSim == nil {
		minSim = r.Config.Tools.SimilarityThreshold
	}
	advanced := mergeToolSelectionAdvanced(ts, r.Config.Tools.AdvancedFiltering, toolsCfg)

	var scopedDB *tools.ToolsDatabase
	if forceDirectEmbedding {
		scopedDB = db
	}

	sticky := r.newStickyToolScope(request, ctx, ts, toolsCfg, stickyCatalogFromDatabase(db))
	selectedTools, strategyOut, confidence, latency, toolErr := r.findToolsForQueryExt(
		request,
		classificationText,
		historySummary,
		ctx,
		toolsCfg,
		topK,
		advanced,
		strategyID,
		scopedDB,
		minSim,
	)
	return r.finalizeToolSelection(request, response, ctx, toolSelectionResult{
		tools: selectedTools, strategyID: strategyOut, confidence: confidence, latency: latency,
		classificationText: classificationText, err: toolErr,
		fallbackOverride: ts.FallbackToEmpty, errorFallbackToEmpty: r.effectiveToolSelectionFallback(ts),
	}, sticky)
}

// stickyCatalogFromDatabase snapshots the database's current definitions for
// sticky eligibility and rehydration. A definition that cannot cross the
// provider boundary is not eligible.
func stickyCatalogFromDatabase(db *tools.ToolsDatabase) []llmprotocol.Tool {
	catalog := make([]llmprotocol.Tool, 0)
	for _, tool := range db.GetAllTools() {
		semantic, err := tools.SemanticTool(tool)
		if err != nil {
			continue
		}
		catalog = append(catalog, semantic)
	}
	return catalog
}

// toolSelectionFilterThreshold is filter mode's effective relevance threshold.
func toolSelectionFilterThreshold(ts *config.ToolSelectionPluginConfig) float32 {
	if ts.RelevanceThreshold != nil {
		return *ts.RelevanceThreshold
	}
	return 0.25
}

func (r *OpenAIRouter) runToolSelectionPluginFilter(
	request *llmprotocol.Request,
	classificationText string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	ts *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
) error {
	// A sticky decision ranks only the offered tools it may emit, so an
	// ineligible tool never takes part in relevance.
	offered := request.Tools
	sticky := r.newStickyToolScope(request, ctx, ts, toolsCfg, request.Tools)
	if sticky != nil {
		offered = sticky.eligible
	}
	start := time.Now()
	// The embedder (and its remote provider) is built once per router, not per
	// request; a nil embedder (provider construction failed at startup) errors
	// inside the filter and lands in the configured fallback below.
	filtered, confidence, ferr := filterRequestToolsAgainstQuerySemantic(
		ctx.embeddingContext(),
		classificationText,
		offered,
		r.toolEmbedder,
		toolSelectionFilterThreshold(ts),
		ts.PreserveCount,
	)
	return r.finalizeToolSelection(request, response, ctx, toolSelectionResult{
		tools: filtered, strategyID: config.ToolSelectionModeFilter, confidence: confidence,
		latency: time.Since(start), classificationText: classificationText, err: ferr,
		fallbackOverride: ts.FallbackToEmpty, errorFallbackToEmpty: r.effectiveToolSelectionFallback(ts),
	}, sticky)
}
