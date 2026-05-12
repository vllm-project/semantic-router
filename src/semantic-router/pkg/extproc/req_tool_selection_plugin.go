package extproc

import (
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
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
	openAIRequest *openai.ChatCompletionNewParams,
	classificationText, historySummary string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	ts *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
) error {
	db, forceDirectEmbedding, err := r.toolDatabaseForSelectionPlugin(ts)
	if err != nil {
		return r.handleToolSelectionError(openAIRequest, response, ctx, err, r.effectiveToolSelectionFallback(ts))
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

	selectedTools, strategyOut, confidence, latency, toolErr := r.findToolsForQueryExt(
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

	emitToolObservability(response, strategyOut, confidence, latency)
	metrics.RecordToolsRetrieval(strategyOut, latency.Seconds())

	if toolErr != nil {
		return r.handleToolSelectionError(openAIRequest, response, ctx, toolErr, r.effectiveToolSelectionFallback(ts))
	}

	if err := r.applySelectedTools(openAIRequest, selectedTools, strategyOut, confidence, latency, classificationText, ts.FallbackToEmpty); err != nil {
		return err
	}
	return r.updateRequestWithTools(openAIRequest, response, ctx)
}

func (r *OpenAIRouter) runToolSelectionPluginFilter(
	openAIRequest *openai.ChatCompletionNewParams,
	classificationText string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	ts *config.ToolSelectionPluginConfig,
) error {
	em := r.Config.EmbeddingModels
	thresh := float32(0.25)
	if ts.RelevanceThreshold != nil {
		thresh = *ts.RelevanceThreshold
	}

	start := time.Now()
	filtered, confidence, ferr := filterRequestToolsAgainstQuerySemantic(
		classificationText,
		openAIRequest.Tools,
		em.EmbeddingConfig.ModelType,
		em.EmbeddingConfig.TargetDimension,
		thresh,
		ts.PreserveCount,
	)
	latency := time.Since(start)

	strategyLabel := config.ToolSelectionModeFilter
	emitToolObservability(response, strategyLabel, confidence, latency)
	metrics.RecordToolsRetrieval(strategyLabel, latency.Seconds())

	if ferr != nil {
		return r.handleToolSelectionError(openAIRequest, response, ctx, ferr, r.effectiveToolSelectionFallback(ts))
	}

	if err := r.applySelectedTools(openAIRequest, filtered, strategyLabel, confidence, latency, classificationText, ts.FallbackToEmpty); err != nil {
		return err
	}
	return r.updateRequestWithTools(openAIRequest, response, ctx)
}
