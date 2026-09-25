package extproc

import (
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	request *llmprotocol.Request,
	classificationText, historySummary string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	ts *config.ToolSelectionPluginConfig,
	toolsCfg *config.ToolsPluginConfig,
) error {
	emptyQuery := strings.TrimSpace(classificationText) == "" && strings.TrimSpace(historySummary) == ""
	db, forceDirectEmbedding, err := r.toolDatabaseForSelectionPlugin(ts)
	if err != nil {
		if emptyQuery {
			emitStickyToolSelectionAdapterFallback("catalog_unavailable")
			return nil
		}
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

	var selectedTools []llmprotocol.Tool
	var strategyOut string
	var confidence float32
	var latency time.Duration
	var toolErr error
	if emptyQuery {
		// There is no request text to rank. A sticky turn still needs the
		// current authorized catalog so historical identities can be
		// revalidated, but it must not invoke the embedding provider.
		strategyOut = strategyID
	} else {
		selectedTools, strategyOut, confidence, latency, toolErr = r.findToolsForQueryExt(
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
	}

	emitToolObservability(response, ctx, strategyOut, confidence, latency)
	metrics.RecordToolsRetrieval(strategyOut, latency.Seconds())

	if toolErr != nil {
		return r.handleToolSelectionError(request, response, ctx, toolErr, r.effectiveToolSelectionFallback(ts))
	}
	if ts.Sticky != nil && ts.Sticky.Enabled {
		toolSnapshot, retrievalFingerprint := db.Snapshot()
		authorizedTools, catalogErr := tools.SemanticTools(toolSnapshot)
		if catalogErr == nil {
			if toolsCfg != nil && toolsCfg.Enabled && toolsCfg.EffectiveMode() == config.ToolsPluginModeFiltered {
				authorizedTools = filterToolsByDecisionPolicy(
					authorizedTools,
					toolsCfg.AllowTools,
					toolsCfg.BlockTools,
				)
			}
			authorizedTools = filterStickyToolCatalog(authorizedTools, advanced)
			policy := effectiveStickyAddPolicy(
				ts,
				topK,
				minSim,
				advanced,
				r.effectiveToolSelectionFallback(ts),
			)
			if emptyQuery {
				var committed bool
				selectedTools, committed = r.applyStickyToolSelectionWithStatusAndRetrieval(
					request,
					authorizedTools,
					selectedTools,
					policy,
					toolsCfg,
					strategyOut,
					ctx,
					retrievalFingerprint,
				)
				if !committed {
					return nil
				}
			} else {
				selectedTools, _ = r.applyStickyToolSelectionWithStatusAndRetrieval(
					request,
					authorizedTools,
					selectedTools,
					policy,
					toolsCfg,
					strategyOut,
					ctx,
					retrievalFingerprint,
				)
			}
		} else {
			emitStickyToolSelectionAdapterFallback("catalog_projection_failed")
			if emptyQuery {
				return nil
			}
		}
	}

	if err := r.applySelectedTools(request, selectedTools, strategyOut, confidence, latency, classificationText, ts.FallbackToEmpty); err != nil {
		return err
	}
	return commitToolSelection(request, ctx)
}

func (r *OpenAIRouter) runToolSelectionPluginFilter(
	request *llmprotocol.Request,
	classificationText string,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	ts *config.ToolSelectionPluginConfig,
) error {
	authorizedTools := append([]llmprotocol.Tool(nil), request.Tools...)
	thresh := float32(0.25)
	if ts.RelevanceThreshold != nil {
		thresh = *ts.RelevanceThreshold
	}

	emptyQuery := strings.TrimSpace(classificationText) == ""
	var filtered []llmprotocol.Tool
	var confidence float32
	var ferr error
	var latency time.Duration
	if emptyQuery {
		// Empty turns still pass the current authorized request tools through
		// the sticky manager, but must not invoke the embedding provider.
		filtered = append([]llmprotocol.Tool(nil), authorizedTools...)
	} else {
		start := time.Now()
		// The embedder (and its remote provider) is built once per router, not per
		// request; a nil embedder (provider construction failed at startup) errors
		// inside the filter and lands in the configured fallback below.
		filtered, confidence, ferr = filterRequestToolsAgainstQuerySemantic(
			ctx.embeddingContext(),
			classificationText,
			authorizedTools,
			r.toolEmbedder,
			thresh,
			ts.PreserveCount,
		)
		latency = time.Since(start)
	}

	strategyLabel := config.ToolSelectionModeFilter
	emitToolObservability(response, ctx, strategyLabel, confidence, latency)
	metrics.RecordToolsRetrieval(strategyLabel, latency.Seconds())

	if ferr != nil {
		return r.handleToolSelectionError(request, response, ctx, ferr, r.effectiveToolSelectionFallback(ts))
	}
	if ts.Sticky != nil && ts.Sticky.Enabled {
		policy := effectiveStickyFilterPolicy(ts, r.effectiveToolSelectionFallback(ts))
		retrievalFingerprint := toolsEmbeddingProviderIdentity(r.Config)
		if emptyQuery {
			var committed bool
			filtered, committed = r.applyStickyToolSelectionWithStatusAndRetrieval(
				request,
				authorizedTools,
				filtered,
				policy,
				resolveDecisionToolsConfig(ctx),
				strategyLabel,
				ctx,
				retrievalFingerprint,
			)
			if !committed {
				return nil
			}
		} else {
			filtered, _ = r.applyStickyToolSelectionWithStatusAndRetrieval(
				request,
				authorizedTools,
				filtered,
				policy,
				resolveDecisionToolsConfig(ctx),
				strategyLabel,
				ctx,
				retrievalFingerprint,
			)
		}
	}

	if err := r.applySelectedTools(request, filtered, strategyLabel, confidence, latency, classificationText, ts.FallbackToEmpty); err != nil {
		return err
	}
	return commitToolSelection(request, ctx)
}
