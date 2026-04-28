package config

import "strings"

const (
	// ToolRetrievalStrategyWeighted is the default: embedding + lexical/tag/name/category weights.
	ToolRetrievalStrategyWeighted = "weighted"
	// ToolRetrievalStrategyHybridHistory combines semantic similarity with short-horizon tool history.
	ToolRetrievalStrategyHybridHistory = "hybrid_history"
)

// EffectiveToolRetrievalStrategy returns normalized strategy; empty defaults to weighted.
func EffectiveToolRetrievalStrategy(advanced *AdvancedToolFilteringConfig) string {
	if advanced == nil {
		return ToolRetrievalStrategyWeighted
	}
	s := strings.TrimSpace(strings.ToLower(advanced.RetrievalStrategy))
	if s == "" {
		return ToolRetrievalStrategyWeighted
	}
	return s
}

// IsHybridHistoryRetrieval reports whether advanced filtering should use hybrid_history ranking.
func IsHybridHistoryRetrieval(advanced *AdvancedToolFilteringConfig) bool {
	return EffectiveToolRetrievalStrategy(advanced) == ToolRetrievalStrategyHybridHistory
}

// ResolveHybridHistoryHorizon returns the max assistant tool names to read from history.
func ResolveHybridHistoryHorizon(advanced *AdvancedToolFilteringConfig) int {
	const defaultHorizon = 8
	if advanced == nil || advanced.HybridHistory == nil || advanced.HybridHistory.HistoryHorizon == nil {
		return defaultHorizon
	}
	h := *advanced.HybridHistory.HistoryHorizon
	if h <= 0 {
		return defaultHorizon
	}
	return h
}
