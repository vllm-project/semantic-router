//go:build !windows && cgo

package apiserver

import "net/http"

type ClassificationMetricsResponse struct {
	UnifiedClassifier        bool           `json:"unified_classifier"`
	FactCheckClassifier      bool           `json:"fact_check_classifier"`
	HallucinationDetector    bool           `json:"hallucination_detector"`
	HallucinationExplainer   bool           `json:"hallucination_explainer"`
	FeedbackDetector         bool           `json:"feedback_detector"`
	DecisionCount            int            `json:"decision_count"`
	ProjectionPartitionCount int            `json:"projection_partition_count"`
	ProjectionScoreCount     int            `json:"projection_score_count"`
	ProjectionMappingCount   int            `json:"projection_mapping_count"`
	SignalCounts             map[string]int `json:"signal_counts"`
	ClassificationConfigAPI  bool           `json:"classification_config_api"`
}

func (s *ClassificationAPIServer) handleClassificationMetrics(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	response := ClassificationMetricsResponse{
		UnifiedClassifier:       s.classificationSvc.HasUnifiedClassifier(),
		FactCheckClassifier:     s.classificationSvc.HasFactCheckClassifier(),
		HallucinationDetector:   s.classificationSvc.HasHallucinationDetector(),
		HallucinationExplainer:  s.classificationSvc.HasHallucinationExplainer(),
		FeedbackDetector:        s.classificationSvc.HasFeedbackDetector(),
		ClassificationConfigAPI: true,
		SignalCounts:            map[string]int{},
	}
	if cfg == nil {
		s.writeJSONResponse(w, http.StatusOK, response)
		return
	}

	response.DecisionCount = len(cfg.Decisions)
	response.ProjectionPartitionCount = len(cfg.Projections.Partitions)
	response.ProjectionScoreCount = len(cfg.Projections.Scores)
	response.ProjectionMappingCount = len(cfg.Projections.Mappings)
	response.SignalCounts = map[string]int{
		"domains":               len(cfg.Categories),
		"keywords":              len(cfg.KeywordRules),
		"embeddings":            len(cfg.EmbeddingRules),
		"fact_check":            len(cfg.FactCheckRules),
		"user_feedback":         len(cfg.UserFeedbackRules),
		"preferences":           len(cfg.PreferenceRules),
		"language":              len(cfg.LanguageRules),
		"context":               len(cfg.ContextRules),
		"complexity":            len(cfg.ComplexityRules),
		"jailbreak":             len(cfg.JailbreakRules),
		"pii":                   len(cfg.PIIRules),
		"projection_partitions": len(cfg.Projections.Partitions),
		"projection_scores":     len(cfg.Projections.Scores),
		"projection_mappings":   len(cfg.Projections.Mappings),
		"routing_models":        len(cfg.ModelConfig),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}
