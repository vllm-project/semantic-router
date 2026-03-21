//go:build !windows && cgo

package apiserver

import "net/http"

type ClassificationMetricsResponse struct {
	UnifiedClassifier       bool           `json:"unified_classifier"`
	FactCheckClassifier     bool           `json:"fact_check_classifier"`
	HallucinationDetector   bool           `json:"hallucination_detector"`
	HallucinationExplainer  bool           `json:"hallucination_explainer"`
	FeedbackDetector        bool           `json:"feedback_detector"`
	DecisionCount           int            `json:"decision_count"`
	SignalGroupCount        int            `json:"signal_group_count"`
	SignalCounts            map[string]int `json:"signal_counts"`
	ClassificationConfigAPI bool           `json:"classification_config_api"`
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
	response.SignalGroupCount = len(cfg.SignalGroups)
	response.SignalCounts = map[string]int{
		"domains":        len(cfg.Categories),
		"keywords":       len(cfg.KeywordRules),
		"embeddings":     len(cfg.EmbeddingRules),
		"fact_check":     len(cfg.FactCheckRules),
		"user_feedback":  len(cfg.UserFeedbackRules),
		"preferences":    len(cfg.PreferenceRules),
		"language":       len(cfg.LanguageRules),
		"context":        len(cfg.ContextRules),
		"complexity":     len(cfg.ComplexityRules),
		"jailbreak":      len(cfg.JailbreakRules),
		"pii":            len(cfg.PIIRules),
		"signal_groups":  len(cfg.SignalGroups),
		"routing_models": len(cfg.ModelConfig),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}
