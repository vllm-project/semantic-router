package classification

import (
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// EmbeddingClassifier performs embedding-based similarity classification.
// When preloading is enabled, candidate embeddings are computed once at
// initialization and reused for all classification requests.
type EmbeddingClassifier struct {
	rules []config.EmbeddingRule

	// rulesByModality is a precomputed lookup keyed by EffectiveQueryModality.
	// Rules do not mutate at runtime so this is populated once in
	// NewEmbeddingClassifier and shared by every classify call.
	rulesByModality map[config.QueryModality][]config.EmbeddingRule

	candidateEmbeddings map[string][]float32
	rulePrototypeBanks  map[string]*prototypeBank

	optimizationConfig config.HNSWConfig
	preloadEnabled     bool
	modelType          string
	backend            string
}

// NewEmbeddingClassifier creates a new EmbeddingClassifier.
// If optimization config has PreloadEmbeddings enabled, candidate embeddings
// will be precomputed at initialization time for better runtime performance.
func NewEmbeddingClassifier(cfgRules []config.EmbeddingRule, optConfig config.HNSWConfig) (*EmbeddingClassifier, error) {
	optConfig = optConfig.WithDefaults()

	c := &EmbeddingClassifier{
		rules:               cfgRules,
		rulesByModality:     buildRulesByModality(cfgRules),
		candidateEmbeddings: make(map[string][]float32),
		rulePrototypeBanks:  make(map[string]*prototypeBank),
		optimizationConfig:  optConfig,
		preloadEnabled:      optConfig.PreloadEmbeddings,
		modelType:           optConfig.ModelType,
		backend:             strings.ToLower(strings.TrimSpace(optConfig.Backend)),
	}

	logging.ComponentEvent("classifier", "embedding_classifier_initialized", map[string]interface{}{
		"model_type":          c.modelType,
		"backend":             c.backend,
		"rules":               len(cfgRules),
		"preload_embeddings":  optConfig.PreloadEmbeddings,
		"target_dimension":    optConfig.TargetDimension,
		"prototype_scoring":   optConfig.PrototypeScoring.IsEnabled(),
		"multimodal_prepared": optConfig.ModelType == "multimodal",
	})

	if optConfig.PreloadEmbeddings {
		if err := c.preloadCandidateEmbeddings(); err != nil {
			logging.ComponentWarnEvent("classifier", "embedding_candidates_preload_failed", map[string]interface{}{
				"model_type":       c.modelType,
				"target_dimension": c.optimizationConfig.TargetDimension,
				"error":            err.Error(),
				"fallback":         "runtime_computation",
			})
			c.preloadEnabled = false
		}
	}

	return c, nil
}

// getModelType returns the model type to use for embeddings.
func (c *EmbeddingClassifier) getModelType() string {
	if model := os.Getenv("EMBEDDING_MODEL_OVERRIDE"); model != "" {
		logging.ComponentDebugEvent("classifier", "embedding_model_override_enabled", map[string]interface{}{
			"model_type": model,
		})
		return model
	}
	return c.modelType
}

// buildRulesByModality groups rules by their effective query modality so the
// classifier can dispatch each request to the correct subset without per-call
// allocation. Rules with an unset modality are bucketed under QueryModalityText.
func buildRulesByModality(rules []config.EmbeddingRule) map[config.QueryModality][]config.EmbeddingRule {
	byModality := make(map[config.QueryModality][]config.EmbeddingRule, 3)
	for _, rule := range rules {
		modality := rule.EffectiveQueryModality()
		byModality[modality] = append(byModality[modality], rule)
	}
	return byModality
}

// MatchedRule holds the result for a matched embedding rule.
type MatchedRule struct {
	RuleName string
	Score    float64
	Method   string
}

type EmbeddingRuleScore struct {
	Name           string
	Score          float64
	Best           float64
	Support        float64
	Threshold      float64
	PrototypeCount int
}

type EmbeddingClassificationResult struct {
	Scores  []EmbeddingRuleScore
	Matches []MatchedRule
}
