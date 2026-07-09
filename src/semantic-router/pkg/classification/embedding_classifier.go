package classification

import (
	"os"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
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
	preloadRequested   bool
	preloadComplete    bool
	preloadMu          sync.Mutex
	modelType          string
	backend            string
	provider           embedding.Provider
}

// NewEmbeddingClassifier creates a new EmbeddingClassifier.
// Construction is side-effect free: model-backed candidate embedding warmup is
// owned by InitializeRuntime or by the first classification request.
func NewEmbeddingClassifier(cfgRules []config.EmbeddingRule, optConfig config.HNSWConfig) (*EmbeddingClassifier, error) {
	return NewEmbeddingClassifierWithProvider(cfgRules, optConfig, nil)
}

func NewEmbeddingClassifierWithProvider(cfgRules []config.EmbeddingRule, optConfig config.HNSWConfig, provider embedding.Provider) (*EmbeddingClassifier, error) {
	optConfig = optConfig.WithDefaults()

	c := &EmbeddingClassifier{
		rules:               cfgRules,
		rulesByModality:     buildRulesByModality(cfgRules),
		candidateEmbeddings: make(map[string][]float32),
		rulePrototypeBanks:  make(map[string]*prototypeBank),
		optimizationConfig:  optConfig,
		preloadRequested:    optConfig.PreloadEmbeddings,
		modelType:           optConfig.ModelType,
		backend:             strings.ToLower(strings.TrimSpace(optConfig.Backend)),
		provider:            provider,
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
