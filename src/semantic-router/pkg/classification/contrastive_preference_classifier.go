package classification

import (
	"errors"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var ErrPreferenceBelowThreshold = errors.New("preference below threshold")

// ContrastivePreferenceClassifier performs few-shot preference routing using embeddings.
// It preloads embeddings for each preference rule's examples/description and selects
// the route whose support set is most similar to the incoming query.
type ContrastivePreferenceClassifier struct {
	modelType string

	rules []config.PreferenceRule

	// ruleEmbeddings maps rule name to its support embeddings
	ruleEmbeddings map[string][][]float32
	ruleBanks      map[string]*prototypeBank
	// ruleThresholds stores per-preference similarity thresholds
	ruleThresholds map[string]float32
	prototypeCfg   config.PrototypeScoringConfig

	mu sync.RWMutex
}

type PreferenceRuleScore struct {
	Name           string
	Score          float32
	Best           float32
	Support        float32
	Threshold      float32
	PrototypeCount int
}

type PreferenceClassificationDetails struct {
	Scores       []PreferenceRuleScore
	BestRule     string
	BestScore    float32
	RunnerUpRule string
	RunnerUp     float32
	Margin       float32
}

// NewContrastivePreferenceClassifier builds a contrastive preference classifier.
// modelType follows GetEmbeddingWithModelType (e.g. "qwen3", "gemma", "mmbert").
func NewContrastivePreferenceClassifier(rules []config.PreferenceRule, modelType string) (*ContrastivePreferenceClassifier, error) {
	if len(rules) == 0 {
		return nil, fmt.Errorf("contrastive preference rules cannot be empty")
	}

	if modelType == "" {
		modelType = "mmbert"
	}

	ruleThresholds := make(map[string]float32, len(rules))
	for _, rule := range rules {
		ruleThresholds[rule.Name] = rule.Threshold
	}

	c := &ContrastivePreferenceClassifier{
		modelType:      modelType,
		rules:          rules,
		ruleEmbeddings: make(map[string][][]float32),
		ruleBanks:      make(map[string]*prototypeBank),
		ruleThresholds: ruleThresholds,
		prototypeCfg:   config.PrototypeScoringConfig{}.WithDefaults(),
	}

	if err := c.preloadRuleEmbeddings(); err != nil {
		return nil, err
	}

	return c, nil
}

func NewContrastivePreferenceClassifierWithConfig(
	rules []config.PreferenceRule,
	modelType string,
	prototypeCfg config.PrototypeScoringConfig,
) (*ContrastivePreferenceClassifier, error) {
	classifier, err := NewContrastivePreferenceClassifier(rules, modelType)
	if err != nil {
		return nil, err
	}
	classifier.prototypeCfg = prototypeCfg.WithDefaults()
	classifier.rebuildRuleBanks()
	return classifier, nil
}

// Classify picks the preference with the highest similarity to the query.
func (c *ContrastivePreferenceClassifier) Classify(text string) (*PreferenceResult, error) {
	details, err := c.ClassifyDetailed(text)
	if err != nil {
		return nil, err
	}
	if details.BestRule == "" {
		return nil, fmt.Errorf("no preference matched by contrastive classifier")
	}
	threshold := c.ruleThresholds[details.BestRule]
	if threshold > 0 && details.BestScore < threshold {
		return nil, fmt.Errorf(
			"%w: preference similarity %.3f below threshold %.3f",
			ErrPreferenceBelowThreshold,
			details.BestScore,
			threshold,
		)
	}
	if c.prototypeCfg.WithDefaults().MarginThreshold > 0 && details.Margin < c.prototypeCfg.WithDefaults().MarginThreshold {
		return nil, fmt.Errorf(
			"%w: preference margin %.3f below threshold %.3f",
			ErrPreferenceBelowThreshold,
			details.Margin,
			c.prototypeCfg.WithDefaults().MarginThreshold,
		)
	}
	return &PreferenceResult{
		Preference: details.BestRule,
		Confidence: details.BestScore,
		Margin:     details.Margin,
	}, nil
}
