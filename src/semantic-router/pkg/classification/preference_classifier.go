package classification

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// PreferenceResult represents the result of preference classification
type PreferenceResult struct {
	Preference string  `json:"route"` // The matched route name
	Confidence float32 `json:"confidence,omitempty"`
	Margin     float32 `json:"margin,omitempty"`
}

// PreferenceClassifier handles route preference matching via external LLM
type PreferenceClassifier struct {
	client             *VLLMClient
	modelName          string
	timeout            time.Duration
	preferenceRules    []config.PreferenceRule
	systemPrompt       string
	userPromptTemplate string
	contrastive        *ContrastivePreferenceClassifier

	useContrastive bool
}

// NewPreferenceClassifier creates a new preference classifier
func NewPreferenceClassifier(externalCfg *config.ExternalModelConfig, rules []config.PreferenceRule, localCfg *config.PreferenceModelConfig) (*PreferenceClassifier, error) {
	return NewPreferenceClassifierWithProvider(externalCfg, rules, localCfg, nil)
}

func NewPreferenceClassifierWithProvider(
	externalCfg *config.ExternalModelConfig,
	rules []config.PreferenceRule,
	localCfg *config.PreferenceModelConfig,
	provider embedding.Provider,
) (*PreferenceClassifier, error) {
	return newPreferenceClassifierWithBackend(externalCfg, rules, localCfg, "", provider)
}

func newPreferenceClassifierWithBackend(
	externalCfg *config.ExternalModelConfig,
	rules []config.PreferenceRule,
	localCfg *config.PreferenceModelConfig,
	backend string,
	provider embedding.Provider,
) (*PreferenceClassifier, error) {
	resolvedLocalCfg := config.PreferenceModelConfig{}
	if localCfg != nil {
		resolvedLocalCfg = *localCfg
	}
	resolvedLocalCfg = resolvedLocalCfg.WithDefaults()

	// Contrastive few-shot preference routing
	if resolvedLocalCfg.ContrastiveEnabled() {
		return newContrastivePreferenceClassifier(rules, resolvedLocalCfg, backend, provider)
	}

	return newExternalPreferenceClassifier(externalCfg, rules)
}

// Classify determines the best route preference for the given conversation
func (p *PreferenceClassifier) Classify(conversationJSON string) (*PreferenceResult, error) {
	if p.useContrastive {
		return p.classifyContrastive(conversationJSON)
	}

	return p.classifyExternal(conversationJSON)
}

// IsInitialized returns true if the classifier is initialized
func (p *PreferenceClassifier) IsInitialized() bool {
	if p == nil {
		return false
	}

	if p.useContrastive {
		return p.contrastive != nil
	}

	return p.client != nil
}
