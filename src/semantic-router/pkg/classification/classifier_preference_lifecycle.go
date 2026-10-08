package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsPreferenceClassifierEnabled checks if preference classification is enabled and properly configured.
func (c *Classifier) IsPreferenceClassifierEnabled() bool {
	if len(c.Config.PreferenceRules) == 0 {
		return false
	}
	if c.models != nil && c.Config.DecisionModel != "" {
		return true
	}

	if c.Config.PreferenceModel.ContrastiveEnabled() {
		return true
	}

	externalCfg := c.Config.FindExternalModelByRole(config.ModelRolePreference)
	return externalCfg != nil &&
		externalCfg.ModelEndpoint.Address != "" &&
		externalCfg.ModelName != ""
}

// initializePreferenceClassifier initializes the preference classifier with external LLM.
func (c *Classifier) initializePreferenceClassifier() error {
	if !c.IsPreferenceClassifierEnabled() {
		return nil
	}
	// An explicitly configured contrastive or external classifier keeps its own
	// execution contract. Omitted preference-model policy uses shared judgment.
	if !c.Config.PreferenceModel.ContrastiveEnabled() && c.Config.FindExternalModelByRole(config.ModelRolePreference) == nil {
		judgment, err := prepareDecisionPreference(c.models, c.Config.PreferenceRules)
		if err != nil {
			return err
		}
		if judgment != nil {
			c.preferenceClassifier = &PreferenceClassifier{judgment: judgment, preferenceRules: c.Config.PreferenceRules}
			return nil
		}
	}

	externalCfg := c.Config.FindExternalModelByRole(config.ModelRolePreference)
	preferenceCfg := c.Config.PreferenceModel.WithDefaults()
	provider, err := c.preferenceEmbeddingProvider()
	if err != nil {
		return err
	}
	classifier, err := NewPreferenceClassifierWithProvider(externalCfg, c.Config.PreferenceRules, &preferenceCfg, provider)
	if err != nil {
		return fmt.Errorf("failed to create preference classifier: %w", err)
	}

	c.preferenceClassifier = classifier
	logPreferenceClassifierInitialized(preferenceCfg, externalCfg, len(c.Config.PreferenceRules))
	return nil
}

func (c *Classifier) preferenceEmbeddingProvider() (embedding.Provider, error) {
	if c == nil || c.Config == nil || !c.Config.PreferenceModel.ContrastiveEnabled() {
		return nil, nil
	}
	model := c.Config.PreferenceModel.EmbeddingModel
	if model == "" {
		model = "mmbert"
	}
	if c.Config.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		model = ""
	}
	return c.EmbeddingForModel(model, 0, 0)
}

func logPreferenceClassifierInitialized(
	preferenceCfg config.PreferenceModelConfig,
	externalCfg *config.ExternalModelConfig,
	routeCount int,
) {
	mode := "external_llm"
	modelRef := ""
	if preferenceCfg.ContrastiveEnabled() {
		mode = "contrastive"
		modelRef = preferenceCfg.EmbeddingModel
	} else if externalCfg != nil {
		modelRef = externalCfg.ModelName
	}
	logging.ComponentEvent("classifier", "preference_classifier_initialized", map[string]interface{}{
		"mode":      mode,
		"model_ref": modelRef,
		"routes":    routeCount,
	})
}
