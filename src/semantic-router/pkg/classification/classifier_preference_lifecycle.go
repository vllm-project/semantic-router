package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// IsPreferenceClassifierEnabled checks if preference classification is enabled and properly configured.
func (c *Classifier) IsPreferenceClassifierEnabled() bool {
	if len(c.Config.PreferenceRules) == 0 {
		return false
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

	externalCfg := c.Config.FindExternalModelByRole(config.ModelRolePreference)
	preferenceCfg := c.Config.PreferenceModel.WithDefaults()
	classifier, err := NewPreferenceClassifier(externalCfg, c.Config.PreferenceRules, &preferenceCfg)
	if err != nil {
		return fmt.Errorf("failed to create preference classifier: %w", err)
	}

	c.preferenceClassifier = classifier
	logPreferenceClassifierInitialized(preferenceCfg, externalCfg, len(c.Config.PreferenceRules))
	return nil
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
