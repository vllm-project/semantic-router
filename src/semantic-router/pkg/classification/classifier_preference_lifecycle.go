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
	if c.Config.PreferenceUsesDecisionTask() && c.models != nil && moduleConfigured(c.Config, "preference") {
		return true
	}

	if c.Config.PreferenceUsesPrototypes() {
		return true
	}

	deployment, external := c.Config.PreferenceExternalDeployment()
	return external && deployment.Endpoint != "" && deployment.Model != ""
}

// initializePreferenceClassifier prepares the selected preference execution contract.
func (c *Classifier) initializePreferenceClassifier() error {
	if !c.IsPreferenceClassifierEnabled() {
		return nil
	}
	if c.Config.PreferenceUsesDecisionTask() {
		judgment, err := prepareDecisionPreference(c.models, c.Config.PreferenceRules)
		if err != nil {
			return err
		}
		if judgment != nil {
			c.preferenceClassifier = &PreferenceClassifier{judgment: judgment, preferenceRules: c.Config.PreferenceRules}
			return nil
		}
	}

	var external *config.ModelDeployment
	if deployment, ok := c.Config.PreferenceExternalDeployment(); ok {
		external = &deployment
	}
	preferenceCfg := c.Config.PreferenceModel.WithDefaults()
	usePrototypes := c.Config.PreferenceUsesPrototypes()
	preferenceCfg.UseContrastive = &usePrototypes
	provider, err := c.preferenceEmbeddingProvider()
	if err != nil {
		return err
	}
	classifier, err := NewPreferenceClassifierWithProvider(external, c.Config.PreferenceRules, &preferenceCfg, provider)
	if err != nil {
		return fmt.Errorf("failed to create preference classifier: %w", err)
	}

	c.preferenceClassifier = classifier
	logPreferenceClassifierInitialized(preferenceCfg, external, len(c.Config.PreferenceRules))
	return nil
}

func (c *Classifier) preferenceEmbeddingProvider() (embedding.Provider, error) {
	if c == nil || c.Config == nil || !c.Config.PreferenceUsesPrototypes() {
		return nil, nil
	}
	model := c.Config.PreferenceModel.EmbeddingModel
	if model == "" {
		model = "mmbert"
	}
	if c.Config.UsesRemoteEmbeddings() {
		model = ""
	}
	return c.EmbeddingForModel(model, 0, 0)
}

func logPreferenceClassifierInitialized(
	preferenceCfg config.PreferenceModelConfig,
	external *config.ModelDeployment,
	routeCount int,
) {
	mode := "external_llm"
	modelRef := ""
	if preferenceCfg.ContrastiveEnabled() {
		mode = "contrastive"
		modelRef = preferenceCfg.EmbeddingModel
	} else if external != nil {
		modelRef = external.Model
	}
	logging.ComponentEvent("classifier", "preference_classifier_initialized", map[string]interface{}{
		"mode":      mode,
		"model_ref": modelRef,
		"routes":    routeCount,
	})
}
