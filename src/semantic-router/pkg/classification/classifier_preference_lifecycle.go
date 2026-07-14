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
	var classifier *PreferenceClassifier
	var err error
	if preferenceCfg.ContrastiveEnabled() {
		provider, plan, prepareErr := c.prepareContrastivePreferenceClassifier(&preferenceCfg)
		if prepareErr != nil {
			return prepareErr
		}
		classifier, err = newPreferenceClassifierWithBackend(
			externalCfg,
			c.Config.PreferenceRules,
			&preferenceCfg,
			plan.Backend,
			provider,
		)
	} else {
		classifier, err = NewPreferenceClassifier(externalCfg, c.Config.PreferenceRules, &preferenceCfg)
	}
	if err != nil {
		return fmt.Errorf("failed to create preference classifier: %w", err)
	}

	c.preferenceClassifier = classifier
	logPreferenceClassifierInitialized(preferenceCfg, externalCfg, len(c.Config.PreferenceRules))
	return nil
}

func (c *Classifier) prepareContrastivePreferenceClassifier(
	preferenceCfg *config.PreferenceModelConfig,
) (embedding.Provider, embedding.RuntimePlan, error) {
	plan, err := resolveTextEmbeddingRuntimePlan(c.Config, preferenceCfg.EmbeddingModel)
	if err != nil {
		return nil, embedding.RuntimePlan{}, fmt.Errorf("invalid text embedding runtime plan for preference rules: %w", err)
	}
	if initErr := c.ensureTextEmbeddingRuntime(plan); initErr != nil {
		return nil, embedding.RuntimePlan{}, fmt.Errorf("failed to initialize text embedding backend for preference rules: %w", initErr)
	}
	preferenceCfg.EmbeddingModel = plan.ModelType
	provider, err := c.preferenceEmbeddingProvider(plan)
	if err != nil {
		return nil, embedding.RuntimePlan{}, err
	}
	return provider, plan, nil
}

func (c *Classifier) preferenceEmbeddingProvider(plan embedding.RuntimePlan) (embedding.Provider, error) {
	if c == nil || c.Config == nil {
		return nil, nil
	}
	if plan.Backend != config.EmbeddingBackendOpenAICompatible {
		return nil, nil
	}
	provider, err := embedding.NewProvider(c.Config.EmbeddingModels, embedding.ProviderOptions{
		BackendOverride: plan.Backend,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create preference embedding provider: %w", err)
	}
	return provider, nil
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
