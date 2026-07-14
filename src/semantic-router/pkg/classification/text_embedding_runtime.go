package classification

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// textEmbeddingRuntime owns the process-global native text model plan for one
// classifier generation. OpenVINO exposes one global embedding model, so all
// families must share a canonical plan and initialization must happen once.
type textEmbeddingRuntime struct {
	mu               sync.Mutex
	plan             *embedding.RuntimePlan
	initializedPlans map[embedding.RuntimePlan]struct{}
}

func (r *textEmbeddingRuntime) ensureInitialized(cfg *config.RouterConfig, plan embedding.RuntimePlan) error {
	if r == nil {
		return fmt.Errorf("text embedding runtime is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	plans, err := configuredTextEmbeddingRuntimePlans(cfg, plan)
	if err != nil {
		return err
	}
	if r.plan != nil && r.plan.Backend == config.EmbeddingBackendOpenVINO && plan.Backend == config.EmbeddingBackendOpenVINO && r.plan.ModelType != plan.ModelType {
		return fmt.Errorf("OpenVINO text embedding model conflict: runtime uses %q but family requested %q", r.plan.ModelType, plan.ModelType)
	}
	if r.initializedPlans != nil {
		if _, ok := r.initializedPlans[plan]; ok {
			return nil
		}
		return fmt.Errorf(
			"text embedding runtime conflict: process-global runtime does not contain backend %q model %q path %q",
			plan.Backend,
			plan.ModelType,
			plan.ModelPath,
		)
	}
	if err := initializeResolvedTextEmbeddingBackend(cfg, plans); err != nil {
		return err
	}
	copyPlan := plan
	r.plan = &copyPlan
	r.initializedPlans = make(map[embedding.RuntimePlan]struct{}, len(plans))
	for _, initializedPlan := range plans {
		r.initializedPlans[initializedPlan] = struct{}{}
	}
	return nil
}

func (c *Classifier) ensureTextEmbeddingRuntime(plan embedding.RuntimePlan) error {
	if c.textBackendRuntime == nil {
		return initializeConfiguredTextEmbeddingBackend(c.Config, plan)
	}
	return c.textBackendRuntime.ensureInitialized(c.Config, plan)
}

func validateOpenVINOTextPlanConsistency(cfg *config.RouterConfig) error {
	if cfg == nil || !hasNonPreferenceTextEmbeddingFamily(cfg) || len(cfg.PreferenceRules) == 0 || !cfg.PreferenceModel.ContrastiveEnabled() {
		return nil
	}
	basePlan, err := resolveTextEmbeddingRuntimePlan(cfg, cfg.EmbeddingConfig.ModelType)
	if err != nil {
		return err
	}
	preferencePlan, err := resolveTextEmbeddingRuntimePlan(cfg, cfg.PreferenceModel.WithDefaults().EmbeddingModel)
	if err != nil {
		return err
	}
	if basePlan.Backend == config.EmbeddingBackendOpenVINO && preferencePlan.Backend == config.EmbeddingBackendOpenVINO && basePlan.ModelType != preferencePlan.ModelType {
		return fmt.Errorf("OpenVINO text embedding model conflict: routing families use %q but preference uses %q", basePlan.ModelType, preferencePlan.ModelType)
	}
	return nil
}

func hasNonPreferenceTextEmbeddingFamily(cfg *config.RouterConfig) bool {
	if len(cfg.EmbeddingRules) > 0 || len(cfg.ReaskRules) > 0 || len(cfg.ComplexityRules) > 0 || len(cfg.KnowledgeBases) > 0 {
		return true
	}
	for _, rule := range cfg.JailbreakRules {
		if rule.Method == "contrastive" {
			return true
		}
	}
	return false
}
