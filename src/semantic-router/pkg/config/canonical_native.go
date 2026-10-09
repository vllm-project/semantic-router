package config

import (
	"fmt"
	"strings"
)

func validateCanonicalNativeResources(canonical *CanonicalConfig) error {
	if err := canonical.Routing.Budget.Validate(); err != nil {
		return err
	}
	for _, recipe := range canonical.Recipes {
		if err := recipe.Routing.Budget.Validate(); err != nil {
			return fmt.Errorf("recipes[%s]: %w", recipe.Name, err)
		}
	}
	if canonical.Evaluation != nil {
		seen := map[string]bool{}
		for _, artifact := range canonical.Evaluation.Calibrations {
			if artifact.Name == "" || artifact.Name != strings.TrimSpace(artifact.Name) || seen[artifact.Name] {
				return fmt.Errorf("evaluation.calibrations requires distinct non-empty names without surrounding whitespace")
			}
			seen[artifact.Name] = true
			if err := validateNativeArtifact(artifact.Source, artifact.SHA256); err != nil {
				return fmt.Errorf("evaluation.calibrations[%s]: %w", artifact.Name, err)
			}
		}
	}
	for _, model := range canonical.Providers.Models {
		if err := validateNativeProviderResource(model); err != nil {
			return fmt.Errorf("providers.models[%s]: %w", model.Name, err)
		}
	}
	return nil
}

func validateNativeProviderResource(model CanonicalProviderModel) error {
	if model.Deployment != "" {
		if model.Deployment != strings.TrimSpace(model.Deployment) || model.APIFormat != APIFormatSystemOne {
			return fmt.Errorf("deployment requires api_format: systemone and an exact deployment key")
		}
		if len(model.BackendRefs) > 0 || model.ProviderModelID != "" || len(model.ExternalModelIDs) > 0 {
			return fmt.Errorf("deployment cannot be combined with backend_refs or provider model identity overrides")
		}
	}
	if model.APIFormat == APIFormatSystemOne && model.Deployment == "" && len(model.BackendRefs) == 0 {
		return fmt.Errorf("systemone requires exactly one of deployment or backend_refs")
	}
	return nil
}

// validateNativeRoutingConfig checks effective local bindings and recipe/API
// compatibility after canonical provider and global resources are compiled.
func validateNativeRoutingConfig(cfg *RouterConfig) error {
	if cfg.GetModelAPIFormat(cfg.DefaultModel) == APIFormatSystemOne {
		return fmt.Errorf("providers.defaults.model %q must be a Chat provider, not a native System One alias", cfg.DefaultModel)
	}
	for alias, model := range cfg.ModelConfig {
		if model.APIFormat == APIFormatSystemOne {
			for key, deployment := range cfg.ModelDeployments {
				if deployment.PublicModelName() == alias && model.Deployment != key {
					return fmt.Errorf("providers.models[%s]: native alias conflicts with public model of deployment %q", alias, key)
				}
			}
		}
		if model.Deployment == "" {
			continue
		}
		deployment, ok := cfg.ModelDeployments[model.Deployment]
		if !ok || !deployment.IsModelRuntime() {
			return fmt.Errorf("providers.models[%s].deployment %q must reference a model_runtime deployment", alias, model.Deployment)
		}
	}
	nativeRecipes := map[RecipeName]bool{}
	chatRecipes := map[RecipeName]bool{}
	for _, entrypoint := range cfg.EffectiveEntrypoints(SystemOneAPI) {
		nativeRecipes[entrypoint.Recipe] = true
	}
	for _, entrypoint := range cfg.EffectiveEntrypoints(ChatAPI) {
		chatRecipes[entrypoint.Recipe] = true
	}
	for _, recipe := range cfg.Recipes {
		if err := validateNativeRecipe(cfg, recipe, nativeRecipes[recipe.Name], chatRecipes[recipe.Name]); err != nil {
			return fmt.Errorf("recipes[%s]: %w", recipe.Name, err)
		}
	}
	return nil
}

func validateNativeRecipe(cfg *RouterConfig, recipe RoutingRecipe, native, chat bool) error {
	hasNativeAlgorithm := false
	for _, decision := range recipe.Profile.Decisions {
		hasNativeAlgorithm = hasNativeAlgorithm || decision.Algorithm.IsNative()
	}
	if native || hasNativeAlgorithm {
		if err := validateNativeProfileSurfaces(recipe.Profile); err != nil {
			return err
		}
		if err := validateNativeSignalBackends(cfg.ConfigForRecipe(&recipe), recipe.Profile); err != nil {
			return err
		}
	}
	if native && chat {
		return fmt.Errorf("a native recipe cannot also be a Chat entrypoint; use isolated recipes")
	}
	if native && recipe.Profile.Budget == nil {
		return fmt.Errorf("systemone routing requires an explicit routing.budget")
	}
	if recipe.Profile.Budget != nil && !native {
		if !hasNativeAlgorithm {
			return fmt.Errorf("routing.budget is currently supported only for native System One recipes")
		}
	}
	for _, decision := range recipe.Profile.Decisions {
		algorithm := decision.Algorithm
		if native && !algorithm.IsNative() {
			return fmt.Errorf("decision %q: System One requires cascade or policy", decision.Name)
		}
		if !algorithm.IsNative() {
			if err := validateNativeChatBoundary(cfg, decision); err != nil {
				return err
			}
			continue
		}
		if chat || recipe.Profile.Budget == nil {
			return fmt.Errorf("decision %q: native algorithms require a native recipe with routing.budget", decision.Name)
		}
		if err := validateNativeDecisionSurfaces(decision); err != nil {
			return err
		}
		if err := validateNativeAlgorithmConfig(decision.Name, decision.ModelRefs, algorithm); err != nil {
			return err
		}
		if err := validateNativeStageBindings(cfg, algorithm); err != nil {
			return fmt.Errorf("decision %q: %w", decision.Name, err)
		}
		if algorithm.Quality.Type == "calibrated" {
			if _, ok := cfg.Calibration(algorithm.Quality.Calibration); !ok {
				return fmt.Errorf("decision %q: unknown evaluation calibration %q", decision.Name, algorithm.Quality.Calibration)
			}
		}
	}
	return nil
}

func validateNativeStageBindings(cfg *RouterConfig, algorithm *AlgorithmConfig) error {
	for _, stage := range algorithm.Stages {
		model, ok := cfg.ModelConfig[stage.Model]
		if !ok {
			return fmt.Errorf("stage %q: model %q must name a provider alias", stage.Name, stage.Model)
		}
		if stage.Kind == "native" {
			if model.APIFormat != APIFormatSystemOne {
				return fmt.Errorf("stage %q: native model %q requires api_format: systemone", stage.Name, stage.Model)
			}
		} else if model.APIFormat != APIFormatOpenAI {
			return fmt.Errorf("stage %q: LLM model %q requires api_format: openai", stage.Name, stage.Model)
		}
	}
	return nil
}

// Calibration resolves a named immutable artifact without doing file I/O.
func (c *RouterConfig) Calibration(name string) (CalibrationArtifact, bool) {
	if c != nil && c.Evaluation != nil {
		for _, artifact := range c.Evaluation.Calibrations {
			if artifact.Name == name {
				return artifact, true
			}
		}
	}
	return CalibrationArtifact{}, false
}

// NativeStageModels returns the active stage candidate aliases in declaration
// order. Disabled actions validate normally but never create runtime demand.
func (a *AlgorithmConfig) NativeStageModels() []string {
	if !a.IsNative() {
		return nil
	}
	var result []string
	seen := map[string]bool{}
	for _, stage := range a.Stages {
		if !stage.IsEnabled() {
			continue
		}
		if !seen[stage.Model] {
			result = append(result, stage.Model)
			seen[stage.Model] = true
		}
	}
	return result
}

// validateNativeChatBoundary prevents native-only aliases from being used in
// Chat selection, orchestration, or error fallback. Native algorithms declare
// their mixed native/judge roles separately in stage bindings.
func validateNativeChatBoundary(cfg *RouterConfig, decision Decision) error {
	aliases := make([]string, 0, len(decision.ModelRefs))
	for _, ref := range decision.ModelRefs {
		aliases = append(aliases, ref.Model)
	}
	for _, iteration := range decision.CandidateIterations {
		for _, ref := range iteration.Models {
			aliases = append(aliases, ref.Model)
		}
	}
	if decision.Action != nil {
		aliases = append(aliases, decision.Action.Destination)
	}
	if algorithm := decision.Algorithm; algorithm != nil {
		if algorithm.Prompt != nil {
			aliases = append(aliases, algorithm.Prompt.Model)
		}
		if algorithm.ReMoM != nil {
			aliases = append(aliases, algorithm.ReMoM.SynthesisModel)
		}
		if fusion := algorithm.Fusion; fusion != nil {
			aliases = append(aliases, fusion.Model, fusion.QuorumFallbackTarget)
			aliases = append(aliases, fusion.AnalysisModels...)
		}
		if workflows := algorithm.Workflows; workflows != nil {
			aliases = append(aliases, workflows.Final.Model)
			aliases = append(aliases, workflows.Planner.Model)
			for _, role := range workflows.Roles {
				aliases = append(aliases, role.Models...)
			}
		}
	}
	for _, alias := range aliases {
		if cfg.GetModelAPIFormat(alias) == APIFormatSystemOne {
			return fmt.Errorf("decision %q: Chat execution cannot use native System One provider %q", decision.Name, alias)
		}
	}
	return nil
}
