package config

import (
	"fmt"
	"strings"
)

func validateFusionContracts(cfg *RouterConfig) error {
	if err := ValidateFusionRuntimeConfig(cfg.Looper.Fusion); err != nil {
		return fmt.Errorf("global.integrations.looper.fusion: %w", err)
	}
	return nil
}

// validateDecisionFusionFallbackTarget validates the recipe-owned
// quorum_fallback_target. The checks are split by what data they need so a
// routing-only fragment, which carries no provider state, still round-trips
// through the DSL while a complete config gets the full contract.
func validateDecisionFusionFallbackTarget(cfg *RouterConfig, decision Decision) error {
	if decision.Algorithm == nil || decision.Algorithm.Fusion == nil {
		return nil
	}
	fusion := decision.Algorithm.Fusion
	if fusion.QuorumFailurePolicy != FusionQuorumFailurePolicyFallback {
		return nil
	}
	target := strings.TrimSpace(fusion.QuorumFallbackTarget)

	if err := validateFusionFallbackTargetIdentity(cfg, decision, fusion, target); err != nil {
		return err
	}
	return validateFusionFallbackTargetModel(cfg, decision, fusion, target)
}

// validateFusionFallbackTargetIdentity covers the rules that need only routing
// data, so they apply to routing-only fragments as well as complete configs.
func validateFusionFallbackTargetIdentity(
	cfg *RouterConfig,
	decision Decision,
	fusion *FusionAlgorithmConfig,
	target string,
) error {
	if isCompositeFusionFallbackTarget(cfg, target) {
		return fusionFallbackTargetError(decision.Name, target,
			"must be a concrete provider model, not a composite or virtual model slug")
	}
	// The effective panel is analysis_models when set, otherwise the decision's
	// modelRefs. Retrying a model that already failed as a panel member is not
	// recovery. A Fusion recipe owns the panel, and context eligibility can only
	// narrow modelRefs at request time, so the panel checked here is the widest
	// one execution can use.
	//
	// Spelling equality is checked here because it needs no catalog, so it also
	// holds for routing-only fragments. Alias equivalence needs the catalog and is
	// enforced by fusionFallbackPanelRejection.
	for _, panelModel := range effectiveFusionPanelModels(decision, fusion) {
		if panelModel == target {
			return fusionFallbackTargetError(decision.Name, target,
				"cannot be one of the analysis models it falls back from")
		}
	}
	return nil
}

// isCompositeFusionFallbackTarget reports whether the target re-enters a
// composite or virtual routing path instead of naming a concrete model.
func isCompositeFusionFallbackTarget(cfg *RouterConfig, target string) bool {
	return cfg.IsAutoModelName(target) ||
		cfg.IsEntrypointModelName(target) ||
		cfg.IsFusionModelName(target) ||
		cfg.IsFlowModelName(target) ||
		cfg.IsReMoMModelName(target)
}

// validateFusionFallbackTargetModel covers the rules that need a declared model
// catalog. A routing-only fragment declares no model cards, so the catalog
// checks are skipped there rather than making the fragment undecompilable.
func validateFusionFallbackTargetModel(
	cfg *RouterConfig,
	decision Decision,
	fusion *FusionAlgorithmConfig,
	target string,
) error {
	// Only a document parsed as a routing fragment skips the catalog rules. An
	// empty catalog in a complete config is not a reason to skip them: it means
	// the target is undeclared, which must fail.
	if cfg.RoutingFragmentOnly {
		return nil
	}
	modelConfig, ok := cfg.ModelConfig[target]
	if !ok {
		return fusionFallbackTargetError(decision.Name, target,
			"is not declared in routing.modelCards")
	}
	if strings.EqualFold(modelConfig.APIFormat, ClientProtocolAnthropic) {
		return fusionFallbackTargetError(decision.Name, target,
			"must use an OpenAI-compatible API format")
	}
	if !promptHelperModalitySupported(modelConfig.Modality) {
		return fusionFallbackTargetError(decision.Name, target,
			"must be chat-capable text modality")
	}
	// Validate against the widest panel the recipe can execute; request-time
	// context eligibility can only narrow it.
	if reason := fusionFallbackPanelRejection(cfg, decision, fusion, target); reason != "" {
		return fusionFallbackTargetError(decision.Name, target, reason)
	}
	if !promptHelperHasBackend(cfg, target) {
		return fusionFallbackTargetError(decision.Name, target,
			"requires a provider backend")
	}
	return nil
}

// fusionFallbackPanelRejection reports why the target cannot serve the effective
// panel, or "" when it can.
//
// Every non-success outcome of the resolver is a rejection, so a recipe
// selecting the fallback policy must supply metadata that proves its target can
// replace the panel. Recipes on the default fail policy, and routing-only
// fragments, never reach here.
func fusionFallbackPanelRejection(
	cfg *RouterConfig,
	decision Decision,
	fusion *FusionAlgorithmConfig,
	target string,
) string {
	resolution := resolveFusionFallbackCapabilities(
		cfg.ModelConfig, effectiveFusionPanelModels(decision, fusion), target)
	switch {
	case len(resolution.unresolved) > 0:
		return "cannot be validated because no model metadata resolves: " +
			strings.Join(resolution.unresolved, ", ")
	case len(resolution.ambiguous) > 0:
		return "cannot be validated because model metadata is ambiguous: " +
			strings.Join(resolution.ambiguous, "; ")
	case len(resolution.conflictingPanelModels) > 0:
		return "dispatches the same model as analysis model(s) it falls back from: " +
			strings.Join(resolution.conflictingPanelModels, ", ")
	case resolution.gap.targetUndeclared:
		return "does not declare capabilities required by the panel it replaces: " +
			"no declared capabilities while the panel declares some"
	case len(resolution.gap.missing) > 0:
		return "does not declare capabilities required by the panel it replaces: " +
			strings.Join(resolution.gap.missing, ", ")
	default:
		return ""
	}
}

// effectiveFusionPanelModels mirrors the looper's panel resolution: explicit
// analysis_models win; otherwise modelRefs use lora_name when present.
func effectiveFusionPanelModels(decision Decision, fusion *FusionAlgorithmConfig) []string {
	if len(fusion.AnalysisModels) > 0 {
		models := make([]string, 0, len(fusion.AnalysisModels))
		for _, model := range fusion.AnalysisModels {
			models = append(models, strings.TrimSpace(model))
		}
		return models
	}
	models := make([]string, 0, len(decision.ModelRefs))
	for _, ref := range decision.ModelRefs {
		if lora := strings.TrimSpace(ref.LoRAName); lora != "" {
			models = append(models, lora)
			continue
		}
		models = append(models, strings.TrimSpace(ref.Model))
	}
	return models
}

func fusionFallbackTargetError(decisionName string, target string, reason string) error {
	return fmt.Errorf(
		"decision '%s', algorithm.fusion.quorum_fallback_target %q %s",
		decisionName, target, reason,
	)
}
