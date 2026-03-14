package config

import "sort"

// CanonicalRoutingFromRouterConfig exports the routing-owned canonical surface
// from the internal runtime config. Deployment bindings and router-global
// runtime settings intentionally stay outside this view.
func CanonicalRoutingFromRouterConfig(cfg *RouterConfig) CanonicalRouting {
	if cfg == nil {
		return CanonicalRouting{}
	}

	return CanonicalRouting{
		ModelCards: routingModelsFromRouterConfig(cfg),
		Signals:    canonicalSignalsFromRouterConfig(cfg),
		Decisions:  copyDecisions(cfg.Decisions),
	}
}

func canonicalSignalsFromRouterConfig(cfg *RouterConfig) CanonicalSignals {
	return CanonicalSignals{
		Keywords:      append([]KeywordRule(nil), cfg.Signals.KeywordRules...),
		Embeddings:    append([]EmbeddingRule(nil), cfg.Signals.EmbeddingRules...),
		Domains:       append([]Category(nil), cfg.Signals.Categories...),
		FactCheck:     append([]FactCheckRule(nil), cfg.Signals.FactCheckRules...),
		UserFeedbacks: append([]UserFeedbackRule(nil), cfg.Signals.UserFeedbackRules...),
		Preferences:   append([]PreferenceRule(nil), cfg.Signals.PreferenceRules...),
		Language:      append([]LanguageRule(nil), cfg.Signals.LanguageRules...),
		Context:       append([]ContextRule(nil), cfg.Signals.ContextRules...),
		Complexity:    append([]ComplexityRule(nil), cfg.Signals.ComplexityRules...),
		Modality:      append([]ModalityRule(nil), cfg.Signals.ModalityRules...),
		RoleBindings:  append([]RoleBinding(nil), cfg.Signals.RoleBindings...),
		Jailbreak:     append([]JailbreakRule(nil), cfg.Signals.JailbreakRules...),
		PII:           append([]PIIRule(nil), cfg.Signals.PIIRules...),
	}
}

func routingModelsFromRouterConfig(cfg *RouterConfig) []RoutingModel {
	modelNames := make(map[string]bool)
	for name := range cfg.ModelConfig {
		modelNames[name] = true
	}
	for _, decision := range cfg.Decisions {
		for _, ref := range decision.ModelRefs {
			if ref.Model != "" {
				modelNames[ref.Model] = true
			}
		}
	}

	if len(modelNames) == 0 {
		return nil
	}

	names := make([]string, 0, len(modelNames))
	for name := range modelNames {
		names = append(names, name)
	}
	sort.Strings(names)

	models := make([]RoutingModel, 0, len(names))
	for _, name := range names {
		params := cfg.ModelConfig[name]
		models = append(models, RoutingModel{
			Name:               name,
			ReasoningFamilyRef: params.ReasoningFamily,
			ParamSize:          params.ParamSize,
			ContextWindowSize:  params.ContextWindowSize,
			Description:        params.Description,
			Capabilities:       append([]string(nil), params.Capabilities...),
			Tags:               append([]string(nil), params.Tags...),
			QualityScore:       params.QualityScore,
			Modality:           params.Modality,
		})
	}
	return models
}
