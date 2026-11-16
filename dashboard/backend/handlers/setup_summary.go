package handlers

import routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func summarizeSetupConfig(configData *routerconfig.CanonicalConfig) setupConfigSummary {
	cfg, err := parseSetupRouterConfig(configData)
	if err != nil {
		return summarizeCanonicalSetupConfig(configData)
	}

	canonical := routerconfig.CanonicalConfigFromRouterConfig(cfg)
	return summarizeCanonicalSetupConfig(&canonical)
}

func parseSetupRouterConfig(configData *routerconfig.CanonicalConfig) (*routerconfig.RouterConfig, error) {
	yamlData, err := marshalYAMLBytes(configData)
	if err != nil {
		return nil, err
	}
	return routerconfig.ParseYAMLBytes(yamlData)
}

func summarizeCanonicalSetupConfig(configData *routerconfig.CanonicalConfig) setupConfigSummary {
	if configData == nil {
		return setupConfigSummary{}
	}
	decisions, signals := 0, 0
	for _, recipe := range configData.Recipes {
		decisions += len(recipe.Decisions)
		signals += countCanonicalSignals(recipe.Signals)
	}
	return setupConfigSummary{
		Models:    countConfiguredModels(configData),
		Decisions: decisions,
		Signals:   signals,
	}
}

func countConfiguredModels(configData *routerconfig.CanonicalConfig) int {
	if configData == nil {
		return 0
	}
	return len(configData.Providers.Models)
}

func countCanonicalSignals(signals routerconfig.CanonicalSignals) int {
	return len(signals.Keywords) +
		len(signals.Domains) +
		len(signals.FactCheck) +
		len(signals.UserFeedbacks) +
		len(signals.Language) +
		len(signals.Context) +
		len(signals.Complexity) +
		len(signals.Modality) +
		len(signals.RoleBindings) +
		len(signals.Jailbreak) +
		len(signals.Safety) +
		len(signals.Hallucination) +
		len(signals.Refusal) +
		len(signals.TurnRelation) +
		len(signals.PII) +
		len(signals.Conversation) +
		len(signals.Metadata) +
		len(signals.SystemOne)
}
