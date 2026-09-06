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
	decisions := len(configData.Routing.Decisions)
	signals := countCanonicalSignals(configData.Routing.Signals)
	for _, recipe := range configData.Recipes {
		decisions += len(recipe.Routing.Decisions)
		signals += countCanonicalSignals(recipe.Routing.Signals)
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
	if len(configData.Routing.ModelCards) > 0 {
		return len(configData.Routing.ModelCards)
	}
	return len(configData.Providers.Models)
}

func countCanonicalSignals(signals routerconfig.CanonicalSignals) int {
	return len(signals.Keywords) +
		len(signals.Embeddings) +
		len(signals.Domains) +
		len(signals.FactCheck) +
		len(signals.UserFeedbacks) +
		len(signals.Reasks) +
		len(signals.Preferences) +
		len(signals.Language) +
		len(signals.Context) +
		len(signals.Structure) +
		len(signals.Complexity) +
		len(signals.Modality) +
		len(signals.RoleBindings) +
		len(signals.Jailbreak) +
		len(signals.PII) +
		len(signals.KB) +
		len(signals.Conversation) +
		len(signals.EventRules) +
		len(signals.Metadata) +
		len(signals.Classifiers)
}
