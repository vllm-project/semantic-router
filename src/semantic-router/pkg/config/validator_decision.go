package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func validateDecisionContracts(cfg *RouterConfig) error {
	if err := validateDecisionModelContracts(cfg); err != nil {
		return err
	}
	return validateDecisionPluginContracts(cfg)
}

func validateDecisionModelContracts(cfg *RouterConfig) error {
	for _, decision := range cfg.Decisions {
		if err := validateDecisionModelRefs(cfg, decision); err != nil {
			return err
		}
		if err := validateDecisionAlgorithmConfig(decision.Name, decision.Algorithm); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionModelRefs(cfg *RouterConfig, decision Decision) error {
	for i, modelRef := range decision.ModelRefs {
		if modelRef.Model == "" {
			return fmt.Errorf("decision '%s', modelRefs[%d]: model name cannot be empty", decision.Name, i)
		}
		if modelRef.UseReasoning == nil {
			return fmt.Errorf("decision '%s', model '%s': missing required field 'use_reasoning'", decision.Name, modelRef.Model)
		}
		if modelRef.LoRAName == "" {
			continue
		}
		if err := validateLoRAName(cfg, modelRef.Model, modelRef.LoRAName); err != nil {
			return fmt.Errorf("decision '%s', model '%s': %w", decision.Name, modelRef.Model, err)
		}
	}
	return nil
}

func validateDecisionPluginContracts(cfg *RouterConfig) error {
	for _, decision := range cfg.Decisions {
		if toolsCfg := decision.GetToolsConfig(); toolsCfg != nil {
			if err := toolsCfg.Validate(); err != nil {
				return fmt.Errorf("decision '%s': %w", decision.Name, err)
			}
		}

		if imageGenCfg := decision.GetImageGenConfig(); imageGenCfg != nil {
			if err := imageGenCfg.Validate(); err != nil {
				return fmt.Errorf("decision '%s': %w", decision.Name, err)
			}
		}

		if err := validateDecisionRAGAndMemoryPlugins(cfg, &decision); err != nil {
			return err
		}
	}
	return nil
}

// validateDecisionRAGAndMemoryPlugins validates RAG config and warns about
// cache + personalization conflicts for a single decision.
func validateDecisionRAGAndMemoryPlugins(cfg *RouterConfig, decision *Decision) error {
	ragCfg := decision.GetRAGConfig()
	if ragCfg != nil {
		if err := ragCfg.Validate(); err != nil {
			return fmt.Errorf("decision '%s': RAG plugin: %w", decision.Name, err)
		}
	}

	cacheCfg := decision.GetSemanticCacheConfig()
	memCfg := decision.GetMemoryConfig()
	cacheActive := cacheCfg != nil && cacheCfg.Enabled
	ragActive := ragCfg != nil && ragCfg.Enabled
	memActive := memCfg != nil && memCfg.Enabled
	if !memActive && cfg.Memory.Enabled {
		memActive = memCfg == nil
	}
	if cacheActive && (ragActive || memActive) {
		logging.Warnf("Decision '%s': semantic-cache is enabled alongside %s. "+
			"Cache reads will be automatically bypassed to preserve personalized responses. "+
			"Cache writes still occur for observability. Remove the cache plugin if this is intentional.",
			decision.Name, cachePersonalizationConflictDescription(ragActive, memActive))
	}
	return nil
}

func cachePersonalizationConflictDescription(ragActive, memActive bool) string {
	switch {
	case ragActive && memActive:
		return "RAG and memory plugins"
	case ragActive:
		return "RAG plugin"
	default:
		return "memory plugin"
	}
}

func hasLegacyLatencyRoutingConfig(cfg *RouterConfig) bool {
	for _, decision := range cfg.Decisions {
		for _, condition := range decision.Rules.Conditions {
			if condition.Type == "latency" {
				return true
			}
		}
	}
	return false
}

func validateDecisionAlgorithmConfig(decisionName string, algorithm *AlgorithmConfig) error {
	if algorithm == nil {
		return nil
	}

	normalizedType := strings.ToLower(strings.TrimSpace(algorithm.Type))
	displayType := strings.TrimSpace(algorithm.Type)
	if displayType == "" {
		displayType = "<empty>"
	}

	configuredBlocks := make([]string, 0, 10)
	addBlock := func(name string, configured bool) {
		if configured {
			configuredBlocks = append(configuredBlocks, name)
		}
	}

	addBlock("confidence", algorithm.Confidence != nil)
	addBlock("ratings", algorithm.Ratings != nil)
	addBlock("remom", algorithm.ReMoM != nil)
	addBlock("elo", algorithm.Elo != nil)
	addBlock("router_dc", algorithm.RouterDC != nil)
	addBlock("automix", algorithm.AutoMix != nil)
	addBlock("hybrid", algorithm.Hybrid != nil)
	addBlock("session_aware", algorithm.SessionAware != nil)
	addBlock("rl_driven", algorithm.RLDriven != nil)
	addBlock("gmtrouter", algorithm.GMTRouter != nil)
	addBlock("latency_aware", algorithm.LatencyAware != nil)

	if len(configuredBlocks) > 1 {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s cannot be combined with multiple algorithm config blocks: %s",
			decisionName,
			displayType,
			strings.Join(configuredBlocks, ", "),
		)
	}

	expectedBlockByType := map[string]string{
		"confidence":    "confidence",
		"ratings":       "ratings",
		"remom":         "remom",
		"elo":           "elo",
		"router_dc":     "router_dc",
		"automix":       "automix",
		"hybrid":        "hybrid",
		"session_aware": "session_aware",
		"rl_driven":     "rl_driven",
		"gmtrouter":     "gmtrouter",
		"latency_aware": "latency_aware",
	}

	expectedBlock, hasExpectedBlock := expectedBlockByType[normalizedType]
	if !hasExpectedBlock {
		if len(configuredBlocks) > 0 {
			return fmt.Errorf(
				"decision '%s': algorithm.type=%s cannot be used with algorithm.%s configuration",
				decisionName,
				displayType,
				configuredBlocks[0],
			)
		}
		return nil
	}

	if len(configuredBlocks) == 1 && configuredBlocks[0] != expectedBlock {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s requires algorithm.%s configuration; found algorithm.%s",
			decisionName,
			displayType,
			expectedBlock,
			configuredBlocks[0],
		)
	}

	if normalizedType == "session_aware" {
		if algorithm.SessionAware == nil {
			return fmt.Errorf("decision '%s': algorithm.type=session_aware requires algorithm.session_aware configuration", decisionName)
		}
		if err := validateSessionAwareAlgorithmConfig(algorithm.SessionAware); err != nil {
			return fmt.Errorf("decision '%s', algorithm.session_aware: %w", decisionName, err)
		}
	}

	if normalizedType == "latency_aware" {
		if algorithm.LatencyAware == nil {
			return fmt.Errorf("decision '%s': algorithm.type=latency_aware requires algorithm.latency_aware configuration", decisionName)
		}
		if err := validateLatencyAwareAlgorithmConfig(algorithm.LatencyAware); err != nil {
			return fmt.Errorf("decision '%s', algorithm.latency_aware: %w", decisionName, err)
		}
	}

	return nil
}

// validateLatencyAwareAlgorithmConfig validates latency_aware algorithm configuration.
func validateLatencyAwareAlgorithmConfig(cfg *LatencyAwareAlgorithmConfig) error {
	hasTPOTPercentile := cfg.TPOTPercentile > 0
	hasTTFTPercentile := cfg.TTFTPercentile > 0

	if !hasTPOTPercentile && !hasTTFTPercentile {
		return fmt.Errorf("must specify at least one of tpot_percentile (1-100) or ttft_percentile (1-100). RECOMMENDED: use both for comprehensive latency evaluation")
	}

	warnIncompleteLatencyAwarePercentiles(hasTPOTPercentile, hasTTFTPercentile)

	for _, field := range []struct {
		name    string
		value   int
		enabled bool
	}{
		{name: "tpot_percentile", value: cfg.TPOTPercentile, enabled: hasTPOTPercentile},
		{name: "ttft_percentile", value: cfg.TTFTPercentile, enabled: hasTTFTPercentile},
	} {
		if err := validateLatencyAwarePercentile(field.name, field.value, field.enabled); err != nil {
			return err
		}
	}

	return nil
}

func warnIncompleteLatencyAwarePercentiles(hasTPOTPercentile bool, hasTTFTPercentile bool) {
	if hasTPOTPercentile && !hasTTFTPercentile {
		logging.Warnf("algorithm.latency_aware: only tpot_percentile is set. RECOMMENDED: also set ttft_percentile for comprehensive latency evaluation (user-perceived latency)")
	}
	if !hasTPOTPercentile && hasTTFTPercentile {
		logging.Warnf("algorithm.latency_aware: only ttft_percentile is set. RECOMMENDED: also set tpot_percentile for comprehensive latency evaluation (token generation throughput)")
	}
}

func validateLatencyAwarePercentile(name string, value int, enabled bool) error {
	if !enabled {
		return nil
	}
	if value < 1 || value > 100 {
		return fmt.Errorf("%s must be between 1 and 100, got: %d", name, value)
	}
	return nil
}

// validateLoRAName checks if the specified LoRA name is defined in the
// canonical routing model catalog for the selected model.
func validateLoRAName(cfg *RouterConfig, modelName string, loraName string) error {
	modelParams, exists := cfg.ModelConfig[modelName]
	if !exists {
		return fmt.Errorf(
			"lora_name %q specified but model %q is not declared in routing.modelCards",
			loraName,
			modelName,
		)
	}

	if len(modelParams.LoRAs) == 0 {
		return fmt.Errorf(
			"lora_name %q specified but model %q declares no routing.modelCards[].loras entries",
			loraName,
			modelName,
		)
	}

	for _, lora := range modelParams.LoRAs {
		if lora.Name == loraName {
			return nil
		}
	}

	availableLoRAs := make([]string, len(modelParams.LoRAs))
	for i, lora := range modelParams.LoRAs {
		availableLoRAs[i] = lora.Name
	}
	return fmt.Errorf(
		"lora_name %q is not declared in routing.modelCards[%q].loras. Available LoRAs: %v",
		loraName,
		modelName,
		availableLoRAs,
	)
}
