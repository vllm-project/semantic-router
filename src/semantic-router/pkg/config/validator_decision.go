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
	if err := validateDecisionEmitContracts(cfg); err != nil {
		return err
	}
	return validateDecisionPluginContracts(cfg)
}

func validateDecisionModelContracts(cfg *RouterConfig) error {
	for _, decision := range cfg.Decisions {
		if err := validateDecisionModelRefs(cfg, decision); err != nil {
			return err
		}
		if err := validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm); err != nil {
			return err
		}
		if err := validateDecisionWorkflowModelRefs(decision); err != nil {
			return err
		}
		if err := validateDecisionCandidateIterations(decision); err != nil {
			return err
		}
		if err := validateDecisionOutputContractSpec(decision); err != nil {
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

func validateDecisionWorkflowModelRefs(decision Decision) error {
	if decision.Algorithm == nil || decision.Algorithm.Workflows == nil {
		return nil
	}
	workflows := decision.Algorithm.Workflows
	allowed := decisionModelRefSet(decision.ModelRefs)
	if workflowMode(workflows.Mode) == WorkflowModeStatic {
		for i, role := range workflows.Roles {
			for j, model := range role.Models {
				normalized := strings.TrimSpace(model)
				if !allowed[normalized] {
					return fmt.Errorf(
						"decision '%s': algorithm.workflows.roles[%d].models[%d] references model %q outside decision modelRefs",
						decision.Name,
						i,
						j,
						model,
					)
				}
			}
		}
	}
	finalModel := strings.TrimSpace(workflows.Final.Model)
	if finalModel != "" && !allowed[finalModel] {
		return fmt.Errorf(
			"decision '%s': algorithm.workflows.final.model references model %q outside decision modelRefs",
			decision.Name,
			workflows.Final.Model,
		)
	}
	return nil
}

func workflowMode(mode string) string {
	normalized := strings.TrimSpace(mode)
	if normalized == "" {
		return WorkflowModeStatic
	}
	return normalized
}

func decisionModelRefSet(refs []ModelRef) map[string]bool {
	allowed := make(map[string]bool, len(refs))
	for _, ref := range refs {
		allowed[strings.TrimSpace(ref.Model)] = true
	}
	return allowed
}

func validateDecisionCandidateIterations(decision Decision) error {
	for i, iter := range decision.CandidateIterations {
		context := fmt.Sprintf("decision '%s', candidateIterations[%d]", decision.Name, i)
		if err := validateDecisionCandidateIteration(decision, iter, context); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionCandidateIteration(decision Decision, iter CandidateIterationConfig, context string) error {
	if strings.TrimSpace(iter.Variable) == "" {
		return fmt.Errorf("%s: variable cannot be empty", context)
	}
	if err := validateDecisionCandidateIterationSource(decision, iter, context); err != nil {
		return err
	}
	return validateDecisionCandidateIterationOutputs(iter, context)
}

func validateDecisionCandidateIterationSource(decision Decision, iter CandidateIterationConfig, context string) error {
	switch strings.TrimSpace(iter.Source) {
	case "decision.candidates":
		if len(decision.ModelRefs) == 0 {
			return fmt.Errorf("%s: source decision.candidates requires non-empty modelRefs", context)
		}
	case "models":
		return validateDecisionCandidateIterationModels(iter.Models, context)
	default:
		return fmt.Errorf("%s: unsupported source %q", context, iter.Source)
	}
	return nil
}

func validateDecisionCandidateIterationModels(models []ModelRef, context string) error {
	if len(models) == 0 {
		return fmt.Errorf("%s: source models requires at least one model", context)
	}
	for j, modelRef := range models {
		if strings.TrimSpace(modelRef.Model) == "" {
			return fmt.Errorf("%s, models[%d]: model name cannot be empty", context, j)
		}
	}
	return nil
}

func validateDecisionCandidateIterationOutputs(iter CandidateIterationConfig, context string) error {
	for j, output := range iter.Outputs {
		if output.Type != "model" {
			return fmt.Errorf("%s, outputs[%d]: unsupported output type %q", context, j, output.Type)
		}
		if output.Value != iter.Variable {
			return fmt.Errorf("%s, outputs[%d]: model output must reference variable %q", context, j, iter.Variable)
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
		if tsCfg := decision.GetToolSelectionConfig(); tsCfg != nil {
			if err := tsCfg.Validate(); err != nil {
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

func validateDecisionAlgorithmConfig(decisionName string, modelRefs []ModelRef, algorithm *AlgorithmConfig) error {
	if algorithm == nil {
		return nil
	}

	normalizedType := strings.ToLower(strings.TrimSpace(algorithm.Type))
	displayType := strings.TrimSpace(algorithm.Type)
	if displayType == "" {
		displayType = "<empty>"
	}
	if normalizedType == "session_aware" || algorithm.SessionAware != nil {
		return fmt.Errorf(
			"decision '%s': algorithm.type=session_aware is no longer supported; remove algorithm.type=session_aware and enable global.router.learning.protection. If this decision needs an explicit base selector, configure a normal algorithm.type; otherwise omit algorithm",
			decisionName,
		)
	}
	if err := validateMigratedLearningAlgorithm(decisionName, normalizedType, algorithm); err != nil {
		return err
	}

	configuredBlocks := configuredAlgorithmBlocks(algorithm)
	if len(configuredBlocks) > 1 {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s cannot be combined with multiple algorithm config blocks: %s",
			decisionName,
			displayType,
			strings.Join(configuredBlocks, ", "),
		)
	}

	expectedBlock, hasExpectedBlock := expectedAlgorithmBlock(normalizedType)
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

	if err := validateSpecializedAlgorithmConfig(decisionName, modelRefs, normalizedType, algorithm); err != nil {
		return err
	}

	return nil
}

func validateMigratedLearningAlgorithm(decisionName string, normalizedType string, algorithm *AlgorithmConfig) error {
	migrations := map[string]string{
		"elo":             "global.router.learning.adaptation",
		"rl_driven":       "global.router.learning.adaptation",
		"gmtrouter":       "global.router.learning.adaptation",
		"bandit":          "global.router.learning.adaptation",
		"personalization": "global.router.learning.adaptation",
	}
	if target, ok := migrations[normalizedType]; ok {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s has moved to %s; remove the learning algorithm type and choose a request-time base algorithm only when needed",
			decisionName,
			normalizedType,
			target,
		)
	}
	if algorithm.Elo != nil {
		return fmt.Errorf("decision '%s': algorithm.elo is no longer supported; use global.router.learning.adaptation", decisionName)
	}
	if algorithm.RLDriven != nil {
		return fmt.Errorf("decision '%s': algorithm.rl_driven is no longer supported; use global.router.learning.adaptation", decisionName)
	}
	if algorithm.GMTRouter != nil {
		return fmt.Errorf("decision '%s': algorithm.gmtrouter is no longer supported; use global.router.learning.adaptation", decisionName)
	}
	return nil
}

func configuredAlgorithmBlocks(algorithm *AlgorithmConfig) []string {
	configuredBlocks := make([]string, 0, 14)
	addBlock := func(name string, configured bool) {
		if configured {
			configuredBlocks = append(configuredBlocks, name)
		}
	}

	addBlock("confidence", algorithm.Confidence != nil)
	addBlock("ratings", algorithm.Ratings != nil)
	addBlock("remom", algorithm.ReMoM != nil)
	addBlock("fusion", algorithm.Fusion != nil)
	addBlock("workflows", algorithm.Workflows != nil)
	addBlock("elo", algorithm.Elo != nil)
	addBlock("router_dc", algorithm.RouterDC != nil)
	addBlock("automix", algorithm.AutoMix != nil)
	addBlock("hybrid", algorithm.Hybrid != nil)
	addBlock("rl_driven", algorithm.RLDriven != nil)
	addBlock("gmtrouter", algorithm.GMTRouter != nil)
	addBlock("latency_aware", algorithm.LatencyAware != nil)
	addBlock("multi_factor", algorithm.MultiFactor != nil)
	addBlock("session_aware", algorithm.SessionAware != nil)
	return configuredBlocks
}

func expectedAlgorithmBlock(normalizedType string) (string, bool) {
	expectedBlockByType := map[string]string{
		"confidence":    "confidence",
		"ratings":       "ratings",
		"remom":         "remom",
		"fusion":        "fusion",
		"workflows":     "workflows",
		"router_dc":     "router_dc",
		"automix":       "automix",
		"hybrid":        "hybrid",
		"latency_aware": "latency_aware",
		"multi_factor":  "multi_factor",
	}
	expectedBlock, ok := expectedBlockByType[normalizedType]
	return expectedBlock, ok
}

func validateSpecializedAlgorithmConfig(decisionName string, modelRefs []ModelRef, normalizedType string, algorithm *AlgorithmConfig) error {
	switch normalizedType {
	case "latency_aware":
		if algorithm.LatencyAware == nil {
			return fmt.Errorf("decision '%s': algorithm.type=latency_aware requires algorithm.latency_aware configuration", decisionName)
		}
		if err := validateLatencyAwareAlgorithmConfig(algorithm.LatencyAware); err != nil {
			return fmt.Errorf("decision '%s', algorithm.latency_aware: %w", decisionName, err)
		}
	case "remom":
		if err := ValidateReMoMAlgorithmConfig(algorithm.ReMoM); err != nil {
			return fmt.Errorf("decision '%s', algorithm.remom: %w", decisionName, err)
		}
		if err := ValidateReMoMModelRefs(algorithm.ReMoM, modelRefs); err != nil {
			return fmt.Errorf("decision '%s', algorithm.remom: %w", decisionName, err)
		}
	case "fusion":
		if err := ValidateFusionAlgorithmConfig(algorithm.Fusion); err != nil {
			return fmt.Errorf("decision '%s', algorithm.fusion: %w", decisionName, err)
		}
	case "workflows":
		if err := ValidateWorkflowsAlgorithmConfig(algorithm.Workflows); err != nil {
			return fmt.Errorf("decision '%s', algorithm.workflows: %w", decisionName, err)
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
