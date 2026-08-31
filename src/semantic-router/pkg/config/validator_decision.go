package config

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func validateDecisionContracts(cfg *RouterConfig) error {
	if err := validateMetadataContracts(cfg); err != nil {
		return err
	}
	if err := validateClassifierSignalContracts(cfg); err != nil {
		return err
	}
	if err := validateInputModalityContracts(cfg); err != nil {
		return err
	}
	if err := validateDecisionModelContracts(cfg); err != nil {
		return err
	}
	if err := validateDecisionEmitContracts(cfg); err != nil {
		return err
	}
	return validateDecisionPluginContracts(cfg)
}

func validateDecisionModelContracts(cfg *RouterConfig) error {
	for _, decision := range cfg.AllRoutingDecisions() {
		if err := validateDecisionRuleNode(cfg, decision.Name, &decision.Rules); err != nil {
			return err
		}
		if err := validateDecisionAnnotations(decision); err != nil {
			return err
		}
		if err := validateDecisionModelRefs(cfg, decision); err != nil {
			return err
		}
		if err := validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm); err != nil {
			return err
		}
		if err := validateDecisionPromptModel(cfg, decision); err != nil {
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

func validateDecisionRuleNode(cfg *RouterConfig, decisionName string, node *RuleNode) error {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		return validateDecisionLeafNode(cfg, decisionName, node)
	}
	for i := range node.Conditions {
		if err := validateDecisionRuleNode(cfg, decisionName, &node.Conditions[i]); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionLeafNode(
	cfg *RouterConfig,
	decisionName string,
	node *RuleNode,
) error {
	if node.Label != "" && !strings.EqualFold(node.Type, SignalTypeClassifier) {
		return fmt.Errorf("decision '%s': label is only supported on classifier conditions", decisionName)
	}
	if strings.EqualFold(node.Type, SignalTypeClassifier) {
		if err := validateClassifierDecisionLeaf(cfg, decisionName, node); err != nil {
			return err
		}
	}
	if strings.EqualFold(node.Type, SignalTypeMetadata) &&
		metadataRuleByName(cfg.MetadataRules, node.Name) == nil {
		return fmt.Errorf(
			"decision '%s': metadata condition references unknown signal %q",
			decisionName,
			node.Name,
		)
	}
	if node.OnError != "" && node.OnError != "no_match" && node.OnError != "match" {
		return fmt.Errorf(
			"decision '%s': condition %s(%q) on_error must be no_match or match",
			decisionName,
			node.Type,
			node.Name,
		)
	}
	if node.OnError != "" && !strings.EqualFold(node.Type, SignalTypeClassifier) {
		return fmt.Errorf(
			"decision '%s': condition %s(%q) on_error is only supported for classifier conditions",
			decisionName,
			node.Type,
			node.Name,
		)
	}
	return validateDecisionLeafPredicate(decisionName, node)
}

func metadataRuleByName(rules []MetadataRule, name string) *MetadataRule {
	for i := range rules {
		if rules[i].Name == name {
			return &rules[i]
		}
	}
	return nil
}

func validateClassifierDecisionLeaf(
	cfg *RouterConfig,
	decisionName string,
	node *RuleNode,
) error {
	rule := classifierSignalRuleByName(cfg.ClassifierRules, node.Name)
	if rule == nil {
		return fmt.Errorf(
			"decision '%s': classifier condition references unknown signal %q",
			decisionName,
			node.Name,
		)
	}
	if node.Label == "" || !stringSliceContains(rule.Labels, node.Label) {
		return fmt.Errorf(
			"decision '%s': classifier condition %q requires a declared label",
			decisionName,
			node.Name,
		)
	}
	if node.Predicate == nil {
		return fmt.Errorf(
			"decision '%s': classifier condition %q requires a score predicate",
			decisionName,
			node.Name,
		)
	}
	if rule.Type == ClassifierSignalTypeLocal {
		return validateLocalClassifierDecisionPredicate(decisionName, node)
	}
	return nil
}

func validateLocalClassifierDecisionPredicate(
	decisionName string,
	node *RuleNode,
) error {
	predicate := node.Predicate
	if predicate.GTE == nil || *predicate.GTE < 0.5 ||
		predicate.GT != nil || predicate.LT != nil || predicate.LTE != nil {
		return fmt.Errorf(
			"decision '%s': local classifier condition %q supports only predicate.gte >= 0.5",
			decisionName,
			node.Name,
		)
	}
	return nil
}

func validateDecisionLeafPredicate(decisionName string, node *RuleNode) error {
	if node.Predicate == nil {
		return nil
	}
	if err := validateNumericPredicateContract(node.Predicate); err != nil {
		return fmt.Errorf(
			"decision '%s': condition %s(%q) %w",
			decisionName,
			node.Type,
			node.Name,
			err,
		)
	}
	return nil
}

func classifierSignalRuleByName(
	rules []ClassifierSignalRule,
	name string,
) *ClassifierSignalRule {
	for i := range rules {
		if rules[i].Name == name {
			return &rules[i]
		}
	}
	return nil
}

func stringSliceContains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func validateDecisionAnnotations(decision Decision) error {
	if len(decision.Annotations) == 0 {
		return nil
	}
	if len(decision.Annotations) > 32 {
		return fmt.Errorf("decision '%s': annotations cannot contain more than 32 keys", decision.Name)
	}
	encoded, err := json.Marshal(decision.Annotations)
	if err != nil {
		return fmt.Errorf("decision '%s': annotations must be JSON-compatible: %w", decision.Name, err)
	}
	if len(encoded) > 4096 {
		return fmt.Errorf("decision '%s': annotations cannot exceed 4096 encoded bytes", decision.Name)
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
	for _, decision := range cfg.AllRoutingDecisions() {
		if err := validateOneDecisionPluginContracts(
			cfg,
			&decision,
		); err != nil {
			return err
		}
	}
	return nil
}

func validateOneDecisionPluginContracts(
	cfg *RouterConfig,
	decision *Decision,
) error {
	seenPluginTypes := make(map[string]string, len(decision.Plugins))
	for index, plugin := range decision.Plugins {
		normalizedType := NormalizeDecisionPluginType(plugin.Type)
		if previous, exists := seenPluginTypes[normalizedType]; exists {
			return fmt.Errorf(
				"decision %q has duplicate plugin %q via %q and %q",
				decision.Name,
				normalizedType,
				previous,
				plugin.Type,
			)
		}
		seenPluginTypes[normalizedType] = plugin.Type
		if err := validateDecisionPluginPayload(
			decision.Name,
			index,
			plugin,
		); err != nil {
			return err
		}
	}
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
	return validateDecisionRAGAndMemoryPlugins(cfg, decision)
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

	cacheCfg := decision.GetResponseCacheConfig()
	memCfg := decision.GetMemoryConfig()
	cacheActive := cacheCfg != nil && cacheCfg.Enabled
	ragActive := ragCfg != nil && ragCfg.Enabled
	memActive := memCfg != nil && memCfg.Enabled
	if !memActive && cfg.Memory.Enabled {
		memActive = memCfg == nil
	}
	if cacheActive && (ragActive || memActive) {
		logging.Warnf("Decision '%s': response_cache is enabled alongside %s. "+
			"Cache reads will be automatically bypassed to preserve personalized responses. "+
			"Cache writes still occur for observability. Remove the cache plugin if this is intentional.",
			decision.Name, cachePersonalizationConflictDescription(ragActive, memActive))
	}
	return validateDecisionContextCompressionRecovery(cfg, decision)
}

func validateDecisionContextCompressionRecovery(
	cfg *RouterConfig,
	decision *Decision,
) error {
	compression := decision.GetContextCompressionConfig()
	if compression == nil ||
		compression.Recovery == nil ||
		!compression.Recovery.Enabled {
		return nil
	}
	if !cfg.Looper.IsEnabled() {
		return fmt.Errorf(
			"decision %q: context_compression recovery requires global.integrations.looper.endpoint",
			decision.Name,
		)
	}
	store := strings.TrimSpace(compression.Recovery.Store)
	if store == "response_cache" {
		store = strings.TrimSpace(cfg.SemanticCache.BackendType)
	}
	switch store {
	case "redis":
		if cfg.SemanticCache.Redis == nil {
			return fmt.Errorf(
				"decision %q: context_compression recovery requires response_cache.redis configuration",
				decision.Name,
			)
		}
	case "valkey":
		if cfg.SemanticCache.Valkey == nil {
			return fmt.Errorf(
				"decision %q: context_compression recovery requires response_cache.valkey configuration",
				decision.Name,
			)
		}
	default:
		return fmt.Errorf(
			"decision %q: context_compression recovery requires a Redis or Valkey shared store",
			decision.Name,
		)
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

func validateDecisionAlgorithmConfig(decisionName string, modelRefs []ModelRef, algorithm *AlgorithmConfig) error {
	if algorithm == nil {
		return nil
	}

	normalizedType, displayType, err := normalizeDecisionAlgorithmType(
		decisionName,
		algorithm,
	)
	if err != nil {
		return err
	}
	if minimumErr := validateDecisionMinimumCandidates(decisionName, modelRefs, algorithm); minimumErr != nil {
		return minimumErr
	}

	configuredBlocks := configuredAlgorithmBlocks(algorithm)
	terminal, err := validateAlgorithmBlockContract(
		decisionName,
		normalizedType,
		displayType,
		configuredBlocks,
	)
	if err != nil {
		return err
	}
	if terminal {
		return nil
	}

	expectedBlock, _ := expectedAlgorithmBlock(normalizedType)
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

func normalizeDecisionAlgorithmType(
	decisionName string,
	algorithm *AlgorithmConfig,
) (string, string, error) {
	displayType := strings.TrimSpace(algorithm.Type)
	normalizedType := strings.ToLower(displayType)
	if displayType != "" && displayType != normalizedType {
		return "", "", fmt.Errorf(
			"decision '%s': algorithm.type must use lowercase canonical value %q",
			decisionName,
			normalizedType,
		)
	}
	if displayType == "" {
		displayType = "<empty>"
	}
	if normalizedType == "session_aware" || algorithm.SessionAware != nil {
		return "", "", fmt.Errorf(
			"decision '%s': algorithm.type=session_aware is no longer supported; remove algorithm.type=session_aware and enable global.router.learning.protection. If this decision needs an explicit base selector, configure a normal algorithm.type; otherwise omit algorithm",
			decisionName,
		)
	}
	if err := validateMigratedLearningAlgorithm(
		decisionName,
		normalizedType,
		algorithm,
	); err != nil {
		return "", "", err
	}
	return normalizedType, displayType, nil
}

func validateAlgorithmBlockContract(
	decisionName string,
	normalizedType string,
	displayType string,
	configuredBlocks []string,
) (bool, error) {
	if len(configuredBlocks) > 1 {
		return false, fmt.Errorf(
			"decision '%s': algorithm.type=%s cannot be combined with multiple algorithm config blocks: %s",
			decisionName,
			displayType,
			strings.Join(configuredBlocks, ", "),
		)
	}
	if _, hasExpectedBlock := expectedAlgorithmBlock(normalizedType); hasExpectedBlock {
		return false, nil
	}
	if !blocklessAlgorithmTypeSupported(normalizedType) {
		return false, fmt.Errorf(
			"decision '%s': unsupported algorithm.type=%s",
			decisionName,
			displayType,
		)
	}
	if len(configuredBlocks) > 0 {
		return false, fmt.Errorf(
			"decision '%s': algorithm.type=%s cannot be used with algorithm.%s configuration",
			decisionName,
			displayType,
			configuredBlocks[0],
		)
	}
	return true, nil
}

func blocklessAlgorithmTypeSupported(normalizedType string) bool {
	switch normalizedType {
	case "static", "knn", "kmeans", "svm", "mlp":
		return true
	default:
		return false
	}
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
	addBlock("prompt", algorithm.Prompt != nil)
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
		"prompt":        "prompt",
	}
	expectedBlock, ok := expectedBlockByType[normalizedType]
	return expectedBlock, ok
}

func validateSpecializedAlgorithmConfig(decisionName string, modelRefs []ModelRef, normalizedType string, algorithm *AlgorithmConfig) error {
	switch normalizedType {
	case "confidence":
		return wrapAlgorithmValidationError(decisionName, "confidence", ValidateConfidenceAlgorithmConfig(algorithm.Confidence))
	case "latency_aware":
		return validateDecisionLatencyAwareAlgorithm(decisionName, algorithm.LatencyAware)
	case "remom":
		return validateDecisionReMoMAlgorithm(decisionName, modelRefs, algorithm.ReMoM)
	case "fusion":
		return validateDecisionFusionAlgorithm(decisionName, modelRefs, algorithm.Fusion)
	case "workflows":
		return validateDecisionWorkflowsAlgorithm(decisionName, modelRefs, algorithm.Workflows)
	case "prompt":
		return validatePromptAlgorithmConfig(decisionName, modelRefs, algorithm)
	}
	return nil
}

func wrapAlgorithmValidationError(decisionName, algorithmType string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("decision '%s', algorithm.%s: %w", decisionName, algorithmType, err)
}

func validateDecisionLatencyAwareAlgorithm(decisionName string, cfg *LatencyAwareAlgorithmConfig) error {
	if cfg == nil {
		return fmt.Errorf("decision '%s': algorithm.type=latency_aware requires algorithm.latency_aware configuration", decisionName)
	}
	return wrapAlgorithmValidationError(decisionName, "latency_aware", validateLatencyAwareAlgorithmConfig(cfg))
}

func validateDecisionReMoMAlgorithm(decisionName string, modelRefs []ModelRef, cfg *ReMoMAlgorithmConfig) error {
	if err := ValidateReMoMAlgorithmConfig(cfg); err != nil {
		return wrapAlgorithmValidationError(decisionName, "remom", err)
	}
	return wrapAlgorithmValidationError(decisionName, "remom", ValidateReMoMModelRefs(cfg, modelRefs))
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
