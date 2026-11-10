package config

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func validateDecisionContracts(cfg *RouterConfig) error {
	if err := validateMetadataContracts(cfg); err != nil {
		return err
	}
	if err := validateInputModalityContracts(cfg); err != nil {
		return err
	}
	if err := validateSystemOneContracts(cfg); err != nil {
		return err
	}
	if err := validateDecisionModelContracts(cfg); err != nil {
		return err
	}
	if err := validateDecisionRetentionContracts(cfg); err != nil {
		return err
	}
	reportAmbiguousConfidencePools(cfg)
	return validateDecisionPluginContracts(cfg)
}

func validateDecisionRuleNode(cfg *RouterConfig, decisionName string, node *RuleNode, root bool) error {
	if node == nil {
		return nil
	}
	if node.OnUnknown != "" {
		if !root {
			return fmt.Errorf("decision '%s': on_unknown is only supported on the root rules node", decisionName)
		}
		if !node.OnUnknown.IsValid() {
			return fmt.Errorf("decision '%s': rules on_unknown must be %s", decisionName, UnknownPolicyChoices())
		}
		if ruleTreeSetsOnError(node) {
			return fmt.Errorf("decision '%s': %s", decisionName, OnUnknownOnErrorConflictMessage)
		}
	}
	if node.IsLeaf() {
		return validateDecisionLeafNode(cfg, decisionName, node)
	}
	for i := range node.Conditions {
		if err := validateDecisionRuleNode(cfg, decisionName, &node.Conditions[i], false); err != nil {
			return err
		}
	}
	return nil
}

const OnUnknownOnErrorConflictMessage = "condition on_error has no effect when rules.on_unknown is set; remove one of them"

func ruleTreeSetsOnError(node *RuleNode) bool {
	if node == nil {
		return false
	}
	if node.OnError != "" {
		return true
	}
	for i := range node.Conditions {
		if ruleTreeSetsOnError(&node.Conditions[i]) {
			return true
		}
	}
	return false
}

func validateDecisionLeafNode(
	cfg *RouterConfig,
	decisionName string,
	node *RuleNode,
) error {
	systemOne := strings.EqualFold(node.Type, SignalTypeSystemOne)
	application := IsApplicationSignalType(strings.ToLower(strings.TrimSpace(node.Type)))
	if node.Label != "" && !systemOne && !application {
		return fmt.Errorf("decision '%s': label is only supported on systemone and application signal conditions", decisionName)
	}
	if systemOne {
		if err := validateSystemOneLeaf(cfg, decisionName, node); err != nil {
			return err
		}
	}
	if application {
		if err := validateApplicationLeaf(cfg, decisionName, node); err != nil {
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
	if node.OnError != "" && !systemOne {
		return fmt.Errorf(
			"decision '%s': condition %s(%q) on_error is only supported for systemone conditions",
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
	if decision.Algorithm.IsNative() {
		return validateNativeDecisionSurfaces(decision)
	}
	for i, modelRef := range decision.ModelRefs {
		if modelRef.Model == "" {
			return fmt.Errorf("decision '%s', modelRefs[%d]: model name cannot be empty", decision.Name, i)
		}
		if modelRef.UseReasoning == nil {
			return fmt.Errorf("decision '%s', model '%s': missing required field 'use_reasoning'", decision.Name, modelRef.Model)
		}
		if err := validateModelRefReasoningControl(cfg, decision.Name, i, modelRef); err != nil {
			return err
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

// validateDecisionPluginContracts checks a recipe's default plugins before its
// decisions', which layer them, so a refused default names its own path.
func validateDecisionPluginContracts(cfg *RouterConfig) error {
	if err := validateDefaultPluginContracts(cfg); err != nil {
		return err
	}
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

// validateGlobalPluginContracts checks global.plugins at every load: a request
// that routes through no recipe runs them as written.
func validateGlobalPluginContracts(cfg *RouterConfig) error {
	configuration := func(_ int, pluginType string) string {
		return fmt.Sprintf("global.plugins[%s].configuration", NormalizeDecisionPluginType(pluginType))
	}
	return validatePluginContracts(cfg, "global.plugins", cfg.GlobalPlugins, configuration)
}

// validateDefaultPluginContracts checks the plugins a request that matches no
// decision runs, global.plugins layered with its recipe's, as a decision's
// plugins are checked.
func validateDefaultPluginContracts(cfg *RouterConfig) error {
	recipes := cfg.Recipes
	if len(recipes) == 0 {
		recipes = []RoutingRecipe{*cfg.DefaultRecipe()}
	}
	for i := range recipes {
		owner := recipePath(string(recipes[i].Name)) + ".plugins"
		configuration := func(_ int, pluginType string) string {
			return fmt.Sprintf("%s[%s].configuration", owner, NormalizeDecisionPluginType(pluginType))
		}
		if err := validatePluginContracts(cfg, owner, recipes[i].Profile.DefaultPlugins, configuration); err != nil {
			return err
		}
	}
	return nil
}

func validateOneDecisionPluginContracts(
	cfg *RouterConfig,
	decision *Decision,
) error {
	configuration := func(index int, _ string) string {
		return cfg.ScopePath(fmt.Sprintf("decisions[%s].plugins[%d].configuration", decision.Name, index))
	}
	if err := validatePluginContracts(cfg, decision.Name, decision.Plugins, configuration); err != nil {
		return err
	}
	if err := validateDecisionSignalsCompression(decision, configuration); err != nil {
		return err
	}
	if err := validateDecisionReplayShadow(cfg, decision); err != nil {
		return err
	}
	warnCachedPersonalization(cfg, decision)
	return nil
}

// validateDecisionSignalsCompression refuses compress.signals in a decision's
// own context plugin: the routing signals read the request before a decision
// is chosen.
func validateDecisionSignalsCompression(decision *Decision, configuration func(index int, pluginType string) string) error {
	for index, plugin := range decision.DeclaredPlugins() {
		if NormalizeDecisionPluginType(plugin.Type) != DecisionPluginContext {
			continue
		}
		context := decodePlugin([]DecisionPlugin{plugin}, DecisionPluginContext, &ContextPluginConfig{})
		if context != nil && context.Compress != nil && context.Compress.Signals != nil {
			return fmt.Errorf("%s.compress.signals: the routing signals read the request before a decision is chosen; set compress.signals in global.plugins or the recipe's plugins", configuration(index, plugin.Type))
		}
	}
	return nil
}

// validatePluginContracts checks the plugins one request runs; owner names
// them, and configuration is the path of a plugin's configuration.
func validatePluginContracts(
	cfg *RouterConfig,
	owner string,
	plugins []DecisionPlugin,
	configuration func(index int, pluginType string) string,
) error {
	seenPluginTypes := make(map[string]string, len(plugins))
	for index, plugin := range plugins {
		normalizedType := NormalizeDecisionPluginType(plugin.Type)
		if previous, exists := seenPluginTypes[normalizedType]; exists {
			return fmt.Errorf(
				"decision %q has duplicate plugin %q via %q and %q",
				owner,
				normalizedType,
				previous,
				plugin.Type,
			)
		}
		seenPluginTypes[normalizedType] = plugin.Type
		payload, err := DecodeDecisionPluginAt(PluginAt{Decision: owner, Index: index, Type: plugin.Type}, plugin)
		if err != nil {
			return err
		}
		path := configuration(index, plugin.Type)
		switch typed := payload.(type) {
		case *ToolsPluginConfig:
			err = validateToolsPlugin(cfg, path, typed)
		case *ContextPluginConfig:
			err = validateContextPlugin(cfg, path, typed)
		case *ResponseCachePluginConfig:
			err = validateMatch(cfg, path+".semantic.match", typed.SemanticMatch(), MatchResponseCache)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// warnCachedPersonalization warns that a decision's response cache is read
// past: a cached response would miss the documents and memories its context
// retrieves.
func warnCachedPersonalization(cfg *RouterConfig, decision *Decision) {
	cacheCfg := decision.GetResponseCacheConfig()
	plugin := cfg.ContextPlugin(decision)
	if cacheCfg == nil || !cacheCfg.Enabled || plugin == nil || len(plugin.Sources) == 0 {
		return
	}
	sources := make([]string, len(plugin.Sources))
	for index, source := range plugin.Sources {
		sources[index] = source.Type
	}
	logging.Warnf("Decision '%s': response_cache is enabled alongside the context plugin's %s sources. "+
		"Cache reads will be automatically bypassed to preserve personalized responses. "+
		"Cache writes still occur for observability. Remove the cache plugin if this is intentional.",
		decision.Name, strings.Join(sources, " and "))
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

	configuredBlocks := configuredDecisionAlgorithmBlocks(algorithm)
	terminal, err := validateAlgorithmBlockContract(
		decisionName,
		normalizedType,
		displayType,
		configuredBlocks,
	)
	if err != nil {
		return err
	}
	if err := validateExecutorSettings(decisionName, normalizedType, algorithm); err != nil {
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
	if normalizedType == "session_aware" {
		return "", "", fmt.Errorf(
			"decision '%s': algorithm.type=session_aware is no longer supported; remove algorithm.type=session_aware and enable global.router.learning.protection. If this decision needs an explicit base selector, configure a normal algorithm.type; otherwise omit algorithm",
			decisionName,
		)
	}
	if err := validateMigratedLearningAlgorithm(decisionName, normalizedType); err != nil {
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
	configField, supported := DecisionAlgorithmConfigField(normalizedType)
	return supported && configField == ""
}

func validateMigratedLearningAlgorithm(decisionName string, normalizedType string) error {
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
	return nil
}

func expectedAlgorithmBlock(normalizedType string) (string, bool) {
	expectedBlock, supported := DecisionAlgorithmConfigField(normalizedType)
	return expectedBlock, supported && expectedBlock != ""
}

func validateSpecializedAlgorithmConfig(decisionName string, modelRefs []ModelRef, normalizedType string, algorithm *AlgorithmConfig) error {
	switch normalizedType {
	case DecisionAlgorithmEscalate:
		return validateEscalateAlgorithmConfig(decisionName, modelRefs, algorithm)
	case DecisionAlgorithmReMoM:
		return validateDecisionReMoMAlgorithm(decisionName, modelRefs, algorithm.ReMoM)
	case DecisionAlgorithmFusion:
		return validateDecisionFusionAlgorithm(decisionName, modelRefs, algorithm.Fusion)
	case DecisionAlgorithmWorkflows:
		return validateDecisionWorkflowsAlgorithm(decisionName, modelRefs, algorithm.Workflows)
	case DecisionAlgorithmSystemOne:
		return validateSystemOneSelectorConfig(decisionName, modelRefs, algorithm)
	case DecisionAlgorithmMultiFactor:
		return validateDecisionMultiFactorAlgorithm(decisionName, algorithm.MultiFactor)
	}
	if _, registered, err := DecodeDecisionAlgorithm(decisionName, algorithm); registered {
		return err
	}
	return nil
}

// validateExecutorSettings checks the executors' shared budget and on_error,
// and refuses them on a selector, which calls one candidate. Escalate's
// on_error depends on its path and is checked with its block.
func validateExecutorSettings(decisionName, algorithmType string, algorithm *AlgorithmConfig) error {
	if !IsExecutorAlgorithmType(algorithmType) {
		if algorithm.Budget != nil || algorithm.OnError != "" {
			return fmt.Errorf("decision '%s': algorithm.budget and algorithm.on_error apply to executors; %s is a selector", decisionName, algorithmType)
		}
		return nil
	}
	if algorithmType == DecisionAlgorithmEscalate {
		return nil
	}
	switch algorithm.OnError {
	case "", FusionOnErrorSkip, FusionOnErrorFail:
	default:
		return fmt.Errorf("decision '%s': algorithm.on_error of %s must be %q or %q, got %q", decisionName, algorithmType, FusionOnErrorSkip, FusionOnErrorFail, algorithm.OnError)
	}
	if err := algorithm.Budget.Validate(); err != nil {
		return fmt.Errorf("decision '%s': %w", decisionName, err)
	}
	return nil
}

func validateDecisionMultiFactorAlgorithm(decisionName string, cfg *MultiFactorSelectionConfig) error {
	if cfg == nil {
		return fmt.Errorf("decision '%s': algorithm.type=multi_factor requires algorithm.multi_factor configuration", decisionName)
	}
	path := fmt.Sprintf("decision '%s', algorithm.multi_factor", decisionName)
	if cfg.ExpectedOutputTokens != nil && *cfg.ExpectedOutputTokens <= 0 {
		return fmt.Errorf("%s.expected_output_tokens must be positive", path)
	}
	if err := validateMultiFactorObjective(cfg, path); err != nil {
		return err
	}
	if err := validateMultiFactorWeights(cfg.Weights, path+".weights"); err != nil {
		return err
	}
	if cfg.LatencyPercentile < 0 || cfg.LatencyPercentile > 100 {
		return fmt.Errorf("%s.latency_percentile must be within [1, 100] when declared", path)
	}
	switch cfg.LatencyMetric {
	case "", "ttft", "tpot":
	default:
		return fmt.Errorf("%s.latency_metric must be %q or %q", path, "ttft", "tpot")
	}
	switch cfg.OnNoCandidates {
	case "", "cheapest", "first", "fail":
	default:
		return fmt.Errorf("%s.on_no_candidates must be %q, %q, or %q", path, "cheapest", "first", "fail")
	}
	if cfg.Quality == nil {
		return nil
	}
	if strings.TrimSpace(cfg.Quality.Index) == "" {
		return fmt.Errorf("%s.quality: index is required", path)
	}
	if cfg.Quality.Index != strings.TrimSpace(cfg.Quality.Index) {
		return fmt.Errorf("%s.quality: index must not contain surrounding whitespace", path)
	}
	if cfg.Quality.MinCoverage < 0 || cfg.Quality.MinCoverage > 1 || math.IsNaN(cfg.Quality.MinCoverage) {
		return fmt.Errorf("%s.quality.min_coverage must be within [0, 1]", path)
	}
	if cfg.Quality.MinScore != nil && (math.IsNaN(*cfg.Quality.MinScore) || math.IsInf(*cfg.Quality.MinScore, 0)) {
		return fmt.Errorf("%s.quality.min_score must be finite", path)
	}
	if cfg.Quality.MinScore != nil && cfg.Quality.OnMissing == QualityEvidenceOnMissingDisable {
		return fmt.Errorf("%s.quality.min_score requires on_missing=%q", path, QualityEvidenceOnMissingExclude)
	}
	switch cfg.Quality.OnMissing {
	case "", QualityEvidenceOnMissingExclude, QualityEvidenceOnMissingDisable:
		return nil
	default:
		return fmt.Errorf("%s.quality: on_missing must be %q or %q", path, QualityEvidenceOnMissingExclude, QualityEvidenceOnMissingDisable)
	}
}

func validateMultiFactorObjective(cfg *MultiFactorSelectionConfig, path string) error {
	if cfg.Objective == nil || cfg.Objective.Strategy == "" || cfg.Objective.Strategy == MultiFactorObjectiveWeighted {
		if cfg.Objective != nil && len(cfg.Objective.Priorities) > 0 {
			return fmt.Errorf("%s.objective.priorities require strategy=%q", path, MultiFactorObjectiveLexicographic)
		}
		return nil
	}
	if cfg.Objective.Strategy != MultiFactorObjectiveLexicographic {
		return fmt.Errorf("%s.objective.strategy must be %q or %q", path, MultiFactorObjectiveWeighted, MultiFactorObjectiveLexicographic)
	}
	if cfg.Weights != nil {
		return fmt.Errorf("%s.weights cannot be combined with a lexicographic objective", path)
	}
	if len(cfg.Objective.Priorities) == 0 {
		return fmt.Errorf("%s.objective.priorities cannot be empty for a lexicographic objective", path)
	}
	seen := map[string]struct{}{}
	for index, priority := range cfg.Objective.Priorities {
		switch priority.Factor {
		case MultiFactorFactorQuality, MultiFactorFactorLatency, MultiFactorFactorCost, MultiFactorFactorLoad:
		default:
			return fmt.Errorf("%s.objective.priorities[%d].factor %q is unsupported", path, index, priority.Factor)
		}
		if _, duplicate := seen[priority.Factor]; duplicate {
			return fmt.Errorf("%s.objective.priorities contains duplicate factor %q", path, priority.Factor)
		}
		seen[priority.Factor] = struct{}{}
		if priority.Tolerance < 0 || priority.Tolerance > 1 || math.IsNaN(priority.Tolerance) {
			return fmt.Errorf("%s.objective.priorities[%d].tolerance must be within [0, 1]", path, index)
		}
	}
	return nil
}

func validateMultiFactorWeights(weights *MultiFactorWeightsConfig, path string) error {
	if weights == nil {
		return nil
	}
	for name, value := range map[string]float64{
		"quality": weights.Quality,
		"latency": weights.Latency,
		"cost":    weights.Cost,
		"load":    weights.Load,
	} {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s.%s must be finite and non-negative", path, name)
		}
	}
	return nil
}

func wrapAlgorithmValidationError(decisionName, algorithmType string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("decision '%s', algorithm.%s: %w", decisionName, algorithmType, err)
}

func validateDecisionReMoMAlgorithm(decisionName string, modelRefs []ModelRef, cfg *ReMoMAlgorithmConfig) error {
	if err := ValidateReMoMAlgorithmConfig(cfg); err != nil {
		return wrapAlgorithmValidationError(decisionName, "remom", err)
	}
	return wrapAlgorithmValidationError(decisionName, "remom", ValidateReMoMModelRefs(cfg, modelRefs))
}
