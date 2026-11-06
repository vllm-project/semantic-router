package config

import (
	"fmt"
	"strings"
)

func validateProjectionContracts(cfg *RouterConfig) error {
	if err := validateProjectionPartitions(cfg); err != nil {
		return err
	}
	scoreNames, err := validateProjectionScores(cfg)
	if err != nil {
		return err
	}
	outputNames, err := validateProjectionMappings(cfg, scoreNames)
	if err != nil {
		return err
	}
	err = validateProjectionScoreDependencyOrder(cfg.Projections.Scores, cfg.Projections.Mappings)
	if err != nil {
		return err
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		if err := validateDecisionProjectionReferences(decision.Name, &decision.Rules, outputNames); err != nil {
			return err
		}
	}
	return nil
}

func validateProjectionPartitions(cfg *RouterConfig) error {
	domains := make(map[string]bool, len(cfg.Categories))
	for _, category := range cfg.Categories {
		domains[category.Name] = true
	}
	for _, partition := range cfg.Projections.Partitions {
		if err := validateProjectionPartition(partition, domains); err != nil {
			return err
		}
	}
	return nil
}

func validateProjectionPartition(partition ProjectionPartition, domains map[string]bool) error {
	if len(partition.Members) == 0 {
		return fmt.Errorf("routing.projections.partitions[%q]: members cannot be empty", partition.Name)
	}
	if err := validateProjectionPartitionSemantics(partition); err != nil {
		return err
	}
	if partition.Default == "" {
		return fmt.Errorf("routing.projections.partitions[%q]: default is required", partition.Name)
	}
	for _, member := range partition.Members {
		if !domains[member] {
			return fmt.Errorf("routing.projections.partitions[%q]: member %q is not a declared domain signal", partition.Name, member)
		}
	}
	if err := validateProjectionPartitionTemperature(partition); err != nil {
		return err
	}
	return validateProjectionPartitionDefault(partition)
}

func validateProjectionPartitionSemantics(partition ProjectionPartition) error {
	switch partition.Semantics {
	case "exclusive", "softmax_exclusive":
		return nil
	default:
		return fmt.Errorf(
			"routing.projections.partitions[%q]: unsupported semantics %q (supported: exclusive, softmax_exclusive)",
			partition.Name,
			partition.Semantics,
		)
	}
}

func validateProjectionPartitionTemperature(partition ProjectionPartition) error {
	if partition.Semantics == "softmax_exclusive" && partition.Temperature <= 0 {
		return fmt.Errorf("routing.projections.partitions[%q]: softmax_exclusive requires temperature > 0", partition.Name)
	}
	return nil
}

func validateProjectionPartitionDefault(partition ProjectionPartition) error {
	if !containsProjectionMember(partition.Members, partition.Default) {
		return fmt.Errorf("routing.projections.partitions[%q]: default %q must also appear in members", partition.Name, partition.Default)
	}
	return nil
}

func containsProjectionMember(members []string, target string) bool {
	for _, member := range members {
		if member == target {
			return true
		}
	}
	return false
}

func validateProjectionScores(cfg *RouterConfig) (map[string]struct{}, error) {
	names := make(map[string]struct{}, len(cfg.Projections.Scores))
	declaredSignals := projectionDeclaredSignals(cfg)
	outputToSource := projectionOutputToSourceScore(cfg.Projections.Mappings)
	for _, score := range cfg.Projections.Scores {
		if err := validateProjectionScore(score, names, declaredSignals, cfg, outputToSource); err != nil {
			return nil, err
		}
	}
	return names, nil
}

func projectionOutputToSourceScore(mappings []ProjectionMapping) map[string]string {
	m := make(map[string]string)
	for _, mapping := range mappings {
		for _, output := range mapping.Outputs {
			if output.Name != "" {
				m[output.Name] = mapping.Source
			}
		}
	}
	return m
}

func buildProjectionScoreAdj(scores []ProjectionScore, outputToSource map[string]string) map[string][]string {
	adj := make(map[string][]string, len(scores))
	for _, score := range scores {
		for _, input := range score.Inputs {
			if !strings.EqualFold(input.Type, SignalTypeProjection) {
				continue
			}
			vs := strings.ToLower(strings.TrimSpace(input.ValueSource))
			if vs == ProjectionValueSourceConfidence {
				if src, ok := outputToSource[input.Name]; ok {
					adj[score.Name] = append(adj[score.Name], src)
				}
			} else {
				adj[score.Name] = append(adj[score.Name], input.Name)
			}
		}
	}
	return adj
}

func formatCyclePath(path []string, name string) string {
	cycle := make([]string, len(path)+1)
	copy(cycle, path)
	cycle[len(path)] = name
	for i, n := range cycle {
		if n == name {
			return strings.Join(cycle[i:], " -> ")
		}
	}
	return strings.Join(cycle, " -> ")
}

func validateProjectionScoreDependencyOrder(scores []ProjectionScore, mappings []ProjectionMapping) error {
	nameIndex := make(map[string]int, len(scores))
	for i, s := range scores {
		nameIndex[s.Name] = i
	}

	outputToSource := projectionOutputToSourceScore(mappings)
	adj := buildProjectionScoreAdj(scores, outputToSource)
	if len(adj) == 0 {
		return nil
	}

	const (
		unvisited = 0
		visiting  = 1
		visited   = 2
	)
	state := make(map[string]int, len(scores))
	var path []string

	var visit func(name string) error
	visit = func(name string) error {
		if state[name] == visited {
			return nil
		}
		if state[name] == visiting {
			return fmt.Errorf(
				"routing.projections.scores: dependency cycle detected: %s",
				formatCyclePath(path, name),
			)
		}
		state[name] = visiting
		path = append(path, name)
		for _, dep := range adj[name] {
			if _, ok := nameIndex[dep]; !ok {
				return fmt.Errorf(
					"routing.projections.scores[%q]: projection input references undefined score %q",
					name, dep,
				)
			}
			if err := visit(dep); err != nil {
				return err
			}
		}
		path = path[:len(path)-1]
		state[name] = visited
		return nil
	}

	for _, score := range scores {
		if state[score.Name] == unvisited {
			if err := visit(score.Name); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateProjectionScore(
	score ProjectionScore,
	names map[string]struct{},
	declaredSignals map[string]map[string]struct{},
	cfg *RouterConfig,
	outputToSource map[string]string,
) error {
	if score.Name == "" {
		return fmt.Errorf("routing.projections.scores: name cannot be empty")
	}
	if _, exists := names[score.Name]; exists {
		return fmt.Errorf("routing.projections.scores[%q]: duplicate score name", score.Name)
	}
	names[score.Name] = struct{}{}
	if score.Method != "weighted_sum" {
		return fmt.Errorf("routing.projections.scores[%q]: unsupported method %q (supported: weighted_sum)", score.Name, score.Method)
	}
	if len(score.Inputs) == 0 {
		return fmt.Errorf("routing.projections.scores[%q]: inputs cannot be empty", score.Name)
	}
	for _, input := range score.Inputs {
		if err := validateProjectionScoreInput(score.Name, input, declaredSignals, cfg, outputToSource); err != nil {
			return err
		}
	}
	return nil
}

func validateProjectionScoreInput(
	scoreName string,
	input ProjectionScoreInput,
	declaredSignals map[string]map[string]struct{},
	cfg *RouterConfig,
	outputToSource map[string]string,
) error {
	if !isProjectionInputTypeSupported(input.Type) {
		return fmt.Errorf(
			"routing.projections.scores[%q]: input %s(%q) uses unsupported type %q",
			scoreName,
			input.Type,
			input.Name,
			input.Type,
		)
	}
	if strings.EqualFold(input.Type, SignalTypeProjection) {
		return validateProjectionInputProjectionRef(scoreName, input, outputToSource)
	}
	if strings.EqualFold(input.Type, SignalTypeSystemOne) {
		if err := validateSystemOneProjectionInput(cfg, scoreName, input); err != nil {
			return err
		}
		return validateProjectionInputValueSource(scoreName, input)
	}
	if signalType := strings.ToLower(strings.TrimSpace(input.Type)); IsApplicationSignalType(signalType) {
		if err := validateApplicationProjectionInput(cfg, scoreName, signalType, input); err != nil {
			return err
		}
		return validateProjectionInputValueSource(scoreName, input)
	}
	if !projectionInputDeclared(declaredSignals, input.Type, input.Name) {
		return fmt.Errorf(
			"routing.projections.scores[%q]: input %s(%q) is not declared in routing.signals",
			scoreName,
			input.Type,
			input.Name,
		)
	}
	return validateProjectionInputValueSource(scoreName, input)
}

// validateApplicationProjectionInput checks an application signal input: a
// declared rule, or "<rule>:<label>" for one option of its preset, whose
// value is that option's probability.
func validateApplicationProjectionInput(cfg *RouterConfig, scoreName, signalType string, input ProjectionScoreInput) error {
	name, option, labelled := strings.Cut(input.Name, ":")
	rule, ok := cfg.ApplicationRuleByName(signalType, name)
	if !ok {
		return fmt.Errorf(
			"routing.projections.scores[%q]: input %s(%q) is not declared in routing.signals",
			scoreName,
			input.Type,
			input.Name,
		)
	}
	if labelled && !stringSliceContains(rule.Rule.Question.OptionKeys(), option) {
		return fmt.Errorf(
			"routing.projections.scores[%q]: input %s(%q) names %q, which is not a %s label; the labels are %s",
			scoreName,
			input.Type,
			input.Name,
			option,
			signalType,
			strings.Join(rule.Rule.Question.OptionKeys(), ", "),
		)
	}
	return nil
}

// validateSystemOneProjectionInput checks a systemone input: a declared rule,
// or "<rule>:<option>" for one option of a choice, set or span question,
// whose value is that option's probability.
func validateSystemOneProjectionInput(cfg *RouterConfig, scoreName string, input ProjectionScoreInput) error {
	name, option, labelled := strings.Cut(input.Name, ":")
	rule := systemOneRuleByName(cfg.SystemOneRules, name)
	if rule == nil {
		return fmt.Errorf(
			"routing.projections.scores[%q]: input %s(%q) is not declared in routing.signals",
			scoreName,
			input.Type,
			input.Name,
		)
	}
	if labelled && (!rule.Question.Labelled() || !stringSliceContains(rule.Question.OptionKeys(), option)) {
		return fmt.Errorf(
			"routing.projections.scores[%q]: input %s(%q) names %q, which is not an option of the %s question %q",
			scoreName,
			input.Type,
			input.Name,
			option,
			rule.Question.Type,
			rule.Name,
		)
	}
	return nil
}

func validateProjectionInputProjectionRef(scoreName string, input ProjectionScoreInput, outputToSource map[string]string) error {
	if input.Name == "" {
		return fmt.Errorf(
			"routing.projections.scores[%q]: projection input requires a name referencing a declared score or mapping output",
			scoreName,
		)
	}
	switch strings.ToLower(strings.TrimSpace(input.ValueSource)) {
	case "", ProjectionValueSourceScore:
		return nil
	case ProjectionValueSourceConfidence:
		if _, ok := outputToSource[input.Name]; !ok {
			return fmt.Errorf(
				"routing.projections.scores[%q]: projection input %q with value_source \"confidence\" references undefined mapping output",
				scoreName,
				input.Name,
			)
		}
		return nil
	default:
		return fmt.Errorf(
			"routing.projections.scores[%q]: projection input %q has unsupported value_source %q (supported: score, confidence)",
			scoreName,
			input.Name,
			input.ValueSource,
		)
	}
}

func validateProjectionInputValueSource(scoreName string, input ProjectionScoreInput) error {
	switch input.ValueSource {
	case "", ProjectionValueSourceBinary, ProjectionValueSourceConfidence, ProjectionValueSourceRaw:
		return nil
	default:
		return fmt.Errorf(
			"routing.projections.scores[%q]: input %s(%q) has unsupported value_source %q (supported: binary, confidence, raw)",
			scoreName,
			input.Type,
			input.Name,
			input.ValueSource,
		)
	}
}

func projectionDeclaredSignals(cfg *RouterConfig) map[string]map[string]struct{} {
	declared := map[string]map[string]struct{}{
		SignalTypeKeyword:       collectKeywordRuleNames(cfg.KeywordRules),
		SignalTypeLanguage:      collectLanguageRuleNames(cfg.LanguageRules),
		SignalTypeContext:       collectContextRuleNames(cfg.ContextRules),
		SignalTypeAuthz:         collectRoleBindingRoles(cfg.GetRoleBindings()),
		SignalTypeConversation:  collectConversationRuleNames(cfg.ConversationRules),
		SignalTypeMetadata:      collectMetadataRuleNames(cfg.MetadataRules),
		SignalTypeInputModality: collectInputModalityRuleNames(cfg.InputModalityRules),
		SignalTypeSystemOne:     collectSystemOneRuleNames(cfg.SystemOneRules),
	}
	for _, spec := range cfg.applicationRuleSpecs() {
		if declared[spec.signalType] == nil {
			declared[spec.signalType] = map[string]struct{}{}
		}
		declared[spec.signalType][spec.name] = struct{}{}
	}
	return declared
}

func projectionInputDeclared(declared map[string]map[string]struct{}, signalType string, name string) bool {
	names, ok := declared[signalType]
	if !ok {
		return false
	}
	_, exists := names[name]
	return exists
}

func validateProjectionMappings(
	cfg *RouterConfig,
	scoreNames map[string]struct{},
) (map[string]struct{}, error) {
	outputNames := make(map[string]struct{})
	for _, mapping := range cfg.Projections.Mappings {
		if err := validateProjectionMapping(mapping, scoreNames, outputNames); err != nil {
			return nil, err
		}
	}
	return outputNames, nil
}

func validateProjectionMapping(
	mapping ProjectionMapping,
	scoreNames map[string]struct{},
	outputNames map[string]struct{},
) error {
	if mapping.Name == "" {
		return fmt.Errorf("routing.projections.mappings: name cannot be empty")
	}
	if _, exists := scoreNames[mapping.Source]; !exists {
		return fmt.Errorf("routing.projections.mappings[%q]: source %q is not a declared projection score", mapping.Name, mapping.Source)
	}
	switch mapping.Method {
	case "", ProjectionMappingMethodThresholdBands, ProjectionMappingMethodMultiEmit:
		// supported (empty defaults to threshold_bands)
	default:
		return fmt.Errorf(
			"routing.projections.mappings[%q]: unsupported method %q (supported: %s, %s)",
			mapping.Name,
			mapping.Method,
			ProjectionMappingMethodThresholdBands,
			ProjectionMappingMethodMultiEmit,
		)
	}
	if len(mapping.Outputs) == 0 {
		return fmt.Errorf("routing.projections.mappings[%q]: outputs cannot be empty", mapping.Name)
	}
	if mapping.Method == ProjectionMappingMethodMultiEmit && len(mapping.Outputs) < 2 {
		return fmt.Errorf(
			"routing.projections.mappings[%q]: method %q requires at least 2 outputs (a single-output multi_emit is equivalent to threshold_bands)",
			mapping.Name,
			ProjectionMappingMethodMultiEmit,
		)
	}
	if err := validateProjectionCalibration(mapping); err != nil {
		return err
	}
	for _, output := range mapping.Outputs {
		if err := validateProjectionMappingOutput(mapping.Name, output, outputNames); err != nil {
			return err
		}
	}
	return nil
}

func validateProjectionCalibration(mapping ProjectionMapping) error {
	if mapping.Calibration == nil {
		return nil
	}
	switch mapping.Calibration.Method {
	case "", "sigmoid_distance":
		return nil
	default:
		return fmt.Errorf(
			"routing.projections.mappings[%q]: unsupported calibration method %q (supported: sigmoid_distance)",
			mapping.Name,
			mapping.Calibration.Method,
		)
	}
}

func validateProjectionMappingOutput(
	mappingName string,
	output ProjectionMappingOutput,
	outputNames map[string]struct{},
) error {
	if output.Name == "" {
		return fmt.Errorf("routing.projections.mappings[%q]: output name cannot be empty", mappingName)
	}
	if _, exists := outputNames[output.Name]; exists {
		return fmt.Errorf("routing.projections.mappings[%q]: duplicate output name %q", mappingName, output.Name)
	}
	if err := validateProjectionOutputThresholds(mappingName, output); err != nil {
		return err
	}
	outputNames[output.Name] = struct{}{}
	return nil
}

func validateProjectionOutputThresholds(mappingName string, output ProjectionMappingOutput) error {
	if output.GT == nil && output.GTE == nil && output.LT == nil && output.LTE == nil {
		return fmt.Errorf(
			"routing.projections.mappings[%q].outputs[%q]: at least one threshold bound is required",
			mappingName,
			output.Name,
		)
	}
	if output.GT != nil && output.GTE != nil {
		return fmt.Errorf("routing.projections.mappings[%q].outputs[%q]: cannot set both gt and gte", mappingName, output.Name)
	}
	if output.LT != nil && output.LTE != nil {
		return fmt.Errorf("routing.projections.mappings[%q].outputs[%q]: cannot set both lt and lte", mappingName, output.Name)
	}
	return nil
}

func validateDecisionProjectionReferences(decisionName string, node *RuleNode, outputs map[string]struct{}) error {
	if node == nil {
		return nil
	}

	if strings.EqualFold(node.Type, SignalTypeProjection) {
		if _, ok := outputs[node.Name]; !ok {
			return fmt.Errorf(
				"decision %q references projection %q, but no routing.projections.mappings output declares that name",
				decisionName,
				node.Name,
			)
		}
	}

	for i := range node.Conditions {
		if err := validateDecisionProjectionReferences(decisionName, &node.Conditions[i], outputs); err != nil {
			return err
		}
	}
	return nil
}
