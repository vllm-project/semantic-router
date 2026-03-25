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
	for _, decision := range cfg.Decisions {
		if err := validateDecisionProjectionReferences(decision.Name, &decision.Rules, outputNames); err != nil {
			return err
		}
	}
	return nil
}

func validateProjectionPartitions(cfg *RouterConfig) error {
	declaredTypes := projectionPartitionMemberTypes(cfg)
	for _, partition := range cfg.Projections.Partitions {
		if err := validateProjectionPartition(partition, declaredTypes); err != nil {
			return err
		}
	}
	return nil
}

func validateProjectionPartition(partition ProjectionPartition, declaredTypes map[string]string) error {
	if len(partition.Members) == 0 {
		return fmt.Errorf("routing.projections.partitions[%q]: members cannot be empty", partition.Name)
	}
	if err := validateProjectionPartitionSemantics(partition); err != nil {
		return err
	}
	if partition.Default == "" {
		return fmt.Errorf("routing.projections.partitions[%q]: default is required", partition.Name)
	}
	memberType, err := projectionPartitionMemberType(partition, declaredTypes)
	if err != nil {
		return err
	}
	if err := validateProjectionPartitionTemperature(partition); err != nil {
		return err
	}
	if err := validateProjectionPartitionDefault(partition); err != nil {
		return err
	}
	return validateProjectionPartitionType(partition, memberType)
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

func projectionPartitionMemberType(
	partition ProjectionPartition,
	declaredTypes map[string]string,
) (string, error) {
	memberType := ""
	for _, member := range partition.Members {
		currentType, ok := declaredTypes[member]
		if !ok {
			return "", fmt.Errorf("routing.projections.partitions[%q]: member %q is not a declared domain or embedding signal", partition.Name, member)
		}
		if memberType == "" {
			memberType = currentType
			continue
		}
		if currentType != memberType {
			return "", fmt.Errorf(
				"routing.projections.partitions[%q]: members must share one supported type (domain or embedding), found %q and %q",
				partition.Name,
				memberType,
				currentType,
			)
		}
	}
	return memberType, nil
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

func validateProjectionPartitionType(partition ProjectionPartition, memberType string) error {
	if memberType != SignalTypeDomain && memberType != SignalTypeEmbedding {
		return fmt.Errorf(
			"routing.projections.partitions[%q]: members must use domain or embedding signals, found %q",
			partition.Name,
			memberType,
		)
	}
	return nil
}

func projectionPartitionMemberTypes(cfg *RouterConfig) map[string]string {
	types := make(map[string]string, len(cfg.Categories)+len(cfg.EmbeddingRules))
	for _, category := range cfg.Categories {
		types[category.Name] = SignalTypeDomain
	}
	for _, rule := range cfg.EmbeddingRules {
		types[rule.Name] = SignalTypeEmbedding
	}
	return types
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
	for _, score := range cfg.Projections.Scores {
		if score.Name == "" {
			return nil, fmt.Errorf("routing.projections.scores: name cannot be empty")
		}
		if _, exists := names[score.Name]; exists {
			return nil, fmt.Errorf("routing.projections.scores[%q]: duplicate score name", score.Name)
		}
		names[score.Name] = struct{}{}
		if score.Method != "weighted_sum" {
			return nil, fmt.Errorf("routing.projections.scores[%q]: unsupported method %q (supported: weighted_sum)", score.Name, score.Method)
		}
		if len(score.Inputs) == 0 {
			return nil, fmt.Errorf("routing.projections.scores[%q]: inputs cannot be empty", score.Name)
		}
		for _, input := range score.Inputs {
			if !isProjectionInputTypeSupported(input.Type) {
				return nil, fmt.Errorf(
					"routing.projections.scores[%q]: input %s(%q) uses unsupported type %q",
					score.Name,
					input.Type,
					input.Name,
					input.Type,
				)
			}
			if !projectionInputDeclared(declaredSignals, input.Type, input.Name) {
				return nil, fmt.Errorf(
					"routing.projections.scores[%q]: input %s(%q) is not declared in routing.signals",
					score.Name,
					input.Type,
					input.Name,
				)
			}
			switch input.ValueSource {
			case "", "binary", "confidence":
			default:
				return nil, fmt.Errorf(
					"routing.projections.scores[%q]: input %s(%q) has unsupported value_source %q (supported: binary, confidence)",
					score.Name,
					input.Type,
					input.Name,
					input.ValueSource,
				)
			}
		}
	}
	return names, nil
}

func isProjectionInputTypeSupported(signalType string) bool {
	switch signalType {
	case SignalTypeKeyword,
		SignalTypeEmbedding,
		SignalTypeDomain,
		SignalTypeFactCheck,
		SignalTypeUserFeedback,
		SignalTypePreference,
		SignalTypeLanguage,
		SignalTypeContext,
		SignalTypeStructure,
		SignalTypeComplexity,
		SignalTypeModality,
		SignalTypeAuthz,
		SignalTypeJailbreak,
		SignalTypePII:
		return true
	default:
		return false
	}
}

func projectionDeclaredSignals(cfg *RouterConfig) map[string]map[string]struct{} {
	declared := map[string]map[string]struct{}{
		SignalTypeKeyword:      collectKeywordRuleNames(cfg.KeywordRules),
		SignalTypeEmbedding:    collectEmbeddingRuleNames(cfg.EmbeddingRules),
		SignalTypeDomain:       collectDomainNames(cfg.Categories),
		SignalTypeFactCheck:    collectFactCheckRuleNames(cfg.FactCheckRules),
		SignalTypeUserFeedback: collectUserFeedbackRuleNames(cfg.UserFeedbackRules),
		SignalTypePreference:   collectPreferenceRuleNames(cfg.PreferenceRules),
		SignalTypeLanguage:     collectLanguageRuleNames(cfg.LanguageRules),
		SignalTypeContext:      collectContextRuleNames(cfg.ContextRules),
		SignalTypeStructure:    collectStructureRuleNames(cfg.StructureRules),
		SignalTypeComplexity:   collectComplexityRuleNames(cfg.ComplexityRules),
		SignalTypeModality:     collectModalityRuleNames(cfg.ModalityRules),
		SignalTypeAuthz:        collectRoleBindingNames(cfg.GetRoleBindings()),
		SignalTypeJailbreak:    collectJailbreakRuleNames(cfg.JailbreakRules),
		SignalTypePII:          collectPIIRuleNames(cfg.PIIRules),
	}
	return declared
}

func projectionInputDeclared(declared map[string]map[string]struct{}, signalType string, name string) bool {
	names, ok := declared[signalType]
	if !ok {
		return false
	}
	if signalType == SignalTypeComplexity {
		return complexityNameDeclared(names, name)
	}
	_, exists := names[name]
	return exists
}

func complexityNameDeclared(names map[string]struct{}, name string) bool {
	baseName := name
	if idx := strings.Index(name, ":"); idx > 0 {
		baseName = name[:idx]
	}
	_, exists := names[baseName]
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
	if mapping.Method != "threshold_bands" {
		return fmt.Errorf("routing.projections.mappings[%q]: unsupported method %q (supported: threshold_bands)", mapping.Name, mapping.Method)
	}
	if len(mapping.Outputs) == 0 {
		return fmt.Errorf("routing.projections.mappings[%q]: outputs cannot be empty", mapping.Name)
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

func collectKeywordRuleNames(rules []KeywordRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectEmbeddingRuleNames(rules []EmbeddingRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectDomainNames(categories []Category) map[string]struct{} {
	names := make(map[string]struct{}, len(categories))
	for _, category := range categories {
		names[category.Name] = struct{}{}
	}
	return names
}

func collectFactCheckRuleNames(rules []FactCheckRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectUserFeedbackRuleNames(rules []UserFeedbackRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectPreferenceRuleNames(rules []PreferenceRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectLanguageRuleNames(rules []LanguageRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectContextRuleNames(rules []ContextRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectStructureRuleNames(rules []StructureRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectComplexityRuleNames(rules []ComplexityRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectModalityRuleNames(rules []ModalityRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectRoleBindingNames(rules []RoleBinding) map[string]struct{} {
	names := make(map[string]struct{}, len(rules)*2)
	for _, rule := range rules {
		if rule.Name != "" {
			names[rule.Name] = struct{}{}
		}
		if rule.Role != "" {
			names[rule.Role] = struct{}{}
		}
	}
	return names
}

func collectJailbreakRuleNames(rules []JailbreakRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectPIIRuleNames(rules []PIIRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
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
