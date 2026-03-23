package config

import (
	"fmt"
	"strings"
)

var (
	supportedStructureFeatureTypes = map[string]struct{}{
		"exists":   {},
		"count":    {},
		"density":  {},
		"sequence": {},
	}
	supportedStructureSourceTypes = map[string]struct{}{
		"regex":       {},
		"keyword_set": {},
		"sequence":    {},
	}
)

func validateStructureContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.StructureRules))
	for _, rule := range cfg.StructureRules {
		if strings.TrimSpace(rule.Name) == "" {
			return fmt.Errorf("routing.signals.structure: name cannot be empty")
		}
		if _, exists := seen[rule.Name]; exists {
			return fmt.Errorf("routing.signals.structure[%q]: duplicate rule name", rule.Name)
		}
		seen[rule.Name] = struct{}{}
		if err := ValidateStructureRuleContract(rule); err != nil {
			return err
		}
	}
	return nil
}

// ValidateStructureRuleContract validates one structure rule against the
// repository-supported structure signal contract.
func ValidateStructureRuleContract(rule StructureRule) error {
	featureType := strings.ToLower(strings.TrimSpace(rule.Feature.Type))
	if _, ok := supportedStructureFeatureTypes[featureType]; !ok {
		return fmt.Errorf(
			"routing.signals.structure[%q]: unsupported feature.type %q (supported: exists, count, density, sequence)",
			rule.Name,
			rule.Feature.Type,
		)
	}
	sourceType := strings.ToLower(strings.TrimSpace(rule.Feature.Source.Type))
	if _, ok := supportedStructureSourceTypes[sourceType]; !ok {
		return fmt.Errorf(
			"routing.signals.structure[%q]: unsupported feature.source.type %q (supported: regex, keyword_set, sequence)",
			rule.Name,
			rule.Feature.Source.Type,
		)
	}
	if err := validateStructureSource(rule.Name, featureType, rule.Feature.Source); err != nil {
		return err
	}
	if err := validateStructurePredicate(rule.Name, featureType, rule.Predicate); err != nil {
		return err
	}
	return nil
}

func validateStructureSource(ruleName string, featureType string, source StructureSource) error {
	sourceType := strings.ToLower(strings.TrimSpace(source.Type))
	if err := validateStructureSourceFields(ruleName, sourceType, source); err != nil {
		return err
	}
	return validateStructureFeatureSourceCompatibility(ruleName, featureType, sourceType)
}

func validateStructureSourceFields(ruleName string, sourceType string, source StructureSource) error {
	switch sourceType {
	case "regex":
		if strings.TrimSpace(source.Pattern) == "" {
			return fmt.Errorf("routing.signals.structure[%q]: regex source requires pattern", ruleName)
		}
	case "keyword_set":
		if len(source.Keywords) == 0 {
			return fmt.Errorf("routing.signals.structure[%q]: keyword_set source requires keywords", ruleName)
		}
	case "sequence":
		return validateStructureSequenceSource(ruleName, source.Sequences)
	}
	return nil
}

func validateStructureSequenceSource(ruleName string, sequences [][]string) error {
	if len(sequences) == 0 {
		return fmt.Errorf("routing.signals.structure[%q]: sequence source requires sequences", ruleName)
	}
	for _, sequence := range sequences {
		if len(sequence) < 2 {
			return fmt.Errorf("routing.signals.structure[%q]: sequence entries must contain at least 2 markers", ruleName)
		}
	}
	return nil
}

func validateStructureFeatureSourceCompatibility(ruleName string, featureType string, sourceType string) error {
	if featureType == "sequence" && sourceType != "sequence" {
		return fmt.Errorf(
			"routing.signals.structure[%q]: feature.type=sequence requires feature.source.type=sequence",
			ruleName,
		)
	}
	return nil
}

func validateStructurePredicate(
	ruleName string,
	featureType string,
	predicate *NumericPredicate,
) error {
	if predicate == nil {
		return nil
	}
	count := structurePredicateComparatorCount(predicate)
	if count == 0 {
		return fmt.Errorf("routing.signals.structure[%q]: predicate must set at least one comparator", ruleName)
	}
	if err := validateStructurePredicateBounds(ruleName, predicate); err != nil {
		return err
	}
	if featureType == "exists" && count > 0 {
		return fmt.Errorf("routing.signals.structure[%q]: feature.type=exists does not accept predicate", ruleName)
	}
	return nil
}

func structurePredicateComparatorCount(predicate *NumericPredicate) int {
	count := 0
	if predicate.GT != nil {
		count++
	}
	if predicate.GTE != nil {
		count++
	}
	if predicate.LT != nil {
		count++
	}
	if predicate.LTE != nil {
		count++
	}
	return count
}

func validateStructurePredicateBounds(ruleName string, predicate *NumericPredicate) error {
	if predicate.GT != nil && predicate.GTE != nil {
		return fmt.Errorf("routing.signals.structure[%q]: predicate cannot set both gt and gte", ruleName)
	}
	if predicate.LT != nil && predicate.LTE != nil {
		return fmt.Errorf("routing.signals.structure[%q]: predicate cannot set both lt and lte", ruleName)
	}
	return nil
}
