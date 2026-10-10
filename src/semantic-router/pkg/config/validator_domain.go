package config

import (
	"fmt"
	"math"
	"strings"
)

func validateDomainContracts(cfg *RouterConfig) error {
	declared := make(map[string]Category, len(cfg.Categories))
	for _, category := range cfg.Categories {
		if err := validateDeclaredDomain(category); err != nil {
			return err
		}
		declared[category.Name] = category
	}

	if err := validateSoftmaxDomainPartitions(cfg.Projections.Partitions, declared); err != nil {
		return err
	}

	for _, decision := range cfg.AllRoutingDecisions() {
		if err := validateDecisionDomainReferences(decision.Name, &decision.Rules, declared); err != nil {
			return err
		}
	}

	return nil
}

func validateDeclaredDomain(category Category) error {
	for _, ms := range category.ModelScores {
		if math.IsNaN(ms.Score) || math.IsInf(ms.Score, 0) {
			if ms.Model != "" {
				return fmt.Errorf(
					"routing.signals.domains[%q].model_scores for model %q: score must be finite, got %v",
					category.Name,
					ms.Model,
					ms.Score,
				)
			}
			return fmt.Errorf(
				"routing.signals.domains[%q].model_scores: score must be finite, got %v",
				category.Name,
				ms.Score,
			)
		}
		if ms.Score < 0.0 || ms.Score > 1.0 {
			if ms.Model != "" {
				return fmt.Errorf(
					"routing.signals.domains[%q].model_scores for model %q: score must be between 0.0 and 1.0, got %v",
					category.Name,
					ms.Model,
					ms.Score,
				)
			}
			return fmt.Errorf(
				"routing.signals.domains[%q].model_scores: score must be between 0.0 and 1.0, got %v",
				category.Name,
				ms.Score,
			)
		}
	}

	for _, value := range category.MMLUCategories {
		if IsSupportedRoutingDomainName(value) {
			continue
		}
		return fmt.Errorf(
			"routing.signals.domains[%q].mmlu_categories contains unsupported value %q; supported values: %s%s",
			category.Name,
			value,
			strings.Join(SupportedRoutingDomainNames(), ", "),
			formatDomainSuggestion(value),
		)
	}

	return nil
}

func validateSoftmaxDomainPartitions(groups []ProjectionPartition, declared map[string]Category) error {
	for _, group := range groups {
		if !shouldValidateSoftmaxDomainPartition(group, declared) {
			continue
		}
		for _, member := range group.Members {
			if err := validateSoftmaxDomainPartitionMember(group.Name, member, declared[member]); err != nil {
				return err
			}
		}
	}
	return nil
}

func shouldValidateSoftmaxDomainPartition(group ProjectionPartition, declared map[string]Category) bool {
	return group.Semantics == "softmax_exclusive" && allMembersDeclaredCategories(group.Members, declared)
}

func validateSoftmaxDomainPartitionMember(groupName, member string, category Category) error {
	if len(category.MMLUCategories) == 0 {
		if IsSupportedRoutingDomainName(category.Name) {
			return nil
		}
		return fmt.Errorf(
			"routing.projections.partitions[%q]: domain member %q must use a supported routing domain name (%s) or declare mmlu_categories explicitly%s",
			groupName,
			member,
			strings.Join(SupportedRoutingDomainNames(), ", "),
			formatDomainSuggestion(member),
		)
	}

	for _, value := range category.MMLUCategories {
		if IsSupportedRoutingDomainName(value) {
			continue
		}
		return fmt.Errorf(
			"routing.projections.partitions[%q]: domain member %q has unsupported mmlu_categories value %q; supported values: %s%s",
			groupName,
			member,
			value,
			strings.Join(SupportedRoutingDomainNames(), ", "),
			formatDomainSuggestion(value),
		)
	}
	return nil
}

func allMembersDeclaredCategories(members []string, declared map[string]Category) bool {
	if len(members) == 0 {
		return false
	}
	for _, member := range members {
		if _, ok := declared[member]; !ok {
			return false
		}
	}
	return true
}

func validateDecisionDomainReferences(decisionName string, node *RuleNode, declared map[string]Category) error {
	if node == nil {
		return nil
	}

	if node.Type == SignalTypeDomain {
		if _, ok := declared[node.Name]; !ok {
			return fmt.Errorf(
				"decision %q references domain %q, but no routing.signals.domains entry declares that name",
				decisionName,
				node.Name,
			)
		}
	}

	for i := range node.Conditions {
		if err := validateDecisionDomainReferences(decisionName, &node.Conditions[i], declared); err != nil {
			return err
		}
	}

	return nil
}

func formatDomainSuggestion(value string) string {
	suggestion := SuggestSupportedRoutingDomainName(value)
	if suggestion == "" || suggestion == value {
		return ""
	}
	return fmt.Sprintf("; did you mean %q?", suggestion)
}
