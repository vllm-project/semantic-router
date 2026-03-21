package config

import (
	"fmt"
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

	if err := validateSoftmaxDomainSignalGroups(cfg.SignalGroups, declared); err != nil {
		return err
	}

	for _, decision := range cfg.Decisions {
		if err := validateDecisionDomainReferences(decision.Name, &decision.Rules, declared); err != nil {
			return err
		}
	}

	return nil
}

func validateDeclaredDomain(category Category) error {
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

func validateSoftmaxDomainSignalGroups(groups []SignalGroup, declared map[string]Category) error {
	for _, group := range groups {
		if !shouldValidateSoftmaxDomainSignalGroup(group, declared) {
			continue
		}
		for _, member := range group.Members {
			if err := validateSoftmaxDomainSignalMember(group.Name, member, declared[member]); err != nil {
				return err
			}
		}
	}
	return nil
}

func shouldValidateSoftmaxDomainSignalGroup(group SignalGroup, declared map[string]Category) bool {
	return group.Semantics == "softmax_exclusive" && allMembersDeclaredCategories(group.Members, declared)
}

func validateSoftmaxDomainSignalMember(groupName, member string, category Category) error {
	if len(category.MMLUCategories) == 0 {
		if IsSupportedRoutingDomainName(category.Name) {
			return nil
		}
		return fmt.Errorf(
			"routing.signals.signal_groups[%q]: domain member %q must use a supported routing domain name (%s) or declare mmlu_categories explicitly%s",
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
			"routing.signals.signal_groups[%q]: domain member %q has unsupported mmlu_categories value %q; supported values: %s%s",
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
