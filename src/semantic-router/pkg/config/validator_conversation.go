package config

import "fmt"

var validConversationFeatureTypes = map[string]bool{
	"count":  true,
	"exists": true,
}

var validConversationSourceTypes = map[string]bool{
	"message":              true,
	"tool_definition":      true,
	"assistant_tool_call":  true,
	"assistant_tool_cycle": true,
}

var validConversationSourceRoles = map[string]bool{
	"user":      true,
	"assistant": true,
	"system":    true,
	"developer": true,
	"tool":      true,
	"non_user":  true,
}

func validateConversationContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.ConversationRules))
	for i, rule := range cfg.ConversationRules {
		if err := ValidateConversationRuleContract(rule); err != nil {
			return fmt.Errorf("conversation rule [%d] %q: %w", i, rule.Name, err)
		}
		if _, exists := seen[rule.Name]; exists {
			return fmt.Errorf("conversation rule [%d]: duplicate name %q", i, rule.Name)
		}
		seen[rule.Name] = struct{}{}
	}
	return nil
}

func ValidateConversationRuleContract(rule ConversationRule) error {
	if rule.Name == "" {
		return fmt.Errorf("name is required")
	}
	if !validConversationFeatureTypes[rule.Feature.Type] {
		return fmt.Errorf("unsupported feature.type %q; valid types: count, exists", rule.Feature.Type)
	}
	if !validConversationSourceTypes[rule.Feature.Source.Type] {
		return fmt.Errorf("unsupported feature.source.type %q; valid types: message, tool_definition, assistant_tool_call, assistant_tool_cycle", rule.Feature.Source.Type)
	}
	if rule.Feature.Source.Role != "" {
		if rule.Feature.Source.Type != "message" {
			return fmt.Errorf("source.role is only valid when source.type is \"message\", got source.type %q", rule.Feature.Source.Type)
		}
		if !validConversationSourceRoles[rule.Feature.Source.Role] {
			return fmt.Errorf("unsupported source.role %q; valid roles: user, assistant, system, developer, tool, non_user", rule.Feature.Source.Role)
		}
	}
	if err := validateConversationPredicate(rule); err != nil {
		return err
	}
	return nil
}

func validateConversationPredicate(rule ConversationRule) error {
	if rule.Predicate == nil {
		return nil
	}
	if rule.Feature.Type == "exists" {
		return fmt.Errorf("feature.type \"exists\" does not accept a predicate")
	}
	if rule.Predicate.GT != nil && rule.Predicate.GTE != nil {
		return fmt.Errorf("predicate cannot set both gt and gte")
	}
	if rule.Predicate.LT != nil && rule.Predicate.LTE != nil {
		return fmt.Errorf("predicate cannot set both lt and lte")
	}
	return nil
}
