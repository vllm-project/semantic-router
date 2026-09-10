package config

import "fmt"

var validConversationFeatureTypes = map[string]bool{
	"count":  true,
	"exists": true,
}

var validConversationSourceTypes = map[string]bool{
	"message":              true,
	"tool_definition":      true,
	"tool_choice_required": true,
	"tool_choice_none":     true,
	"assistant_tool_call":  true,
	"assistant_tool_cycle": true,
	"active_tool_loop":     true,
	"flow_tool_state":      true,
	"image_content":        true,
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
		return fmt.Errorf("unsupported feature.source.type %q; valid types: message, tool_definition, tool_choice_required, tool_choice_none, assistant_tool_call, assistant_tool_cycle, active_tool_loop, flow_tool_state, image_content", rule.Feature.Source.Type)
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
	if err := validateNumericPredicateContract(rule.Predicate); err != nil {
		return err
	}
	if rule.Feature.Type == "exists" {
		return fmt.Errorf("feature.type \"exists\" does not accept a predicate")
	}
	return nil
}
