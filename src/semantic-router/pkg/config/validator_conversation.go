package config

import (
	"fmt"
	"regexp"
)

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
	"last_tool_call":       true,
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
		return fmt.Errorf("unsupported feature.source.type %q; valid types: message, tool_definition, tool_choice_required, tool_choice_none, assistant_tool_call, assistant_tool_cycle, active_tool_loop, flow_tool_state, last_tool_call", rule.Feature.Source.Type)
	}
	if err := validateLastToolCallSource(rule.Feature.Source); err != nil {
		return err
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
		if rule.Feature.Type == "count" {
			return fmt.Errorf("feature.type \"count\" requires a predicate, such as gte: 1")
		}
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

func validateLastToolCallSource(source ConversationSource) error {
	if source.Type != "last_tool_call" {
		if len(source.Tools) > 0 || len(source.Calls) > 0 || source.Match != "" {
			return fmt.Errorf("source.tools, source.calls and source.match are only valid when source.type is \"last_tool_call\", got source.type %q", source.Type)
		}
		return nil
	}
	if len(source.Tools) > 0 && len(source.Calls) > 0 {
		return fmt.Errorf("source.tools and source.calls cannot both be set; list the names in a calls entry")
	}
	if source.Match != "" && source.Match != "match" && source.Match != "no_match" {
		return fmt.Errorf("unsupported source.match %q; valid values: match, no_match", source.Match)
	}
	for i, call := range source.Calls {
		if call.Argument != "" && call.Pattern == "" {
			return fmt.Errorf("source.calls[%d].argument needs a pattern", i)
		}
		if _, err := regexp.Compile(call.Pattern); err != nil {
			return fmt.Errorf("source.calls[%d].pattern: %w", i, err)
		}
	}
	return nil
}
