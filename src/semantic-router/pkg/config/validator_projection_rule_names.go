package config

// Per-family rule-name collectors used by the projection validator to check
// that every projection input references a declared heuristic or systemone
// rule; application signal inputs are checked against their compiled rules.

func collectKeywordRuleNames(rules []KeywordRule) map[string]struct{} {
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

// collectRoleBindingRoles returns the roles the bindings grant: the authz
// signal emits a matched binding's role, never its name.
func collectRoleBindingRoles(rules []RoleBinding) map[string]struct{} {
	roles := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		if rule.Role != "" {
			roles[rule.Role] = struct{}{}
		}
	}
	return roles
}

func collectConversationRuleNames(rules []ConversationRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectInputModalityRuleNames(rules []InputModalityRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectSystemOneRuleNames(rules []SystemOneRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectMetadataRuleNames(rules []MetadataRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}
