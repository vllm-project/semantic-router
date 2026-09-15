package config

// Per-family rule-name collectors used by the projection validator to check
// that every projection input references a declared signal rule. Each new
// signal family adds one collector here.

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

func collectReaskRuleNames(rules []ReaskRule) map[string]struct{} {
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

func collectKBRuleNames(rules []KBSignalRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectConversationRuleNames(rules []ConversationRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectEventRuleNames(rules []EventRule) map[string]struct{} {
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

func collectMetadataRuleNames(rules []MetadataRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectClassifierRuleNames(
	rules []ClassifierSignalRule,
) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}
