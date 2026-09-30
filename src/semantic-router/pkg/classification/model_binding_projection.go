package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (m *classifierModelRuntime) projectBindings() error {
	projected, err := config.ProjectRecipeModelBindings(m.cfg, m.plan, m.recipe)
	if err == nil {
		m.cfg = projected
	}
	return err
}

func (m *classifierModelRuntime) mappings(category *CategoryMapping, pii *PIIMapping, jailbreak *JailbreakMapping) (*CategoryMapping, *PIIMapping, *JailbreakMapping, error) {
	var err error
	if spec, ok := m.plan.Lookup(m.recipe, "domain_classifier"); m.cfg.NeedsCategoryMappingForRouting() && (category == nil || ok && spec.Binding.MappingPath != "") {
		category, err = LoadCategoryMapping(m.cfg.CategoryMappingPath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
	}
	if spec, ok := m.plan.Lookup(m.recipe, "pii_classifier"); m.cfg.NeedsPIIMappingForRouting() && (pii == nil || ok && spec.Binding.MappingPath != "") {
		pii, err = LoadPIIMapping(m.cfg.PIIMappingPath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
	}
	if spec, ok := m.plan.Lookup(m.recipe, "prompt_guard"); m.cfg.NeedsJailbreakMappingForRouting() && (jailbreak == nil || ok && spec.Binding.MappingPath != "") {
		jailbreak, err = LoadJailbreakMapping(m.cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
	}
	return category, pii, jailbreak, nil
}
