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

// mappings loads each routing consumer's mapping file, or, for a local model
// without one, takes the labels the model serves.
func (m *classifierModelRuntime) mappings(category *CategoryMapping, pii *PIIMapping, jailbreak *JailbreakMapping) (*CategoryMapping, *PIIMapping, *JailbreakMapping, error) {
	var err error
	if spec, ok := m.plan.Lookup(m.recipe, "domain_classifier"); m.cfg.NeedsCategoryMappingForRouting() && (category == nil || ok && spec.Binding.MappingPath != "") {
		if path := m.cfg.CategoryMappingPath; path != "" {
			category, err = LoadCategoryMapping(path)
		} else {
			category, err = mappingFromServedLabels(m, "domain_classifier", m.cfg.CategoryModel.ModelID, config.RemoteClassifierContractLabelDistribution, m.cfg.CategoryModel.UseCPU, categoryMappingFromLabels)
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
	}
	if spec, ok := m.plan.Lookup(m.recipe, "pii_classifier"); m.cfg.NeedsPIIMappingForRouting() && (pii == nil || ok && spec.Binding.MappingPath != "") {
		if path := m.cfg.PIIMappingPath; path != "" {
			pii, err = LoadPIIMapping(path)
		} else {
			pii, err = mappingFromServedLabels(m, "pii_classifier", m.cfg.PIIModel.ModelID, config.RemoteClassifierContractTokenSpans, m.cfg.PIIModel.UseCPU, piiMappingFromLabels)
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
	}
	if spec, ok := m.plan.Lookup(m.recipe, "prompt_guard"); m.cfg.NeedsJailbreakMappingForRouting() && (jailbreak == nil || ok && spec.Binding.MappingPath != "") {
		if path := m.cfg.PromptGuard.JailbreakMappingPath; path != "" {
			jailbreak, err = LoadJailbreakMapping(path)
		} else {
			jailbreak, err = mappingFromServedLabels(m, "prompt_guard", m.cfg.PromptGuard.ModelID, config.RemoteClassifierContractLabelDistribution, m.cfg.PromptGuard.UseCPU, jailbreakMappingFromLabels)
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
	}
	return category, pii, jailbreak, nil
}

func mappingFromServedLabels[T any](m *classifierModelRuntime, consumer, artifact, contract string, useCPU bool, build func([]string) (*T, error)) (*T, error) {
	labels, err := m.servedLabels(consumer, artifact, contract, useCPU)
	if err != nil {
		return nil, err
	}
	return build(labels)
}
