package config

import (
	"fmt"
	"strings"
)

func validateClassifierSignalContracts(cfg *RouterConfig) error {
	if err := validateExternalModelNames(cfg.ExternalModels); err != nil {
		return err
	}
	return validateClassifierSignalRules(cfg)
}

func validateExternalModelNames(models []ExternalModelConfig) error {
	externalNames := make(map[string]struct{})
	for _, external := range models {
		if external.Name == "" {
			continue
		}
		if _, exists := externalNames[external.Name]; exists {
			return fmt.Errorf("global.model_catalog.external: duplicate name %q", external.Name)
		}
		externalNames[external.Name] = struct{}{}
	}
	return nil
}

func validateClassifierSignalRules(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.ClassifierRules))
	localClassifierCount := 0
	for i, rule := range cfg.ClassifierRules {
		if err := validateClassifierSignalIdentity(rule, i, seen); err != nil {
			return err
		}
		if err := validateClassifierLabels(rule); err != nil {
			return err
		}
		if rule.Type == "local" {
			if err := validateLocalClassifierSignal(rule); err != nil {
				return err
			}
			localClassifierCount++
			if localClassifierCount > 1 {
				return fmt.Errorf(
					"routing.signals.classifiers: only one local classifier is supported; use llm classifiers or a specialized signal for additional models",
				)
			}
			continue
		}
		if rule.Type != "llm" {
			return fmt.Errorf(
				"routing.signals.classifiers[%q]: unsupported type %q (supported: local, llm)",
				rule.Name,
				rule.Type,
			)
		}
		if err := validateLLMClassifierSignal(cfg, rule); err != nil {
			return err
		}
	}
	return nil
}

func validateClassifierSignalIdentity(
	rule ClassifierSignalRule,
	index int,
	seen map[string]struct{},
) error {
	if strings.TrimSpace(rule.Name) == "" {
		return fmt.Errorf("routing.signals.classifiers[%d]: name is required", index)
	}
	if _, exists := seen[rule.Name]; exists {
		return fmt.Errorf("routing.signals.classifiers[%d]: duplicate name %q", index, rule.Name)
	}
	seen[rule.Name] = struct{}{}
	return nil
}

func validateLocalClassifierSignal(rule ClassifierSignalRule) error {
	if strings.TrimSpace(rule.ModelPath) == "" {
		return fmt.Errorf("routing.signals.classifiers[%q]: local classifiers require model_path", rule.Name)
	}
	if rule.Model != "" || rule.Instructions != "" {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: local classifiers do not accept model or instructions",
			rule.Name,
		)
	}
	if len(rule.Labels) != 2 {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: local classifiers require exactly two labels",
			rule.Name,
		)
	}
	return nil
}

func validateClassifierLabels(rule ClassifierSignalRule) error {
	if len(rule.Labels) == 0 {
		return fmt.Errorf("routing.signals.classifiers[%q]: labels cannot be empty", rule.Name)
	}
	labels := make(map[string]struct{}, len(rule.Labels))
	for _, label := range rule.Labels {
		if strings.TrimSpace(label) == "" {
			return fmt.Errorf("routing.signals.classifiers[%q]: labels cannot contain empty values", rule.Name)
		}
		if _, exists := labels[label]; exists {
			return fmt.Errorf("routing.signals.classifiers[%q]: duplicate label %q", rule.Name, label)
		}
		labels[label] = struct{}{}
	}
	return nil
}

func validateLLMClassifierSignal(cfg *RouterConfig, rule ClassifierSignalRule) error {
	if strings.TrimSpace(rule.Model) == "" {
		return fmt.Errorf("routing.signals.classifiers[%q]: llm classifiers require model", rule.Name)
	}
	if rule.ModelPath != "" || rule.UseCPU {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: llm classifiers do not accept model_path or use_cpu",
			rule.Name,
		)
	}
	if strings.TrimSpace(rule.Instructions) == "" {
		return fmt.Errorf("routing.signals.classifiers[%q]: llm classifiers require instructions", rule.Name)
	}
	external := cfg.FindExternalModelByName(rule.Model)
	if external == nil {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: llm model %q is not declared in global.model_catalog.external[].name",
			rule.Name,
			rule.Model,
		)
	}
	if external.ModelRole != ModelRoleClassification {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: external model %q must use model_role %q",
			rule.Name,
			rule.Model,
			ModelRoleClassification,
		)
	}
	return nil
}
