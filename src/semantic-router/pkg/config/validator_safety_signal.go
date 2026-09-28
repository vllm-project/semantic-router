package config

import (
	"fmt"
	"math"
	"slices"
)

func validateSafetySignalContracts(cfg *RouterConfig) error {
	if err := validateExternalModelNames(cfg.ExternalModels); err != nil {
		return err
	}
	seen := make(map[string]struct{}, len(cfg.SafetyRules))
	for i, rule := range cfg.SafetyRules {
		classifier := ClassifierSignalRule{
			Name: rule.Name, Type: ClassifierSignalTypeSequenceClassifier,
			Model: rule.Model, Labels: rule.EffectiveLabels(),
		}
		if err := validateClassifierSignalIdentity(classifier, i, seen); err != nil {
			return fmt.Errorf("routing.signals.safety: %w", err)
		}
		if err := validateSafetyClassifier(cfg, "safety."+rule.Name, classifier, cfg.SafetyModels.Safety); err != nil {
			return fmt.Errorf("safety %q: %w", rule.Name, err)
		}
		if err := validateSafetyThreshold(rule.Threshold); err != nil {
			return fmt.Errorf("safety %q threshold: %w", rule.Name, err)
		}
		if err := validateSafetySelectedLabels(rule.EffectiveLabels(), rule.EffectiveUnsafeLabels()); err != nil {
			return fmt.Errorf("safety %q unsafe_labels: %w", rule.Name, err)
		}
		if len(rule.EffectiveUnsafeLabels()) == len(rule.EffectiveLabels()) {
			return fmt.Errorf("safety %q must leave at least one non-unsafe label", rule.Name)
		}
		if rule.Hazard == nil {
			continue
		}
		hazard := ClassifierSignalRule{
			Name: rule.Name, Type: ClassifierSignalTypeSequenceClassifier,
			Model: rule.Hazard.Model, Labels: rule.Hazard.Labels,
		}
		if err := validateSafetyClassifier(cfg, "safety."+rule.Name+".hazard", hazard, cfg.SafetyModels.Hazard); err != nil {
			return fmt.Errorf("safety %q hazard: %w", rule.Name, err)
		}
		if err := validateSafetyThreshold(rule.Hazard.Threshold); err != nil {
			return fmt.Errorf("safety %q hazard.threshold: %w", rule.Name, err)
		}
		if err := validateSafetySelectedLabels(rule.Hazard.Labels, rule.Hazard.Categories); err != nil {
			return fmt.Errorf("safety %q hazard.categories: %w", rule.Name, err)
		}
	}
	return nil
}

func validateSafetyClassifier(cfg *RouterConfig, consumer string, rule ClassifierSignalRule, local SequenceHeadModelConfig) error {
	if err := validateClassifierLabels(rule); err != nil {
		return err
	}
	if decl, bound := cfg.ModelBindings[consumer]; bound {
		deployment, exists := cfg.ModelDeployments[decl.Deployment]
		if !exists {
			return fmt.Errorf("unknown safety deployment %q", decl.Deployment)
		}
		deployment = deployment.WithDefaults()
		if err := deployment.validate(cfg); err != nil {
			return err
		}
		if err := validateTaskModelBinding(consumer, decl, deployment); err != nil {
			return err
		}
		if err := validateSafetyModelBinding(cfg.SafetyRules, consumer, decl, deployment); err != nil {
			return err
		}
		// Bindings replace only this recipe consumer. Keep inline label/policy
		// declarations and validate the actual endpoint or local input budget.
		if deployment.Provider == "http" {
			if rule.Model == "" && local.Window != nil {
				return fmt.Errorf("remote safety head cannot use local token windows")
			}
			rule.Model = deployment.ExternalModel
			return validateSequenceClassifierSignal(cfg, rule)
		}
		local.ModelID, local.MaxSequenceLength = deployment.Artifact, deployment.Input.MaxTokens
		if rule.Model != "" {
			local.Window = nil
		}
	} else if rule.Model != "" {
		return validateSequenceClassifierSignal(cfg, rule)
	}
	if local.ModelID == "" {
		return fmt.Errorf("built-in safety head requires a model_id or model_ref in global.model_catalog.modules.safety")
	}
	if local.InputLimit() <= 0 {
		return fmt.Errorf("max_sequence_length must be positive")
	}
	return local.ValidateWindow()
}

func validateSafetyThreshold(value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 || value > 1 {
		return fmt.Errorf("must be finite and greater than 0 and at most 1")
	}
	return nil
}

func validateSafetySelectedLabels(labels, selected []string) error {
	if len(selected) == 0 {
		return fmt.Errorf("must select at least one declared label")
	}
	seen := make(map[string]bool, len(selected))
	for _, label := range selected {
		if !slices.Contains(labels, label) {
			return fmt.Errorf("label %q is not declared by the model", label)
		}
		if seen[label] {
			return fmt.Errorf("duplicate label %q", label)
		}
		seen[label] = true
	}
	return nil
}
