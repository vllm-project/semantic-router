package config

import (
	"fmt"
	"slices"
	"strings"
)

func validateClassifierSignalContracts(cfg *RouterConfig) error {
	if err := validateExternalModelNames(cfg.ExternalModels); err != nil {
		return err
	}
	return validateClassifierSignalRules(cfg)
}

// validateGlobalClassifierRuntimeContracts enforces the process-global native
// classifier seam while allowing each recipe to declare its own local rule
// name. Every local rule must use the same model, labels, and device.
func validateGlobalClassifierRuntimeContracts(cfg *RouterConfig) error {
	var (
		signature *localClassifierSignature
		owner     RecipeName
	)
	for _, ref := range localClassifierRuleRefs(cfg) {
		next := newLocalClassifierSignature(ref.Rule)
		if signature == nil {
			signature = &next
			owner = ref.Recipe
			continue
		}
		if signature.Equal(next) {
			continue
		}
		return fmt.Errorf(
			"routing recipes %q and %q declare incompatible local classifiers; "+
				"the process-global local classifier requires identical model_path, labels, and use_cpu",
			owner,
			ref.Recipe,
		)
	}
	return nil
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
	trimmedName := strings.TrimSpace(rule.Name)
	if trimmedName == "" {
		return fmt.Errorf("routing.signals.classifiers[%d]: name is required", index)
	}
	if trimmedName != rule.Name {
		return fmt.Errorf(
			"routing.signals.classifiers[%d]: name must not contain surrounding whitespace",
			index,
		)
	}
	if strings.Contains(rule.Name, ":") {
		return fmt.Errorf(
			"routing.signals.classifiers[%d]: name cannot contain ':'",
			index,
		)
	}
	normalizedName := strings.ToLower(rule.Name)
	if _, exists := seen[normalizedName]; exists {
		return fmt.Errorf("routing.signals.classifiers[%d]: duplicate name %q", index, rule.Name)
	}
	seen[normalizedName] = struct{}{}
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
		trimmedLabel := strings.TrimSpace(label)
		if trimmedLabel == "" {
			return fmt.Errorf("routing.signals.classifiers[%q]: labels cannot contain empty values", rule.Name)
		}
		if trimmedLabel != label {
			return fmt.Errorf(
				"routing.signals.classifiers[%q]: label %q must not contain surrounding whitespace",
				rule.Name,
				label,
			)
		}
		if strings.Contains(label, ":") {
			return fmt.Errorf(
				"routing.signals.classifiers[%q]: label %q cannot contain ':'",
				rule.Name,
				label,
			)
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
	return validateLLMClassifierExternalDependency(rule, external)
}

func validateLLMClassifierExternalDependency(
	rule ClassifierSignalRule,
	external *ExternalModelConfig,
) error {
	if strings.TrimSpace(external.ModelName) == "" {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: external model %q requires llm_model_name",
			rule.Name,
			rule.Model,
		)
	}
	if strings.TrimSpace(external.ModelEndpoint.Address) == "" ||
		external.ModelEndpoint.Port < 1 ||
		external.ModelEndpoint.Port > 65535 {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: external model %q requires a valid llm_endpoint address and port",
			rule.Name,
			rule.Model,
		)
	}
	protocol := strings.ToLower(
		strings.TrimSpace(external.ModelEndpoint.Protocol),
	)
	if protocol != "" && protocol != "http" && protocol != "https" {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: external model %q llm_endpoint.protocol must be http or https",
			rule.Name,
			rule.Model,
		)
	}
	return nil
}

// ValidateLocalClassifierReload rejects mutations that cannot be applied
// atomically by the process-global native classifier binding.
func ValidateLocalClassifierReload(current *RouterConfig, next *RouterConfig) error {
	currentSignature := firstLocalClassifierSignature(current)
	nextSignature := firstLocalClassifierSignature(next)
	if currentSignature == nil && nextSignature == nil {
		return nil
	}
	if currentSignature == nil ||
		nextSignature == nil ||
		!currentSignature.Equal(*nextSignature) {
		return fmt.Errorf(
			"local classifier model, labels, and device cannot change during hot reload; restart the router",
		)
	}
	return nil
}

type localClassifierRuleRef struct {
	Recipe RecipeName
	Rule   ClassifierSignalRule
}

type localClassifierSignature struct {
	ModelPath string
	UseCPU    bool
	Labels    []string
}

func newLocalClassifierSignature(
	rule ClassifierSignalRule,
) localClassifierSignature {
	return localClassifierSignature{
		ModelPath: rule.ModelPath,
		UseCPU:    rule.UseCPU,
		Labels:    append([]string(nil), rule.Labels...),
	}
}

func (s localClassifierSignature) Equal(other localClassifierSignature) bool {
	return s.ModelPath == other.ModelPath &&
		s.UseCPU == other.UseCPU &&
		slices.Equal(s.Labels, other.Labels)
}

func firstLocalClassifierSignature(
	cfg *RouterConfig,
) *localClassifierSignature {
	refs := localClassifierRuleRefs(cfg)
	if len(refs) == 0 {
		return nil
	}
	signature := newLocalClassifierSignature(refs[0].Rule)
	return &signature
}

func localClassifierRuleRefs(cfg *RouterConfig) []localClassifierRuleRef {
	if cfg == nil {
		return nil
	}
	refs := make([]localClassifierRuleRef, 0)
	if len(cfg.Recipes) > 0 {
		for _, recipe := range cfg.Recipes {
			for _, rule := range recipe.Profile.Signals.ClassifierRules {
				if rule.Type == "local" {
					refs = append(refs, localClassifierRuleRef{
						Recipe: recipe.Name,
						Rule:   rule,
					})
				}
			}
		}
		return refs
	}
	for _, rule := range cfg.ClassifierRules {
		if rule.Type == "local" {
			refs = append(refs, localClassifierRuleRef{
				Recipe: DefaultRecipeName,
				Rule:   rule,
			})
		}
	}
	return refs
}
