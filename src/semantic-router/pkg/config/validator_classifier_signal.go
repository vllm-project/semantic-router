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

// validateGlobalClassifierRuntimeContracts validates each recipe independently.
// Native instances belong to a generation and may differ across recipes.
func validateGlobalClassifierRuntimeContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for i := range cfg.Recipes {
		if err := validateClassifierSignalRules(cfg.ConfigForRecipe(&cfg.Recipes[i])); err != nil {
			return err
		}
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
	for i, rule := range cfg.ClassifierRules {
		if err := validateClassifierSignalIdentity(rule, i, seen); err != nil {
			return err
		}
		if err := validateClassifierLabels(rule); err != nil {
			return err
		}
		if decl, exists := cfg.ModelBindings["classifier."+rule.Name]; exists {
			deployment, found := cfg.ModelDeployments[decl.Deployment]
			if !found {
				return fmt.Errorf("classifier %q: unknown deployment %q", rule.Name, decl.Deployment)
			}
			if err := validateGenericModelBinding(cfg, &rule, decl, deployment.WithDefaults()); err != nil {
				return err
			}
			continue
		}
		switch rule.Type {
		case ClassifierSignalTypeLocal:
			if err := validateLocalClassifierSignal(rule); err != nil {
				return err
			}
		case ClassifierSignalTypeLLM:
			if err := validateLLMClassifierSignal(cfg, rule); err != nil {
				return err
			}
		case ClassifierSignalTypeSequenceClassifier:
			if err := validateSequenceClassifierSignal(cfg, rule); err != nil {
				return err
			}
		default:
			return fmt.Errorf(
				"routing.signals.classifiers[%q]: unsupported type %q (supported: %s, %s, %s)",
				rule.Name,
				rule.Type,
				ClassifierSignalTypeLocal,
				ClassifierSignalTypeLLM,
				ClassifierSignalTypeSequenceClassifier,
			)
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
	if len(rule.Labels) < 2 {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: local classifiers require at least two labels",
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

// validateSequenceClassifierSignal checks a rule against the http_classify
// contract, where the declared label list is the response contract and no
// prompt instructions apply.
func validateSequenceClassifierSignal(cfg *RouterConfig, rule ClassifierSignalRule) error {
	if strings.TrimSpace(rule.Model) == "" {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: sequence_classifier classifiers require model",
			rule.Name,
		)
	}
	if rule.ModelPath != "" || rule.UseCPU || rule.Instructions != "" {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: sequence_classifier classifiers do not accept model_path, use_cpu or instructions",
			rule.Name,
		)
	}
	// The connector aligns a response against the full declared label set, so
	// a single label has no distribution to report.
	if len(rule.Labels) < 2 {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: sequence_classifier classifiers require at least two labels",
			rule.Name,
		)
	}
	external := cfg.FindExternalModelByName(rule.Model)
	if external == nil {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: sequence_classifier model %q is not declared in global.model_catalog.external[].name",
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
	return validateClassifierExternalEndpoint(rule, external)
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
	parserType := strings.ToLower(strings.TrimSpace(external.ParserType))
	if parserType != "" && parserType != "json" {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: external model %q parser_type must be json for llm classifiers",
			rule.Name,
			rule.Model,
		)
	}
	return validateClassifierExternalEndpoint(rule, external)
}

func validateClassifierExternalEndpoint(
	rule ClassifierSignalRule,
	external *ExternalModelConfig,
) error {
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

// ValidateLocalClassifierReload validates candidate declarations. Model changes
// are prepared in independent generation resources before atomic activation.
func ValidateLocalClassifierReload(_ *RouterConfig, next *RouterConfig) error {
	if next == nil {
		return nil
	}
	if len(next.Recipes) > 0 {
		return validateGlobalClassifierRuntimeContracts(next)
	}
	return validateClassifierSignalRules(next)
}
