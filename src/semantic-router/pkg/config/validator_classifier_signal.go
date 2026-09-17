package config

import (
	"fmt"
	"strings"
)

func validateClassifierSignalContracts(cfg *RouterConfig) error {
	if err := validateExternalModelNames(cfg.ExternalModels); err != nil {
		return err
	}
	if err := validateExternalModelReasoningContracts(cfg); err != nil {
		return err
	}
	return validateClassifierSignalRules(cfg)
}

func validateExternalModelReasoningContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for i := range cfg.ExternalModels {
		external := &cfg.ExternalModels[i]
		path := fmt.Sprintf("global.model_catalog.external[%d]", i)
		if external.Name != "" {
			path = fmt.Sprintf("global.model_catalog.external[%q]", external.Name)
		}
		if err := validateExternalModelReasoningSyntax(external, path); err != nil {
			return err
		}
	}
	return nil
}

func validateExternalModelReasoningSyntax(
	external *ExternalModelConfig,
	path string,
) error {
	if external == nil || external.Reasoning == nil {
		return nil
	}
	reasoning := external.Reasoning
	if strings.TrimSpace(reasoning.Family) == "" {
		return fmt.Errorf("%s.reasoning.family is required", path)
	}
	if strings.TrimSpace(reasoning.Family) != reasoning.Family {
		return fmt.Errorf("%s.reasoning.family must not contain surrounding whitespace", path)
	}
	if reasoning.UseReasoning == nil {
		return fmt.Errorf("%s.reasoning.use_reasoning is required", path)
	}
	if strings.TrimSpace(reasoning.ReasoningEffort) != reasoning.ReasoningEffort {
		return fmt.Errorf("%s.reasoning.reasoning_effort must not contain surrounding whitespace", path)
	}
	return nil
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
	if rule.Model != "" || rule.Instructions != "" || rule.DisableRationale {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: local classifiers do not accept model, instructions or disable_rationale",
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
	return validateLLMClassifierExternalDependency(cfg, rule, external)
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
	if rule.ModelPath != "" || rule.UseCPU || rule.Instructions != "" || rule.DisableRationale {
		return fmt.Errorf(
			"routing.signals.classifiers[%q]: sequence_classifier classifiers do not accept model_path, use_cpu, instructions or disable_rationale",
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
	cfg *RouterConfig,
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
	if err := validateLLMClassifierReasoningControl(cfg, rule, external); err != nil {
		return err
	}
	return validateClassifierExternalEndpoint(rule, external)
}

func validateLLMClassifierReasoningControl(
	cfg *RouterConfig,
	rule ClassifierSignalRule,
	external *ExternalModelConfig,
) error {
	if external.Reasoning == nil {
		return nil
	}
	path := fmt.Sprintf(
		"routing.signals.classifiers[%q]: external model %q reasoning",
		rule.Name,
		rule.Model,
	)
	if strings.ToLower(strings.TrimSpace(external.Provider)) != "vllm" {
		return fmt.Errorf("%s requires llm_provider %q", path, "vllm")
	}
	reasoning := external.Reasoning
	family, exists := cfg.ReasoningFamilies[reasoning.Family]
	if !exists {
		return fmt.Errorf("%s family %q is not configured", path, reasoning.Family)
	}
	if reasoning.UseReasoning == nil {
		return fmt.Errorf("%s use_reasoning is required", path)
	}
	effort := reasoning.ReasoningEffort
	if !*reasoning.UseReasoning {
		if !externalReasoningFamilyCanDisable(&family) {
			return fmt.Errorf("%s cannot disable always-on family %q", path, reasoning.Family)
		}
		if effort != "" {
			return fmt.Errorf("%s reasoning_effort cannot be set while reasoning is disabled", path)
		}
		return nil
	}
	if effort == "" {
		return nil
	}
	if family.Type == ReasoningFamilyTypeReasoningMode || family.Type == ReasoningFamilyTypeChatTemplateKwargs {
		return fmt.Errorf("%s reasoning_effort cannot be used with mode-only family %q", path, reasoning.Family)
	}
	if !reasoningFamilyAllowsLevel(&family, effort) {
		return fmt.Errorf("%s reasoning_effort %q is not supported by family %q", path, effort, reasoning.Family)
	}
	return nil
}

func externalReasoningFamilyCanDisable(family *ReasoningFamilyConfig) bool {
	if family != nil && family.Type == ReasoningFamilyTypeTopLevelReasoningEffort {
		return family.Disabled != ""
	}
	return reasoningFamilyCanDisableConfig(family)
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
