package config

import (
	"cmp"
	"fmt"
	"strings"
)

// Generic rules retain their declared labels and score policy. A deployment
// replaces execution selectors, while an LLM rule retains its scored extraction
// prompt rather than becoming a sequence classifier or categorical chat guard.
// ClassifierRuleKind is how a classifier rule's deployment answers it: an
// openai_chat deployment as an LLM classifier, a classify deployment as a
// sequence classifier, a deployment a deprecated model_path declared as a
// local classify head, and any other (a decision model) as a choice question.
func ClassifierRuleKind(deployment ModelDeployment) string {
	switch deployment.Provider {
	case OpenAIChatProvider:
		return ClassifierSignalTypeLLM
	case ClassifyProvider, LegacyHTTPProvider:
		return ClassifierSignalTypeSequenceClassifier
	}
	return ClassifierSignalTypeLocal
}

func validateGenericModelBinding(cfg *RouterConfig, rule *ClassifierSignalRule, decl ModelBinding, deployment ModelDeployment) error {
	if rule == nil {
		return fmt.Errorf("generic classifier binding requires an existing rule in the same recipe")
	}
	if decl.MappingPath != "" {
		return fmt.Errorf("generic classifier labels define the mapping; mapping_path is not supported")
	}
	if err := validateClassifierLabels(*rule); err != nil {
		return err
	}
	kind := ClassifierRuleKind(deployment)
	legacyType := cmp.Or(rule.Type, rule.legacyType)
	if legacyType == ClassifierSignalTypeLLM && kind != ClassifierSignalTypeLLM {
		return fmt.Errorf("an llm classifier rule needs an openai_chat deployment")
	}
	if legacy := deployment.LegacyExternal(); legacy != nil {
		named := *rule
		named.Model = legacy.Name
		if err := validateExternalClassifierModel(legacy, legacy.Name, ModelRoleClassification, decl.Adapter); err != nil {
			return err
		}
		if err := validateClassifierExternalEndpoint(named, legacy); err != nil {
			return err
		}
	}
	if decl.Contract == RemoteClassifierContractLabelScores {
		// The model runtime serves the package's own operating point.
		if !deployment.IsModelRuntime() || kind == ClassifierSignalTypeLLM {
			return fmt.Errorf("independent scores require a model_runtime deployment of a sequence classifier")
		}
		if deployment.Input.MaxTokens <= 0 || deployment.Input.Overflow != "reject" {
			return fmt.Errorf("operating_point requires an explicit document token budget with reject overflow")
		}
	} else if decl.OperatingPoint != nil {
		return fmt.Errorf("operating_point requires label_scores.v1")
	}
	head := deployment.Translated() || decl.Head != "" || decl.Contract == RemoteClassifierContractLabelScores ||
		legacyType == ClassifierSignalTypeLocal || legacyType == ClassifierSignalTypeSequenceClassifier
	switch kind {
	case ClassifierSignalTypeLocal, ClassifierSignalTypeSequenceClassifier:
		if len(rule.Labels) < 2 {
			return fmt.Errorf("classifier bindings require at least two labels")
		}
		if (kind == ClassifierSignalTypeSequenceClassifier || head) && (rule.Instructions != "" || rule.DisableRationale) {
			return fmt.Errorf("sequence classifier bindings take no instructions or disable_rationale")
		}
		if remoteTaskProvider(deployment) && decl.Adapter != RemoteClassifierProtocolHTTPClassify {
			return fmt.Errorf("sequence classifier binding requires http_classify adapter")
		}
	case ClassifierSignalTypeLLM:
		if decl.Adapter != RemoteClassifierProtocolHTTPChat {
			return fmt.Errorf("llm classifier binding requires HTTP http_chat scored extraction")
		}
		if strings.TrimSpace(rule.Instructions) == "" {
			return fmt.Errorf("llm classifiers require instructions")
		}
	}
	return nil
}
