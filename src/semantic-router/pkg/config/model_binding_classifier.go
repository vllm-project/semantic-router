package config

import "fmt"

// Generic rules retain their declared labels and score policy. A deployment
// replaces execution selectors, while an LLM rule retains its scored extraction
// prompt rather than becoming a sequence classifier or categorical chat guard.
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
	switch rule.Type {
	case ClassifierSignalTypeLocal, ClassifierSignalTypeSequenceClassifier:
		if len(rule.Labels) < 2 || rule.Instructions != "" {
			return fmt.Errorf("sequence classifier bindings require at least two labels and no instructions")
		}
		if deployment.Provider == "http" && decl.Adapter != RemoteClassifierProtocolHTTPClassify {
			return fmt.Errorf("sequence classifier binding requires http_classify adapter")
		}
	case ClassifierSignalTypeLLM:
		if deployment.Provider != "http" || decl.Adapter != RemoteClassifierProtocolHTTPChat {
			return fmt.Errorf("llm classifier binding requires HTTP http_chat scored extraction")
		}
	default:
		return fmt.Errorf("unsupported generic classifier type %q", rule.Type)
	}
	projected := projectGenericClassifierRule(*rule, deployment)
	if deployment.Provider != "http" {
		return nil
	}
	if projected.Type == ClassifierSignalTypeLLM {
		return validateLLMClassifierSignal(cfg, projected)
	}
	return validateSequenceClassifierSignal(cfg, projected)
}

func projectGenericClassifierRule(rule ClassifierSignalRule, deployment ModelDeployment) ClassifierSignalRule {
	rule.Model, rule.ModelPath, rule.UseCPU = "", "", false
	if deployment.Provider == "http" {
		rule.Model = deployment.ExternalModel
		if rule.Type != ClassifierSignalTypeLLM {
			rule.Type = ClassifierSignalTypeSequenceClassifier
		}
	} else {
		rule.Type = ClassifierSignalTypeLocal
		rule.ModelPath = ResolveModelPath(deployment.Artifact)
		rule.UseCPU = deployment.Device == "cpu"
	}
	return rule
}
