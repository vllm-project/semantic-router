package config

import (
	"fmt"
	"strings"
)

// DecisionModelBinding selects one declared resource. Model identity belongs
// only to deployments; this reference never guesses an artifact or alias.
type DecisionModelBinding struct {
	Deployment string `yaml:"deployment" json:"deployment" jsonschema:"required,minLength=1"`
}

const (
	DefaultDecisionDeployment = "primary"
	DefaultDecisionModel      = "Vela-2.0-0.3B"
	DecisionModelVela2_03B    = "Vela-2.0-0.3B"
	DecisionModelVela2_08B    = "Vela-2.0-0.8B"
	DecisionModelVela2_4B     = "Vela-2.0-4B"
	DecisionModelVela2_9B     = "Vela-2.0-9B"
	DecisionModelVela1        = "Vela-1.0"
	vela1HazardModel          = "models/Vela-1.0-Encoder-307M-Hazard"
	vela1ModalityModel        = "models/Vela-1.0-Encoder-307M-Modality"
)

// DecisionModelSpec is a projection of the selected deployment for existing
// module consumers. It is not a catalog of allowed model families.
type DecisionModelSpec struct {
	Name     string
	Model    string
	System   CanonicalSystemModels
	Modality string
}

func decisionModelSpec(name string, deployment ModelDeployment) DecisionModelSpec {
	model := deployment.ServedModel(name)
	if registered := GetModelByPath(model); registered != nil {
		model = registered.LocalPath
	} else {
		for _, registered := range DefaultModelRegistry {
			if registered.RepoID == model {
				model = registered.LocalPath
				break
			}
		}
	}
	return DecisionModelSpec{Name: name, Model: model, Modality: model, System: CanonicalSystemModels{
		Safety: model, Hazard: vela1HazardModel, PromptGuard: model, DomainClassifier: model,
		PIIClassifier: model, FactCheckClassifier: model, HallucinationDetector: model, FeedbackDetector: model,
	}}
}

// systemLine is one built-in module's line under global.model_catalog.system.
type systemLine struct {
	key   string
	value func(*CanonicalSystemModels) *string
}

var systemLines = []systemLine{
	{"safety", func(s *CanonicalSystemModels) *string { return &s.Safety }},
	{"hazard", func(s *CanonicalSystemModels) *string { return &s.Hazard }},
	{"prompt_guard", func(s *CanonicalSystemModels) *string { return &s.PromptGuard }},
	{"domain_classifier", func(s *CanonicalSystemModels) *string { return &s.DomainClassifier }},
	{"pii_classifier", func(s *CanonicalSystemModels) *string { return &s.PIIClassifier }},
	{"fact_check_classifier", func(s *CanonicalSystemModels) *string { return &s.FactCheckClassifier }},
	{"hallucination_detector", func(s *CanonicalSystemModels) *string { return &s.HallucinationDetector }},
	{"feedback_detector", func(s *CanonicalSystemModels) *string { return &s.FeedbackDetector }},
}

// applyDecisionModel resolves the default judgment resource, preserving every
// explicitly authored specialist module and deployment declaration.
func applyDecisionModel(resolved *CanonicalGlobal, raw *StructuredPayload) error {
	system := &resolved.ModelCatalog.System
	name := strings.TrimSpace(system.DecisionModel.Deployment)
	if name == "" || name != system.DecisionModel.Deployment {
		return fmt.Errorf("global.model_catalog.system.decision_model.deployment must name a declared deployment")
	}
	deployment, exists := resolved.ModelCatalog.Deployments[name]
	if !exists {
		return fmt.Errorf("global.model_catalog.system.decision_model.deployment: unknown deployment %q", name)
	}
	if !deployment.IsModelRuntime() {
		return fmt.Errorf("global.model_catalog.system.decision_model.deployment %q must use provider model_runtime", name)
	}
	explicit := explicitSystemLines(raw, *system)
	defaults := decisionModelSpec(name, deployment).System
	for _, line := range systemLines {
		if !explicit[line.key] {
			*line.value(system) = *line.value(&defaults)
		}
	}
	return nil
}

// explicitSystemLines are the system lines a configuration sets: the keys of
// its raw global.model_catalog.system, or, for a configuration built in code,
// the lines that differ from the defaults.
func explicitSystemLines(raw *StructuredPayload, system CanonicalSystemModels) map[string]bool {
	explicit := make(map[string]bool, len(systemLines))
	if raw != nil {
		if global, err := raw.AsStringMap(); err == nil {
			for key := range nestedStringMap(nestedStringMap(global["model_catalog"])["system"]) {
				explicit[key] = true
			}
			return explicit
		}
	}
	defaults := DefaultSystemModels()
	for _, line := range systemLines {
		if *line.value(&system) != *line.value(&defaults) {
			explicit[line.key] = true
		}
	}
	return explicit
}

// DecisionModelSpec projects the selected deployment's model identity.
func (c *RouterConfig) DecisionModelSpec() DecisionModelSpec {
	name, deployment, _, err := c.DecisionModelDeployment()
	if err != nil && (c == nil || c.DecisionModel == "") {
		// Directly constructed module configurations still expose their normal
		// built-in defaults. This does not manufacture a runtime deployment.
		deployment, _ = ImplicitModelRuntimeDeployment(Vela2SignalModel, true)
	}
	return decisionModelSpec(name, deployment)
}

// RequiresGPU reports whether a module model runs on a GPU only.
func RequiresGPU(model string) bool {
	spec := GetModelByPath(strings.TrimSpace(model))
	return spec != nil && spec.RequiresGPU
}

// DecisionModelDeployment resolves only an exact declaration key. Canonical
// defaults declare primary explicitly; no resource is created at request time.
func (c *RouterConfig) DecisionModelDeployment() (name string, deployment ModelDeployment, ok bool, err error) {
	if c == nil {
		return "", ModelDeployment{}, false, fmt.Errorf("decision model requires configuration")
	}
	name = c.DecisionModel
	if name == "" {
		name = DefaultDecisionDeployment
	}
	deployment, ok = c.ModelDeployments[name]
	if !ok {
		return name, ModelDeployment{}, false, fmt.Errorf("unknown decision deployment %q", name)
	}
	if !deployment.IsModelRuntime() {
		return name, deployment, false, fmt.Errorf("decision deployment %q must use provider model_runtime", name)
	}
	return name, deployment.WithDefaults(), true, nil
}

// DecisionQuestionDeployment returns the deployment a decision question asks:
// its own, or the decision model's shared deployment when it names none.
func (c *RouterConfig) DecisionQuestionDeployment(rule DecisionSignalRule) string {
	return c.deploymentOrDecisionModel(rule.Deployment)
}

// DecisionSelectorDeployment returns the deployment a decision selector asks:
// its own, or the decision model's shared deployment when it names none, so
// the model that answered the request's signals also chooses its model.
func (c *RouterConfig) DecisionSelectorDeployment(selector DecisionSelectionConfig) string {
	return c.deploymentOrDecisionModel(selector.Deployment)
}

// deploymentOrDecisionModel returns the named deployment, or the decision
// model's shared deployment for an empty name; empty when there is none.
func (c *RouterConfig) deploymentOrDecisionModel(deployment string) string {
	if named := strings.TrimSpace(deployment); named != "" {
		return named
	}
	name, _, ok, err := c.DecisionModelDeployment()
	if !ok || err != nil {
		return ""
	}
	return name
}

// ImplicitDeploymentRequiresGPU reports whether an implicit deployment serves
// a built-in model that runs on a GPU only.
func ImplicitDeploymentRequiresGPU(name string, deployment ModelDeployment) bool {
	if !strings.HasPrefix(name, ImplicitDeploymentPrefix) {
		return false
	}
	for i := range DefaultModelRegistry {
		if spec := &DefaultModelRegistry[i]; spec.RequiresGPU && spec.RepoID == deployment.Artifact {
			return true
		}
	}
	return false
}
