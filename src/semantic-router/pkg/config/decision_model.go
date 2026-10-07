package config

import (
	"fmt"
	"strings"
)

// The decision model is the Vela model that answers the Router's own
// questions: every built-in signal it covers, and every decision question
// that names no deployment, on one shared deployment and in one call per
// request. global.model_catalog.system.decision_model names it; a
// system.<module> line still binds one signal to another model.

// Decision model names, in size order.
const (
	DecisionModelVela2_03B = "Vela-2.0-0.3B"
	DecisionModelVela2_08B = "Vela-2.0-0.8B"
	DecisionModelVela2_4B  = "Vela-2.0-4B"
	DecisionModelVela2_9B  = "Vela-2.0-9B"
	DecisionModelVela1     = "Vela-1.0"

	// DefaultDecisionModel answers the Router's questions when the
	// configuration names none.
	DefaultDecisionModel = DecisionModelVela2_03B
)

const (
	vela1HazardModel   = "models/Vela-1.0-Encoder-307M-Hazard"
	vela1ModalityModel = "models/Vela-1.0-Encoder-307M-Modality"
)

// DecisionModelSpec is what a decision model binds.
type DecisionModelSpec struct {
	// Name is the canonical name.
	Name string
	// Model is the registry path of the one model that answers every
	// question; empty for Vela 1.0, whose specialists answer none of a
	// configuration's own questions.
	Model string
	// System binds the built-in modules.
	System CanonicalSystemModels
	// Modality is the modality classifier's model when it names none.
	Modality string
}

var decisionModels = []DecisionModelSpec{
	vela2DecisionModel(DecisionModelVela2_03B, Vela2SignalModel),
	vela2DecisionModel(DecisionModelVela2_08B, Vela2Model08B),
	vela2DecisionModel(DecisionModelVela2_4B, Vela2Model4B),
	vela2DecisionModel(DecisionModelVela2_9B, Vela2Model9B),
	{Name: DecisionModelVela1, System: Vela1SystemModels(), Modality: vela1ModalityModel},
}

// vela2DecisionModel binds every built-in signal a Vela 2.0 size answers to
// it; Hazard has no trained question and stays on Vela 1.0 Hazard.
func vela2DecisionModel(name, model string) DecisionModelSpec {
	return DecisionModelSpec{Name: name, Model: model, Modality: model, System: CanonicalSystemModels{
		Safety:                model,
		Hazard:                vela1HazardModel,
		PromptGuard:           model,
		DomainClassifier:      model,
		PIIClassifier:         model,
		FactCheckClassifier:   model,
		HallucinationDetector: model,
		FeedbackDetector:      model,
	}}
}

// DecisionModelNames lists the decision models in size order.
func DecisionModelNames() []string {
	names := make([]string, len(decisionModels))
	for i, spec := range decisionModels {
		names[i] = spec.Name
	}
	return names
}

// LookupDecisionModel resolves a configured decision model name,
// case-insensitively; an empty name is the default.
func LookupDecisionModel(name string) (DecisionModelSpec, error) {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		trimmed = DefaultDecisionModel
	}
	if spec, ok := decisionModelByName(trimmed); ok {
		return spec, nil
	}
	return DecisionModelSpec{}, decisionModelError(name)
}

func decisionModelByName(name string) (DecisionModelSpec, bool) {
	for _, spec := range decisionModels {
		if strings.EqualFold(spec.Name, name) {
			return spec, true
		}
	}
	return DecisionModelSpec{}, false
}

func decisionModelChoices() string {
	names := DecisionModelNames()
	return names[0] + " (the default), " + strings.Join(names[1:len(names)-1], ", ") + " or " + names[len(names)-1]
}

// decisionModel2Families are the Decision 2.0 model names.
var decisionModel2Families = map[string]bool{"kai": true, "eos": true, "sol": true, "nox": true, "lux": true, "vega": true}

func decisionModelError(name string) error {
	trimmed := strings.TrimSpace(name)
	base := trimmed[strings.LastIndex(trimmed, "/")+1:]
	if spec, ok := decisionModelByName(base); ok {
		return fmt.Errorf("decision_model %q: name the model %s, without a repository or path", name, spec.Name)
	}
	if isDecision2Model(base) {
		return fmt.Errorf("decision_model %q is a Decision 2.0 model. The built-in signals ask the questions a Vela model was trained on, so the decision model is %s. "+
			"A Decision 2.0 model answers your own questions: declare it as a model_runtime deployment under global.model_catalog.deployments and name that deployment in a routing.signals.decision question",
			name, decisionModelChoices())
	}
	return fmt.Errorf("decision_model %q is not a decision model; choose %s", name, decisionModelChoices())
}

func isDecision2Model(name string) bool {
	lower := strings.ToLower(name)
	if strings.HasPrefix(lower, "decision-2") || strings.HasPrefix(lower, "decision2") {
		return true
	}
	for _, word := range strings.FieldsFunc(lower, func(r rune) bool { return r == '-' || r == '_' || r == ' ' || r == '.' }) {
		if decisionModel2Families[word] {
			return true
		}
	}
	return false
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

// applyDecisionModel resolves global.model_catalog.system.decision_model to
// its canonical name and binds every module whose system line the
// configuration does not set to the decision model's model.
func applyDecisionModel(resolved *CanonicalGlobal, raw *StructuredPayload) error {
	system := &resolved.ModelCatalog.System
	spec, err := LookupDecisionModel(system.DecisionModel)
	if err != nil {
		return fmt.Errorf("global.model_catalog.system.%w", err)
	}
	system.DecisionModel = spec.Name
	explicit := explicitSystemLines(raw, *system)
	decided := spec.System
	for _, line := range systemLines {
		if !explicit[line.key] {
			*line.value(system) = *line.value(&decided)
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

// DecisionModelSpec returns the configuration's decision model.
func (c *RouterConfig) DecisionModelSpec() DecisionModelSpec {
	spec, err := LookupDecisionModel(c.DecisionModel)
	if err != nil {
		spec, _ = LookupDecisionModel(DefaultDecisionModel)
	}
	return spec
}

// RequiresGPU reports whether a module model runs on a GPU only.
func RequiresGPU(model string) bool {
	spec := GetModelByPath(strings.TrimSpace(model))
	return spec != nil && spec.RequiresGPU
}

// DecisionModelDeployment returns the decision model's shared implicit
// deployment, which answers the decision questions that name no deployment:
// on a GPU when a module runs the model there, else on the CPU. ok is false
// for Vela 1.0, whose specialists answer no decision questions.
func (c *RouterConfig) DecisionModelDeployment() (name string, deployment ModelDeployment, ok bool, err error) {
	spec := c.DecisionModelSpec()
	model := GetModelByPath(spec.Model)
	if spec.Model == "" || model == nil {
		return "", ModelDeployment{}, false, nil
	}
	useCPU := true
	for _, consumer := range c.implicitConsumers() {
		if module, found := c.implicitModule(consumer); found && !module.useCPU {
			if served := GetModelByPath(module.model); served != nil && served.LocalPath == model.LocalPath {
				useCPU = false
			}
		}
	}
	deployment, err = ImplicitModelRuntimeDeployment(spec.Model, useCPU)
	return sharedDeploymentName(model, deployment.Device), deployment, true, err
}

// DecisionQuestionDeployment returns the deployment a decision question asks:
// its own, or the decision model's shared deployment when it names none.
func (c *RouterConfig) DecisionQuestionDeployment(rule DecisionSignalRule) string {
	if deployment := strings.TrimSpace(rule.Deployment); deployment != "" {
		return deployment
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
