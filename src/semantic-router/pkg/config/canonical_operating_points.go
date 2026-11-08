package config

import "fmt"

// ModuleThresholds are the thresholds of the built-in modules that set one.
type ModuleThresholds struct {
	PromptGuard, Domain, PII, FactCheck, Feedback float32
}

// A threshold belongs to its model. The module thresholds of vela3-fast and of
// each Vela 2.0 size are calibrated on the router signal suite's dev split to
// keep the Vela 1.0 specialists' operating points (src/model-runtime/docs/
// records/vela3-router-signals.md, vela2-router-signals.md and
// vela2-decision-model-sizes.md), so a module that sets none takes the
// thresholds of the model it runs. Any other model keeps the Vela 1.0
// specialists' thresholds, which the modules defaulted to before.
var (
	vela1ModuleThresholds = ModuleThresholds{PromptGuard: 0.5, Domain: 0.5, PII: 0.9, FactCheck: 0.95, Feedback: 0.7}

	calibratedModuleThresholds = map[string]ModuleThresholds{
		Vela3SignalModel: {PromptGuard: 0.88, Domain: 0.41, PII: 0.18, FactCheck: 0.9999, Feedback: 0.47},
		Vela2SignalModel: {PromptGuard: 0.75, Domain: 0.28, PII: 0.01, FactCheck: 0.93, Feedback: 0.37},
		Vela2Model08B:    {PromptGuard: 0.71, Domain: 0.38, PII: 0.07, FactCheck: 0.994, Feedback: 0.34},
		Vela2Model4B:     {PromptGuard: 0.63, Domain: 0.45, PII: 0.05, FactCheck: 0.9984, Feedback: 0.33},
		Vela2Model9B:     {PromptGuard: 0.42, Domain: 0.46, PII: 0.14, FactCheck: 0.998, Feedback: 0.35},
	}
)

// neutralModuleThresholds apply to a module answered by a remote decision
// model that publishes no calibrated thresholds: the natural boundary of a
// probability, never another model's calibration.
var neutralModuleThresholds = ModuleThresholds{PromptGuard: 0.5, Domain: 0.5, PII: 0.5, FactCheck: 0.5, Feedback: 0.5}

// thresholdsOf returns the module thresholds of a module model; remote is a
// module served by a backend rather than a model it names.
func thresholdsOf(model string, remote bool) ModuleThresholds {
	if remote {
		return vela1ModuleThresholds
	}
	if spec := registeredModel(model); spec != nil {
		if thresholds, ok := calibratedModuleThresholds[spec.LocalPath]; ok {
			return thresholds
		}
	}
	return vela1ModuleThresholds
}

// RegisteredModel is the built-in model a path, alias or Hub repository
// names, nil for any other model.
func RegisteredModel(model string) *ModelSpec { return registeredModel(model) }

func registeredModel(model string) *ModelSpec {
	if spec := GetModelByPath(model); spec != nil {
		return spec
	}
	for i := range DefaultModelRegistry {
		if DefaultModelRegistry[i].RepoID == model {
			return &DefaultModelRegistry[i]
		}
	}
	return nil
}

// ModuleThresholdsOf returns the module thresholds calibrated for a module
// model: vela3-fast's or a Vela 2.0 size's own, else the Vela 1.0 specialists'.
func ModuleThresholdsOf(model string) ModuleThresholds {
	return thresholdsOf(model, false)
}

// ThresholdsOfDeployment are the module thresholds of the model a
// deployment serves: its calibrated ones for a built-in model the model
// runtime serves, the Vela 1.0 specialists' for another local model or a
// remote classifier, and the neutral ones for a remote decision model.
// calibrated is false for the neutral ones.
func ThresholdsOfDeployment(name string, deployment ModelDeployment) (thresholds ModuleThresholds, calibrated bool) {
	switch {
	case deployment.IsModelRuntime():
		return thresholdsOf(deployment.ServedModel(name), false), true
	case deployment.IsSystemOne():
		return neutralModuleThresholds, false
	}
	return vela1ModuleThresholds, true
}

// moduleThreshold is one module threshold: the consumer it belongs to, the
// field it is written in, and how to read and pick it.
type moduleThreshold struct {
	consumer string
	field    string
	value    func(*RouterConfig) *float32
	pick     func(ModuleThresholds) float32
}

var moduleThresholdFields = []moduleThreshold{
	{"prompt_guard", "global.model_catalog.modules.prompt_guard.threshold", func(c *RouterConfig) *float32 { return &c.PromptGuard.Threshold }, func(t ModuleThresholds) float32 { return t.PromptGuard }},
	{"domain_classifier", "global.model_catalog.modules.classifier.domain.threshold", func(c *RouterConfig) *float32 { return &c.CategoryModel.Threshold }, func(t ModuleThresholds) float32 { return t.Domain }},
	{"pii_classifier", "global.model_catalog.modules.classifier.pii.threshold", func(c *RouterConfig) *float32 { return &c.PIIModel.Threshold }, func(t ModuleThresholds) float32 { return t.PII }},
	{"fact_check_classifier", "global.model_catalog.modules.hallucination_mitigation.fact_check.threshold", func(c *RouterConfig) *float32 { return &c.HallucinationMitigation.FactCheckModel.Threshold }, func(t ModuleThresholds) float32 { return t.FactCheck }},
	{"feedback_detector", "global.model_catalog.modules.feedback_detector.threshold", func(c *RouterConfig) *float32 { return &c.FeedbackDetector.Threshold }, func(t ModuleThresholds) float32 { return t.Feedback }},
}

// authoredModuleThresholds lists the module thresholds a document sets.
func authoredModuleThresholds(raw *StructuredPayload) map[string]bool {
	authored := map[string]bool{}
	if raw == nil {
		return authored
	}
	global, err := raw.AsStringMap()
	if err != nil {
		return authored
	}
	modules := nestedStringMap(nestedStringMap(global["model_catalog"])["modules"])
	classifier := nestedStringMap(modules["classifier"])
	halu := nestedStringMap(modules["hallucination_mitigation"])
	for consumer, module := range map[string]map[string]interface{}{
		"prompt_guard":          nestedStringMap(modules["prompt_guard"]),
		"domain_classifier":     nestedStringMap(classifier["domain"]),
		"pii_classifier":        nestedStringMap(classifier["pii"]),
		"fact_check_classifier": nestedStringMap(halu["fact_check"]),
		"feedback_detector":     nestedStringMap(modules["feedback_detector"]),
	} {
		if hasRawKey(module, "threshold") {
			authored[consumer] = true
		}
	}
	return authored
}

// applyModuleThresholds sets every module threshold the configuration does
// not set to the one of the deployment that answers the module in this
// scope. A configuration built in code keeps the thresholds it sets.
func (c *RouterConfig) applyModuleThresholds() {
	if c == nil || c.AuthoredThresholds == nil {
		return
	}
	for _, threshold := range moduleThresholdFields {
		if c.AuthoredThresholds[threshold.consumer] {
			continue
		}
		resolved, ok, err := c.resolveTask(threshold.consumer)
		if !ok || err != nil {
			continue
		}
		thresholds, _ := ThresholdsOfDeployment(resolved.Name, resolved.Deployment)
		*threshold.value(c) = threshold.pick(thresholds)
	}
}

// uncalibratedThresholdWarnings name the modules a scope runs on a remote
// decision model with the neutral thresholds, because neither the model nor
// the configuration gives one.
func (c *RouterConfig) uncalibratedThresholdWarnings(scope RecipeName) []ConfigWarning {
	var warnings []ConfigWarning
	for _, threshold := range moduleThresholdFields {
		if c.AuthoredThresholds[threshold.consumer] || !TaskConsumerInUse(c, scope, threshold.consumer) {
			continue
		}
		resolved, ok, err := c.resolveTask(threshold.consumer)
		if !ok || err != nil {
			continue
		}
		if _, calibrated := ThresholdsOfDeployment(resolved.Name, resolved.Deployment); calibrated {
			continue
		}
		warnings = append(warnings, ConfigWarning{
			Code: "uncalibrated_threshold", Field: threshold.field,
			Message: fmt.Sprintf("%s runs on deployment %q, which publishes no calibrated threshold; it uses %.2f until the threshold is set", BindingKey(threshold.consumer), resolved.Name, *threshold.value(c)),
		})
	}
	return warnings
}
