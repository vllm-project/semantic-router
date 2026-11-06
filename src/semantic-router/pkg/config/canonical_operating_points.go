package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// ModuleThresholds are the thresholds of the built-in modules that set one.
type ModuleThresholds struct {
	PromptGuard, Domain, PII, FactCheck, Feedback float32
}

// A threshold belongs to its model. A module that sets none takes the
// threshold calibrated for its preset on the model it runs (presets), which
// vela3-fast and each Vela 2.0 size have. A Vela 1.0 specialist keeps its own
// thresholds, which the modules defaulted to before. No other model has
// calibrated thresholds: it takes the neutral ones, and the load warns.
var vela1ModuleThresholds = ModuleThresholds{PromptGuard: 0.5, Domain: 0.5, PII: 0.9, FactCheck: 0.95, Feedback: 0.7}

// neutralModuleThresholds apply to a module answered by a model without
// calibrated thresholds: the natural boundary of a probability, never another
// model's calibration.
var neutralModuleThresholds = ModuleThresholds{
	PromptGuard: presets.NeutralThreshold, Domain: presets.NeutralThreshold, PII: presets.NeutralThreshold,
	FactCheck: presets.NeutralThreshold, Feedback: presets.NeutralThreshold,
}

// thresholdsOf returns the module thresholds calibrated for a model the
// model runtime serves, and whether it has any: vela3-fast's and each Vela
// 2.0 size's own, a Vela 1.0 specialist's, and none for any other model.
func thresholdsOf(model string) (ModuleThresholds, bool) {
	spec := registeredModel(model)
	if spec == nil {
		return neutralModuleThresholds, false
	}
	if thresholds, ok := calibratedModuleThresholds(spec.RepoID); ok {
		return thresholds, true
	}
	if spec.Purpose == PurposeRoutingSignals {
		return neutralModuleThresholds, false
	}
	return vela1ModuleThresholds, true
}

// calibratedModuleThresholds are a model's calibrated thresholds of the
// presets the modules ask, when it has every one of them.
func calibratedModuleThresholds(repo string) (ModuleThresholds, bool) {
	var thresholds ModuleThresholds
	for _, module := range []struct {
		preset string
		value  *float32
	}{
		{presets.Jailbreak, &thresholds.PromptGuard},
		{presets.Domain, &thresholds.Domain},
		{presets.PII, &thresholds.PII},
		{presets.FactCheck, &thresholds.FactCheck},
		{presets.Feedback, &thresholds.Feedback},
	} {
		threshold, ok := presets.Threshold(repo, module.preset, presets.StageRequest)
		if !ok {
			return ModuleThresholds{}, false
		}
		*module.value = float32(threshold)
	}
	return thresholds, true
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

// ModuleThresholdsOf returns the module thresholds of a module model: its
// calibrated ones, else the neutral ones.
func ModuleThresholdsOf(model string) ModuleThresholds {
	thresholds, _ := thresholdsOf(model)
	return thresholds
}

// ThresholdsOfDeployment are the module thresholds of the model a
// deployment serves: the calibrated ones of a model the model runtime serves
// (thresholdsOf), the Vela 1.0 specialists' for a remote classifier, and the
// neutral ones for a remote decision model or a local model without
// calibration. calibrated is false for the neutral ones.
func ThresholdsOfDeployment(name string, deployment ModelDeployment) (thresholds ModuleThresholds, calibrated bool) {
	switch {
	case deployment.IsModelRuntime():
		return thresholdsOf(deployment.ServedModel(name))
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

// uncalibratedThresholdWarnings name the modules a scope runs with the
// neutral thresholds, because neither the answering model nor the
// configuration gives one.
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
			Code: "uncalibrated_preset", Field: threshold.field,
			Message: fmt.Sprintf("%s runs on deployment %q, whose model has no calibrated threshold for it; it uses %.2f until the threshold is set", BindingKey(threshold.consumer), resolved.Name, *threshold.value(c)),
		})
	}
	return warnings
}
