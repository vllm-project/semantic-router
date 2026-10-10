package config

// ModuleThresholds are the thresholds of the built-in modules that set one.
type ModuleThresholds struct {
	PromptGuard, Domain, PII, FactCheck, Feedback float32
}

// A threshold belongs to its model. Each Vela 2.0 size's module thresholds are
// calibrated on the router signal suite's dev split to keep the Vela 1.0
// specialists' operating points (src/model-runtime/docs/records/
// vela2-router-signals.md and vela2-decision-model-sizes.md), so a module that
// sets none takes the thresholds of the model it runs. Any other model keeps
// the Vela 1.0 specialists' thresholds, which the modules defaulted to before.
var (
	vela1ModuleThresholds = ModuleThresholds{PromptGuard: 0.5, Domain: 0.5, PII: 0.9, FactCheck: 0.95, Feedback: 0.7}

	vela2ModuleThresholds = map[string]ModuleThresholds{
		Vela2SignalModel: {PromptGuard: 0.75, Domain: 0.28, PII: 0.01, FactCheck: 0.93, Feedback: 0.37},
		Vela2Model08B:    {PromptGuard: 0.71, Domain: 0.38, PII: 0.07, FactCheck: 0.994, Feedback: 0.34},
		Vela2Model4B:     {PromptGuard: 0.63, Domain: 0.45, PII: 0.05, FactCheck: 0.9984, Feedback: 0.33},
		Vela2Model9B:     {PromptGuard: 0.42, Domain: 0.46, PII: 0.14, FactCheck: 0.998, Feedback: 0.35},
	}
)

// thresholdsOf returns the module thresholds of a module model; remote is a
// module served by a backend rather than a model it names.
func thresholdsOf(model string, remote bool) ModuleThresholds {
	if remote {
		return vela1ModuleThresholds
	}
	if spec := GetModelByPath(model); spec != nil {
		if thresholds, ok := vela2ModuleThresholds[spec.LocalPath]; ok {
			return thresholds
		}
	}
	return vela1ModuleThresholds
}

// ModuleThresholdsOf returns the module thresholds calibrated for a module
// model: a Vela 2.0 size's own, else the Vela 1.0 specialists'.
func ModuleThresholdsOf(model string) ModuleThresholds {
	return thresholdsOf(model, false)
}

// normalizeModuleOperatingPoints runs after module model references resolve:
// every module threshold the configuration does not set becomes the one of
// the model the module runs.
func normalizeModuleOperatingPoints(resolved *CanonicalGlobal, raw *StructuredPayload) {
	if resolved == nil || raw == nil {
		return
	}
	global, err := raw.AsStringMap()
	if err != nil {
		return
	}
	modules := nestedStringMap(nestedStringMap(global["model_catalog"])["modules"])
	classifier := nestedStringMap(modules["classifier"])
	halu := nestedStringMap(modules["hallucination_mitigation"])
	m := &resolved.ModelCatalog.Modules
	guard := thresholdsOf(m.PromptGuard.ModelID, m.PromptGuard.Backend != nil)
	domain := thresholdsOf(m.Classifier.Domain.ModelID, m.Classifier.Domain.Backend != nil)
	pii := thresholdsOf(m.Classifier.PII.ModelID, m.Classifier.PII.Backend != nil)
	factCheck := thresholdsOf(m.HallucinationMitigation.FactCheck.ModelID, false)
	feedback := thresholdsOf(m.FeedbackDetector.ModelID, false)
	for _, module := range []struct {
		raw       map[string]interface{}
		threshold *float32
		value     float32
	}{
		{nestedStringMap(modules["prompt_guard"]), &m.PromptGuard.Threshold, guard.PromptGuard},
		{nestedStringMap(classifier["domain"]), &m.Classifier.Domain.Threshold, domain.Domain},
		{nestedStringMap(classifier["pii"]), &m.Classifier.PII.Threshold, pii.PII},
		{nestedStringMap(halu["fact_check"]), &m.HallucinationMitigation.FactCheck.Threshold, factCheck.FactCheck},
		{nestedStringMap(modules["feedback_detector"]), &m.FeedbackDetector.Threshold, feedback.Feedback},
	} {
		if !hasRawKey(module.raw, "threshold") {
			*module.threshold = module.value
		}
	}
}
