package config

// The module defaults' thresholds are Vela 2.0 0.3B's, calibrated on the
// router signal suite's dev split to keep the Vela 1.0 specialists'
// operating points (src/model-runtime/docs/records/vela2-router-signals.md).
// A threshold belongs to its model: a module that runs any other model and
// sets none keeps the threshold the module defaulted to before.
var previousModuleThresholds = struct {
	PromptGuard, Domain, PII, FactCheck, Feedback float32
}{PromptGuard: 0.5, Domain: 0.5, PII: 0.9, FactCheck: 0.95, Feedback: 0.7}

// normalizeModuleOperatingPoints runs after module model references resolve.
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
	for _, module := range []struct {
		raw       map[string]interface{}
		model     string
		remote    bool
		threshold *float32
		previous  float32
	}{
		{nestedStringMap(modules["prompt_guard"]), m.PromptGuard.ModelID, m.PromptGuard.Backend != nil, &m.PromptGuard.Threshold, previousModuleThresholds.PromptGuard},
		{nestedStringMap(classifier["domain"]), m.Classifier.Domain.ModelID, m.Classifier.Domain.Backend != nil, &m.Classifier.Domain.Threshold, previousModuleThresholds.Domain},
		{nestedStringMap(classifier["pii"]), m.Classifier.PII.ModelID, m.Classifier.PII.Backend != nil, &m.Classifier.PII.Threshold, previousModuleThresholds.PII},
		{nestedStringMap(halu["fact_check"]), m.HallucinationMitigation.FactCheck.ModelID, false, &m.HallucinationMitigation.FactCheck.Threshold, previousModuleThresholds.FactCheck},
		{nestedStringMap(modules["feedback_detector"]), m.FeedbackDetector.ModelID, false, &m.FeedbackDetector.Threshold, previousModuleThresholds.Feedback},
	} {
		if !hasRawKey(module.raw, "threshold") && (module.remote || !runsVela2SignalModel(module.model)) {
			*module.threshold = module.previous
		}
	}
}

func runsVela2SignalModel(model string) bool {
	spec := GetModelByPath(model)
	return spec != nil && spec.LocalPath == Vela2SignalModel
}
