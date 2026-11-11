package presets

// Calibrated thresholds by served model and preset (request stage): the
// threshold that keeps the Vela 1.0 specialists' operating point on the
// router signal suite's dev split (src/model-runtime/docs/records/
// vela3-router-signals.md, vela2-router-signals.md and
// vela2-decision-model-sizes.md). It bounds the probability of the option a
// rule matches (Choice) or of each span (Span). They were measured with the
// canonical wording, so they hold for the trained bindings of these models.
var calibration = map[string]map[string]float64{
	"vllm-sr/vela3-fast":    {Jailbreak: 0.88, Domain: 0.41, PII: 0.18, FactCheck: 0.9999, Feedback: 0.47, Harm: 0.52, Modality: 0.63},
	"vllm-sr/Vela-2.0-0.3B": {Jailbreak: 0.75, Domain: 0.28, PII: 0.01, FactCheck: 0.93, Feedback: 0.37, Harm: 0.46, Modality: 0.51},
	"vllm-sr/Vela-2.0-0.8B": {Jailbreak: 0.71, Domain: 0.38, PII: 0.07, FactCheck: 0.994, Feedback: 0.34},
	"vllm-sr/Vela-2.0-4B":   {Jailbreak: 0.63, Domain: 0.45, PII: 0.05, FactCheck: 0.9984, Feedback: 0.33},
	"vllm-sr/Vela-2.0-9B":   {Jailbreak: 0.42, Domain: 0.46, PII: 0.14, FactCheck: 0.998, Feedback: 0.35},
}

// NeutralThreshold is the predicate of a preset on a model without a
// calibrated threshold for it: the natural boundary of a probability.
const NeutralThreshold = 0.5

// Threshold returns the calibrated threshold of a preset in a stage on a
// model (its Hugging Face repository ID), false when it has none.
func Threshold(model, preset, stage string) (float64, bool) {
	if stage != "" && stage != StageRequest {
		return 0, false
	}
	threshold, ok := calibration[model][preset]
	return threshold, ok
}
