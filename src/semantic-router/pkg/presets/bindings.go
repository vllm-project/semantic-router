package presets

import "slices"

// Model families with trained bindings. The vela3_d3 sizes (vela3-lite,
// -mini, -flash, -turbo) turn a trained adapter on by exact question ID;
// vela3 (vela3-fast) and vela2 (Vela 2.0) pick it by exact wording, which for
// every preset they were trained on is the canonical wording.
const (
	FamilyVela3   = "vela3"
	FamilyVela3D3 = "vela3_d3"
	FamilyVela2   = "vela2"
)

// Binding is the form a model family was trained to be asked a preset in.
// The zero Binding asks the canonical question under the Router's own ID.
type Binding struct {
	// ID is the trained question ID; empty keeps the Router's own ID.
	ID string
	// Options, when set, replace the canonical options in the trained
	// order; Canonical maps each trained key to the canonical option it
	// answers.
	Options   []Option
	Canonical map[string]string
	// Covers lists the canonical options the trained form answers; a
	// question that needs another one is asked in its canonical form.
	// Empty covers every option.
	Covers []string
	// ModelPreset asks the model's own preset question (pii, halu) instead.
	ModelPreset string
	// NativeSet asks a Set question natively: the family answers it for
	// the trained wording although its card lists no set type.
	NativeSet bool
	// LabelNouls asks a Set question as one Noul per label, in the
	// family's trained template.
	LabelNouls *NoulTemplate
}

// NoulTemplate is a trained per-label Noul: its ID prefix (followed by the
// label key), its instructions (%s is the label's description) and its
// false and true outcomes.
type NoulTemplate struct {
	IDPrefix     string
	Instructions string
	Outcomes     []Option
}

// CoversOptions reports whether the binding answers every one of keys.
func (b Binding) CoversOptions(keys []string) bool {
	if len(b.Covers) == 0 {
		return true
	}
	for _, key := range keys {
		if !slices.Contains(b.Covers, key) {
			return false
		}
	}
	return true
}

// CanonicalKey is the canonical option a trained option key answers.
func (b Binding) CanonicalKey(trained string) string {
	if canonical, ok := b.Canonical[trained]; ok {
		return canonical
	}
	return trained
}

type bindingKey struct{ family, preset, stage string }

var (
	modalityOptions = []Option{
		{"AR", "a text answer only, including code, analysis or describing an existing image"},
		{"DIFFUSION", "a newly generated image only"},
		{"BOTH", "a newly generated image together with a separate written explanation"},
	}
	modalityCanonical = map[string]string{"AR": "text", "DIFFUSION": "image", "BOTH": "text_image"}
	answerHarmOptions = []Option{
		{"safe response", "an assistant answer that is itself harmless"},
		{"unsafe response", "an assistant answer that is itself harmful, or that supplies the harmful assistance asked for"},
	}
	answerHarmCanonical = map[string]string{"safe response": "safe", "unsafe response": "unsafe"}
	hazardNouls         = &NoulTemplate{
		IDPrefix:     "hazard:",
		Instructions: "Is this a harmful request, and does its harm involve %s?",
		Outcomes: []Option{
			{"false", "No. It is not harmful, or its harm is of another kind; a safe request that only mentions the topic is No."},
			{"true", "Yes. Fulfilling it would cause or facilitate harm of this kind."},
		},
	}
)

// vela3Bindings are the router questions both Vela 3.0 families were
// trained on (router_questions.json), under their trained IDs.
func vela3Bindings(family string) map[bindingKey]Binding {
	bindings := map[bindingKey]Binding{
		{family, Domain, StageRequest}:         {ID: "domain"},
		{family, FactCheck, StageRequest}:      {ID: "factcheck"},
		{family, Feedback, StageRequest}:       {ID: "feedback"},
		{family, Jailbreak, StageRequest}:      {ID: "attack"},
		{family, Jailbreak, StageResponse}:     {ID: "attack"},
		{family, Harm, StageRequest}:           {ID: "p_harm"},
		{family, Harm, StageResponse}:          {ID: "r_harm", Options: answerHarmOptions, Canonical: answerHarmCanonical},
		{family, Refusal, StageResponse}:       {ID: "r_refusal"},
		{family, Modality, StageRequest}:       {ID: "modality", Options: modalityOptions, Canonical: modalityCanonical, Covers: []string{"text", "image", "text_image"}},
		{family, PII, StageRequest}:            {ModelPreset: "pii"},
		{family, Hallucination, StageResponse}: {ModelPreset: "halu"},
	}
	if family == FamilyVela3 {
		bindings[bindingKey{family, HarmCategories, StageRequest}] = Binding{ID: "p_cats", NativeSet: true}
		bindings[bindingKey{family, HarmCategories, StageResponse}] = Binding{ID: "r_cats", NativeSet: true}
	} else {
		// The vela3_d3 packages were trained on the longer NO_FEEDBACK
		// description; vela3-fast's thresholds were calibrated with the
		// canonical one, which it reads on its router adapter just as well.
		feedback, _ := Lookup(Feedback, StageRequest)
		for i := range feedback.Options {
			if feedback.Options[i].Key == "NO_FEEDBACK" {
				feedback.Options[i].Description = "the user turn gives no feedback on the previous answer: it continues the conversation, adds details or starts a new request"
			}
		}
		bindings[bindingKey{family, Feedback, StageRequest}] = Binding{ID: "feedback", Options: feedback.Options}
		bindings[bindingKey{family, HarmCategories, StageRequest}] = Binding{LabelNouls: hazardNouls}
	}
	return bindings
}

var bindings = func() map[bindingKey]Binding {
	all := map[bindingKey]Binding{}
	for _, family := range []string{FamilyVela3, FamilyVela3D3} {
		for key, binding := range vela3Bindings(family) {
			all[key] = binding
		}
	}
	for _, preset := range []string{Domain, FactCheck, Feedback, Harm} {
		all[bindingKey{FamilyVela2, preset, StageRequest}] = Binding{}
	}
	all[bindingKey{FamilyVela2, Jailbreak, StageRequest}] = Binding{}
	all[bindingKey{FamilyVela2, Jailbreak, StageResponse}] = Binding{}
	all[bindingKey{FamilyVela2, PII, StageRequest}] = Binding{ModelPreset: "pii"}
	all[bindingKey{FamilyVela2, Hallucination, StageResponse}] = Binding{ModelPreset: "halu"}
	// A model of no known family, such as a remote API's, answers a span
	// preset with its own question where its card lists one.
	all[bindingKey{"", PII, StageRequest}] = Binding{ModelPreset: "pii"}
	all[bindingKey{"", Hallucination, StageResponse}] = Binding{ModelPreset: "halu"}
	return all
}()

// Trained returns the form a model family was trained to be asked a preset
// in, for a stage, and false when the family answers it zero-shot.
func Trained(family, preset, stage string) (Binding, bool) {
	if stage == "" {
		stage = StageRequest
	}
	binding, ok := bindings[bindingKey{family, preset, stage}]
	if !ok {
		return Binding{}, false
	}
	binding.Options = slices.Clone(binding.Options)
	binding.Covers = slices.Clone(binding.Covers)
	return binding, true
}
