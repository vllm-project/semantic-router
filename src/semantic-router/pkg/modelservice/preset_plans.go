package modelservice

import (
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// A task a preset backs asks the preset's canonical question. A model family
// trained on the preset answers it on a trained adapter only when it goes out
// the way it was trained (presets.Trained): under the trained ID and with the
// trained options, as one native Set, as the model's own preset question, or
// as one trained Noul per label. Every other family answers the canonical
// question zero-shot under the Router's own ID, which always contains a ':'
// and so never equals a trained ID.

// Plan implementations: a question asked as written, asked in its family's
// trained form, or as one Noul per Set label, as written or as trained.
const (
	implementationNative       = "native"
	implementationTrained      = "trained"
	implementationComposedNoul = "composed_noul"
	implementationTrainedNoul  = "trained_noul"
)

// decisionAdapterFamily is the family whose packages may answer a question
// no adapter of theirs was trained on with an opt-in decision adapter.
const decisionAdapterFamily = presets.FamilyVela3D3

// PresetQuestion is a preset's canonical question under id: its type,
// instructions and options, or only the named options, in catalog order. A
// question that need not read a long input whole reads its first tokens.
func PresetQuestion(id string, preset presets.Preset, options ...string) Question {
	question := Question{ID: id, Type: preset.Type, Instructions: preset.Instructions, Truncate: !preset.Full}
	choices := presetChoices(preset.Options)
	if len(options) > 0 {
		choices = slices.DeleteFunc(choices, func(choice Choice) bool { return !slices.Contains(options, choice.Key) })
	}
	if preset.Type == presets.Set || preset.Type == presets.Span {
		question.Labels = choices
	} else {
		question.Choices = choices
	}
	return question
}

func presetChoices(options []presets.Option) []Choice {
	choices := make([]Choice, len(options))
	for index, option := range options {
		choices[index] = Choice{Key: option.Key, Description: option.Description}
	}
	return choices
}

// presetPlan returns the questions that ask a preset-backed task in the form
// the card's family was trained on, their implementation and the canonical
// option each trained option answers; false when the family answers the
// task's question as asked. A Choice question a consumer specialized (other
// wording, options that are not the preset's) is asked as written.
func presetPlan(definition TaskDefinition, question Question, card ModelCard) (trainedForm, bool) {
	if definition.Preset == "" || !card.Serves("decisions") {
		return trainedForm{}, false
	}
	preset, ok := presets.Lookup(definition.Preset, definition.Stage)
	if !ok {
		return trainedForm{}, false
	}
	binding, ok := presets.Trained(card.Family, definition.Preset, definition.Stage)
	if !ok {
		return trainedForm{}, false
	}
	switch {
	case binding.ModelPreset != "":
		if !card.HasPreset(binding.ModelPreset) {
			return trainedForm{}, false
		}
		asked := Question{ID: question.ID, Preset: binding.ModelPreset, Threshold: question.Threshold, Truncate: question.Truncate, RequireFullInput: question.RequireFullInput}
		return trainedForm{questions: []Question{asked}, implementation: implementationTrained}, true
	case binding.LabelNouls != nil:
		return trainedLabelNouls(question, *binding.LabelNouls, card)
	case binding.NativeSet:
		return trainedSet(question, preset, binding)
	}
	reads := question.Reads
	if reads == nil {
		reads = optionKeys(question.Choices)
	}
	if !asksPreset(question, preset) || !binding.CoversOptions(reads) || !card.Answers(question.Type) {
		return trainedForm{}, false
	}
	trained := question
	if binding.ID != "" {
		trained.ID = binding.ID
	}
	form := trainedForm{implementation: implementationTrained, semantic: question}
	if len(binding.Options) > 0 {
		trained.Choices = presetChoices(binding.Options)
		form.canonical = make(map[string]string, len(binding.Options))
		form.semantic.Choices = make([]Choice, len(binding.Options))
		for index, option := range binding.Options {
			key := binding.CanonicalKey(option.Key)
			form.canonical[option.Key] = key
			form.semantic.Choices[index] = canonicalChoice(preset, key)
		}
	}
	form.questions = []Question{trained}
	return form, true
}

// trainedForm is how a family is asked a preset: the questions on the wire,
// their implementation, the semantic question their answer reads as when the
// trained options are another set than the asked ones, and the canonical
// option each trained option answers.
type trainedForm struct {
	questions      []Question
	implementation string
	semantic       Question
	canonical      map[string]string
}

// asksPreset reports whether a Choice or Noul question is the preset's
// canonical question, or asks only some of its options.
func asksPreset(question Question, preset presets.Preset) bool {
	if question.Type != preset.Type || question.Instructions != preset.Instructions || len(question.Levels) > 0 || len(question.Labels) > 0 {
		return false
	}
	next := 0
	for _, choice := range question.Choices {
		index := slices.IndexFunc(preset.Options, func(option presets.Option) bool { return option.Key == choice.Key })
		if index < next || preset.Options[index].Description != choice.Description {
			return false
		}
		next = index + 1
	}
	return true
}

func canonicalChoice(preset presets.Preset, key string) Choice {
	for _, option := range preset.Options {
		if option.Key == key {
			return Choice{Key: option.Key, Description: option.Description}
		}
	}
	return Choice{Key: key}
}

// trainedSet asks a Set question as the family's trained Set when every
// label is one of the preset's: its trained ID, instructions and label
// descriptions.
func trainedSet(question Question, preset presets.Preset, binding presets.Binding) (trainedForm, bool) {
	if question.Type != presets.Set || len(question.Labels) == 0 {
		return trainedForm{}, false
	}
	labels := make([]Choice, len(question.Labels))
	for index, label := range question.Labels {
		if !preset.HasOption(label.Key) {
			return trainedForm{}, false
		}
		labels[index] = canonicalChoice(preset, label.Key)
	}
	return trainedForm{implementation: implementationTrained, questions: []Question{{
		ID: binding.ID, Type: presets.Set, Instructions: preset.Instructions, Labels: labels,
		Threshold: question.Threshold, Truncate: question.Truncate, RequireFullInput: question.RequireFullInput,
	}}}, true
}

// trainedLabelNouls asks a Set question as one trained Noul per label,
// worded with each label's description.
func trainedLabelNouls(question Question, template presets.NoulTemplate, card ModelCard) (trainedForm, bool) {
	if question.Type != presets.Set || len(question.Labels) == 0 || !card.Answers(presets.Noul) {
		return trainedForm{}, false
	}
	outcomes := presetChoices(template.Outcomes)
	questions := make([]Question, len(question.Labels))
	for index, label := range question.Labels {
		description := label.Description
		if description == "" {
			description = label.Key
		}
		questions[index] = Question{
			ID: template.IDPrefix + label.Key, Type: presets.Noul, Truncate: question.Truncate, RequireFullInput: question.RequireFullInput,
			Instructions: fmt.Sprintf(template.Instructions, description), Choices: slices.Clone(outcomes),
		}
	}
	return trainedForm{questions: questions, implementation: implementationTrainedNoul}, true
}

// withDecisionAdapter marks the Choice, Noul and Score questions of a plan no
// adapter was trained on for a family whose packages may answer them with an
// opt-in decision adapter. A package without one ignores the field.
func withDecisionAdapter(plan TaskPlan, card ModelCard) TaskPlan {
	if card.Family != decisionAdapterFamily || (plan.Implementation != implementationNative && plan.Implementation != implementationComposedNoul) {
		return plan
	}
	plan.Questions = slices.Clone(plan.Questions)
	for index, question := range plan.Questions {
		switch question.Type {
		case presets.Choice, presets.Noul, presets.Score:
			plan.Questions[index].DecisionAdapter = true
		}
	}
	return plan
}

func optionKeys(choices []Choice) []string {
	keys := make([]string, len(choices))
	for index, choice := range choices {
		keys[index] = choice.Key
	}
	return keys
}

// sameQuestion reports whether two questions ask the same: their type,
// wording and options, in order.
func sameQuestion(asked, canonical Question) bool {
	return asked.Type == canonical.Type && asked.Instructions == canonical.Instructions &&
		slices.Equal(asked.Choices, canonical.Choices) && slices.Equal(asked.Levels, canonical.Levels) &&
		slices.Equal(asked.Labels, canonical.Labels)
}

// sameWireQuestion reports whether two questions under one ID ask the same
// thing, so one answer serves both.
func sameWireQuestion(a, b Question) bool {
	return sameQuestion(a, b) && a.Head == b.Head && a.Preset == b.Preset && a.Truncate == b.Truncate &&
		a.RequireFullInput == b.RequireFullInput && a.DecisionAdapter == b.DecisionAdapter && equalThreshold(a.Threshold, b.Threshold)
}

func equalThreshold(a, b *float64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}
