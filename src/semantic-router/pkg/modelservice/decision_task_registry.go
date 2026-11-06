package modelservice

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// BuiltinTasks is the canonical semantic template registry shared by the
// Router and Playground. A task a preset backs asks the preset's canonical
// question, and a model family trained on the preset is asked it in its
// trained form (presetPlan). Consumers may specialize labels and
// instructions; the resulting question still compiles through CompileTask.
func BuiltinTasks() []TaskDefinition {
	choice := func(id, title, instructions string, full bool, options ...Choice) TaskDefinition {
		return TaskDefinition{
			ID: id, Title: title, Description: instructions, Stage: "request", Input: "text", Output: "choice", FullInput: full,
			Question: Question{ID: id, Type: "choice", Instructions: instructions, Choices: options, Truncate: !full},
		}
	}
	preset := func(id, title, presetID, stage, input string, options ...string) TaskDefinition {
		canonical, _ := presets.Lookup(presetID, stage)
		question := PresetQuestion(id, canonical, options...)
		return TaskDefinition{
			ID: id, Title: title, Description: canonical.Instructions, Stage: stage, Input: input, Output: canonical.Type,
			FullInput: canonical.Full, Preset: presetID, Question: question,
		}
	}
	noul := func(id, title, instructions, stage, input string) TaskDefinition {
		return TaskDefinition{
			ID: id, Title: title, Description: instructions, Stage: stage, Input: input, Output: "noul", FullInput: true,
			Question: Question{ID: id, Type: "noul", Instructions: instructions},
		}
	}
	labels := func(id, title, instructions, kind, stage, input string, options []Choice) TaskDefinition {
		return TaskDefinition{
			ID: id, Title: title, Description: instructions, Stage: stage, Input: input, Output: kind, FullInput: true,
			Question: Question{ID: id, Type: kind, Instructions: instructions, Labels: options},
		}
	}
	tasks := []TaskDefinition{
		preset("domain", "Subject area", presets.Domain, "request", "text"),
		preset("fact_check", "Need for fact checking", presets.FactCheck, "request", "text"),
		preset("user_feedback", "Feedback on the previous answer", presets.Feedback, "request", "text"),
		preset("modality", "Requested output", presets.Modality, "request", "text"),
		preset("jailbreak", "Instruction attack", presets.Jailbreak, "request", "text"),
		preset("safety", "Harmful request", presets.Harm, "request", "text"),
		preset("complexity", "Task difficulty", presets.Complexity, "request", "text"),
		preset("turn_relation", "Relation to the conversation", presets.TurnRelation, "request", "conversation"),
		preset("refusal", "Refused answer", presets.Refusal, "response", "answer"),
		preset("safety_categories", "Harm categories", presets.HarmCategories, "request", "text"),
		noul("pii_presence", "Personal data present", "Does any part of the input contain personal or sensitive identifying information?", "request", "text"),
		labels("pii_categories", "Personal data categories", "Which personal-data categories occur in the input?", "set", "request", "text", PIICategoryLabels()),
		preset("pii_spans", "Personal data locations", presets.PII, "request", "text"),
		noul("hallucination", "Grounded answer", "Does the answer contain any factual claim that is unsupported by or contradicts the supplied grounding context? Judge the answer against context, not outside knowledge.", "response", "grounded"),
		preset("hallucination_spans", "Unsupported answer locations", presets.Hallucination, "response", "grounded"),
		choice("model_selection", "Choose a candidate model", "Which candidate model is best suited to answer this request? Choose only a listed candidate.", false,
			Choice{"fast", "fast model for straightforward requests"}, Choice{"reasoning", "model for complex reasoning"}),
	}
	tasks[len(tasks)-1].Stage = "selection"
	for i := range tasks {
		kind, consumer, binding, optional := "signal", tasks[i].ID, tasks[i].ID, false
		switch tasks[i].ID {
		case "pii_presence", "pii_categories":
			kind, consumer, binding = "plugin", "masking", "pii"
		case "pii_spans":
			consumer, binding = "pii", "pii"
		case "safety_categories":
			consumer, binding = "safety", "safety"
		case "hallucination":
			kind, binding = "plugin", "hallucination"
		case "hallucination_spans":
			binding = "hallucination"
			consumer = "hallucination"
		case "model_selection":
			kind, consumer, binding = "algorithm", "systemone", ""
		}
		tasks[i].Consumers = []TaskConsumerReference{{Kind: kind, Type: consumer, Optional: optional, Binding: binding}}
	}
	return tasks
}

// presetTasks are the built-in tasks that ask each preset an application
// signal asks.
var presetTasks = map[string]string{
	presets.Domain: "domain", presets.FactCheck: "fact_check", presets.Feedback: "user_feedback", presets.Modality: "modality",
	presets.Jailbreak: "jailbreak", presets.Harm: "safety", presets.HarmCategories: "safety_categories", presets.Complexity: "complexity",
	presets.TurnRelation: "turn_relation", presets.Refusal: "refusal", presets.PII: "pii_spans", presets.Hallucination: "hallucination_spans",
}

// PresetTask is the built-in task that asks a preset.
func PresetTask(preset string) string {
	return presetTasks[preset]
}

// builtinTaskPresets are the presets the built-in tasks ask, by task ID.
var builtinTaskPresets = sync.OnceValue(func() map[string]string {
	presetOf := map[string]string{}
	for _, task := range BuiltinTasks() {
		if task.Preset != "" {
			presetOf[task.ID] = task.Preset
		}
	}
	return presetOf
})

func BuiltinTask(id string) (TaskDefinition, bool) {
	for _, definition := range BuiltinTasks() {
		if definition.ID == id {
			return definition, true
		}
	}
	return TaskDefinition{}, false
}

// PIICategoryLabels are the pii preset's labels with their descriptions. A
// positive presence verdict also covers personal information outside them.
func PIICategoryLabels() []Choice {
	pii, _ := presets.Lookup(presets.PII, presets.StageRequest)
	return presetChoices(pii.Options)
}

// TaskTemplate is native System One wire data for an editable example. It
// deliberately does not promise the chosen model supports that native type.
type TaskTemplate struct {
	State     any            `json:"state"`
	Questions map[string]any `json:"questions"`
}

func (d TaskDefinition) Template() TaskTemplate {
	question := map[string]any{"type": d.Question.Type, "instructions": d.Question.Instructions}
	if d.FullInput {
		question["require_full_input"] = true
	}
	criteria := map[string]string{}
	for _, choice := range append(append([]Choice(nil), d.Question.Choices...), d.Question.Labels...) {
		criteria[choice.Key] = choice.Description
	}
	if len(criteria) > 0 {
		question["criteria"] = criteria
	} else if len(d.Question.Levels) > 0 {
		question["criteria"] = d.Question.Levels
	}
	var state any = "Please explain how binary search works and give a Python example."
	if d.Input == "grounded" {
		state = map[string]string{"context": "The library is open Monday through Friday, 09:00–17:00.", "request": "Is the library open on Sunday?", "answer": "Yes, it is open on Sunday."}
	} else if d.Input == "pair" {
		state = map[string]string{"current": "Could you explain binary search more clearly?", "prior": "How does binary search work?"}
	} else if d.ID == "pii_presence" || d.ID == "pii_categories" || d.ID == "pii_spans" {
		state = "Please email Alex at alex@example.com."
	}
	return TaskTemplate{State: state, Questions: map[string]any{d.ID: question}}
}
