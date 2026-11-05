package modelservice

import (
	"fmt"
	"slices"
	"sync"
)

// Model families trained on the Router's built-in questions answer them on
// trained adapters only when a question goes out the way it was trained. The
// vela3_d3 sizes (vela3-lite, -mini, -flash, -turbo) turn an adapter on by
// exact question ID; vela3 (vela3-fast) by exact wording, which is the
// built-in wording its thresholds were calibrated with. Every other family
// answers the canonical question zero-shot, under the Router's own ID, which
// always contains a ':' and so never equals a trained ID.
const (
	familyVela3   = "vela3"
	familyVela3D3 = "vela3_d3"
)

// Plan implementations: a question asked as written, asked in its family's
// trained form, or as one Noul per Set label, as written or as trained.
const (
	implementationNative       = "native"
	implementationTrained      = "trained"
	implementationComposedNoul = "composed_noul"
	implementationTrainedNoul  = "trained_noul"
)

// trainedQuestionIDs are the IDs the Vela 3.0 packages trained each built-in
// question under (router_questions.json).
var trainedQuestionIDs = map[string]string{
	"domain":        "domain",
	"fact_check":    "factcheck",
	"user_feedback": "feedback",
	"modality":      "modality",
	"jailbreak":     "attack",
	"safety":        "p_harm",
}

// trainedFeedbackNone is the NO_FEEDBACK description the vela3_d3 packages
// were trained with.
const trainedFeedbackNone = "the user turn gives no feedback on the previous answer: it continues the conversation, adds details or starts a new request"

// Harm categories: vela3-fast was trained on one Set question over 34 labels
// (the AEGIS 2.0 categories and the MLCommons hazard codes); a vela3_d3
// package that ships a hazard adapter turns it on for every Noul whose ID
// starts with "hazard:", one per category, worded as trained.
const (
	trainedHazardSetID        = "p_cats"
	trainedHazardSet          = "Which hazard categories apply to this request?"
	trainedHazardNoulPrefix   = "hazard:"
	trainedHazardNoulQuestion = "Is this a harmful request, and does its harm involve %s?"
)

var trainedHazardNoulOutcomes = []Choice{
	{"false", "No. It is not harmful, or its harm is of another kind; a safe request that only mentions the topic is No."},
	{"true", "Yes. Fulfilling it would cause or facilitate harm of this kind."},
}

// trainedHazardCategories are vela3-fast's 34 harm categories with their
// trained descriptions.
var trainedHazardCategories = map[string]string{
	"Controlled/Regulated Substances":     "obtaining, producing or misusing drugs or regulated substances",
	"Copyright/Trademark/Plagiarism":      "infringing copyright or passing off others' work",
	"Criminal Planning/Confessions":       "planning, assisting or confessing to a crime",
	"Fraud/Deception":                     "scams, phishing, forgery or deceiving others for gain",
	"Guns and Illegal Weapons":            "acquiring, building or using weapons illegally",
	"Harassment":                          "targeted abuse, bullying or intimidation of a person",
	"Hate/Identity Hate":                  "hatred or dehumanisation targeting a protected identity",
	"High Risk Gov Decision Making":       "high-stakes government, military or public-safety decisions",
	"Illegal Activity":                    "unlawful activity not covered by a more specific category",
	"Immoral/Unethical":                   "conduct that is unethical without being clearly illegal",
	"Needs Caution":                       "ambiguous content warranting caution without a clear violation",
	"Other":                               "a safety concern that no other category describes",
	"PII/Privacy":                         "seeking or exposing personal or private information about someone",
	"Political/Misinformation/Conspiracy": "political manipulation, misinformation or conspiracy content",
	"Profanity":                           "obscene or vulgar language",
	"S1":                                  "violent crimes, including terrorism and physical assault",
	"S2":                                  "non-violent crimes such as theft, trafficking or cybercrime",
	"S3":                                  "sex-related crimes including trafficking and sexual assault",
	"S4":                                  "child sexual exploitation of any kind",
	"S5":                                  "defamation, false damaging statements about a real person",
	"S6":                                  "specialised medical, legal or financial advice without qualification",
	"S7":                                  "privacy violations, exposing personal information",
	"S8":                                  "intellectual property infringement",
	"S9":                                  "indiscriminate weapons, chemical, biological, nuclear or radiological",
	"S10":                                 "hate targeting a protected characteristic",
	"S11":                                 "suicide and self-harm",
	"S12":                                 "sexual content involving adults",
	"S14":                                 "abuse of a code interpreter or execution environment",
	"Sexual":                              "sexually explicit content involving adults",
	"Sexual (minor)":                      "any sexual content involving a minor",
	"Suicide and Self Harm":               "self-harm, suicide or encouragement of either",
	"Threat":                              "an explicit threat of harm directed at someone",
	"Unauthorized Advice":                 "medical, legal or financial advice given without qualification",
	"Violence":                            "inciting, threatening or describing physical violence against people",
}

// canonicalQuestions are the built-in tasks' questions by task ID.
var canonicalQuestions = sync.OnceValue(func() map[string]Question {
	questions := make(map[string]Question)
	for _, task := range BuiltinTasks() {
		questions[task.ID] = task.Question
	}
	return questions
})

// trainedPlan returns the questions that ask a built-in task in the form the
// card's family was trained on, and their implementation; false when the
// family has no trained form of it. A question a consumer specialized (other
// options, other wording) keeps the canonical path.
func trainedPlan(taskID string, question Question, card ModelCard) ([]Question, string, bool) {
	if (card.Family != familyVela3 && card.Family != familyVela3D3) || !card.Serves("decisions") {
		return nil, "", false
	}
	if taskID == "safety_categories" {
		return trainedHazardQuestions(question, card)
	}
	id, ok := trainedQuestionIDs[taskID]
	if !ok || !sameQuestion(question, canonicalQuestions()[taskID]) || !card.Answers(question.Type) {
		return nil, "", false
	}
	trained := question
	trained.ID = id
	if card.Family == familyVela3D3 && taskID == "user_feedback" {
		trained.Choices = slices.Clone(question.Choices)
		for i := range trained.Choices {
			if trained.Choices[i].Key == "NO_FEEDBACK" {
				trained.Choices[i].Description = trainedFeedbackNone
			}
		}
	}
	return []Question{trained}, implementationTrained, true
}

// trainedHazardQuestions asks harm categories as trained: vela3-fast's Set,
// when every label is one of its categories, or one hazard Noul per label on
// a vela3_d3 size.
func trainedHazardQuestions(question Question, card ModelCard) ([]Question, string, bool) {
	if question.Type != "set" || len(question.Labels) == 0 {
		return nil, "", false
	}
	if card.Family == familyVela3 {
		labels := make([]Choice, len(question.Labels))
		for i, label := range question.Labels {
			description, trained := trainedHazardCategories[label.Key]
			if !trained {
				return nil, "", false
			}
			labels[i] = Choice{Key: label.Key, Description: description}
		}
		return []Question{{
			ID: trainedHazardSetID, Type: "set", Instructions: trainedHazardSet, Labels: labels,
			Threshold: question.Threshold, Truncate: question.Truncate, RequireFullInput: question.RequireFullInput,
		}}, implementationTrained, true
	}
	if !card.Answers("noul") {
		return nil, "", false
	}
	questions := make([]Question, len(question.Labels))
	for i, label := range question.Labels {
		description := label.Description
		if description == "" {
			description = label.Key
		}
		questions[i] = Question{
			ID: trainedHazardNoulPrefix + label.Key, Type: "noul", Truncate: question.Truncate, RequireFullInput: question.RequireFullInput,
			Instructions: fmt.Sprintf(trainedHazardNoulQuestion, description), Choices: slices.Clone(trainedHazardNoulOutcomes),
		}
	}
	return questions, implementationTrainedNoul, true
}

// sameQuestion reports whether a consumer asks a task's canonical question:
// its type, wording and options, in order.
func sameQuestion(asked, canonical Question) bool {
	return asked.Type == canonical.Type && asked.Instructions == canonical.Instructions &&
		slices.Equal(asked.Choices, canonical.Choices) && slices.Equal(asked.Levels, canonical.Levels) &&
		slices.Equal(asked.Labels, canonical.Labels)
}

// sameWireQuestion reports whether two questions under one ID ask the same
// thing, so one answer serves both.
func sameWireQuestion(a, b Question) bool {
	return sameQuestion(a, b) && a.Head == b.Head && a.Preset == b.Preset && a.Truncate == b.Truncate &&
		a.RequireFullInput == b.RequireFullInput && equalThreshold(a.Threshold, b.Threshold)
}

func equalThreshold(a, b *float64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}
