package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func familyCard(family string, types ...string) ModelCard {
	return ModelCard{ID: family + "-model", Family: family, Surfaces: []string{"decisions"}, QuestionTypes: types}
}

// The cards of the families the presets bind, and of families they do not.
var presetFamilyCards = map[string]ModelCard{
	"vela3":     familyCard("vela3", "choice", "noul", "span"),
	"vela3_d3":  familyCard("vela3_d3", "choice", "noul", "score", "span"),
	"vela2":     familyCard("vela2", "choice", "noul", "score", "set", "span"),
	"decision2": familyCard("decision2", "choice", "noul", "score"),
	"decision3": familyCard("decision3", "choice", "noul", "score"),
	"unknown":   familyCard("", "choice", "noul", "score"),
}

// A preset-backed built-in question goes out under the ID its family was
// trained on (the vela3_d3 sizes turn their adapter on by ID), in the trained
// wording, which is the canonical wording; Vela 2.0 was trained on the
// canonical question under the Router's own ID; every other family answers it
// zero-shot under the Router's ID. The answer comes back under the Router's ID.
func TestPresetTasksGoOutInTheTrainedFormPerFamily(t *testing.T) {
	trainedIDs := map[string]string{
		"domain": "domain", "fact_check": "factcheck", "user_feedback": "feedback",
		"modality": "modality", "jailbreak": "attack", "safety": "p_harm",
	}
	vela2Trained := map[string]bool{"domain": true, "fact_check": true, "user_feedback": true, "jailbreak": true, "safety": true}
	for family, card := range presetFamilyCards {
		for task, trainedID := range trainedIDs {
			definition, _ := BuiltinTask(task)
			question := definition.Question
			question.ID = "consumer:" + task
			plan, err := CompileTask(definition, question, card)
			if err != nil {
				t.Fatalf("%s/%s: %v", family, task, err)
			}
			want, implementation := question.ID, implementationNative
			switch {
			case family == "vela3" || family == "vela3_d3":
				want, implementation = trainedID, implementationTrained
			case family == "vela2" && vela2Trained[task]:
				implementation = implementationTrained
			}
			asked := plan.Questions[0]
			if len(plan.Questions) != 1 || asked.ID != want || plan.Implementation != implementation || plan.Question.ID != question.ID {
				t.Fatalf("%s/%s: asked %+v (%s), want ID %q (%s)", family, task, plan.Questions, plan.Implementation, want, implementation)
			}
			if asked.Instructions != definition.Question.Instructions || strings.Contains(asked.ID, ":") == (want == trainedID) {
				t.Fatalf("%s/%s: the trained wording is the canonical wording, and only a Router ID has a ':': %+v", family, task, asked)
			}
			if asked.DecisionAdapter {
				t.Fatalf("%s/%s: a question a family was trained on, or a family without a decision adapter, takes no decision_adapter", family, task)
			}
			answered := reduceTaskAnswer(plan, Response{Answers: map[string]Answer{want: answerFor(asked)}})
			if answered.Error != "" || validateTaskAnswer(plan.Question, answered) != "" {
				t.Fatalf("%s/%s: the answer under %q is the plan's: %+v", family, task, want, answered)
			}
		}
	}
}

func answerFor(question Question) Answer {
	probabilities := map[string]float64{}
	for i, choice := range question.Choices {
		probabilities[choice.Key] = 0
		if i == 0 {
			probabilities[choice.Key] = 1
		}
	}
	return Answer{Type: question.Type, Choice: question.Choices[0].Key, Probabilities: probabilities, InputCoverage: "complete"}
}

// The modality question asks the preset's text, image and text_image; the
// Vela 3.0 families were trained on it with the AR, DIFFUSION and BOTH
// options, and their answer reads as the canonical options.
func TestTrainedModalityAnswersTheCanonicalOptions(t *testing.T) {
	definition, _ := BuiltinTask("modality")
	question := definition.Question
	question.ID = "modality_detector:modality"
	if keys := optionKeys(question.Choices); !slices.Equal(keys, []string{"text", "image", "text_image"}) {
		t.Fatalf("the modality task asks %v", keys)
	}
	for _, family := range []string{"vela3", "vela3_d3"} {
		plan, err := CompileTask(definition, question, presetFamilyCards[family])
		if err != nil {
			t.Fatal(err)
		}
		asked := plan.Questions[0]
		if asked.ID != "modality" || !slices.Equal(optionKeys(asked.Choices), []string{"AR", "DIFFUSION", "BOTH"}) || asked.Choices[1].Description != "a newly generated image only" {
			t.Fatalf("%s: trained modality question %+v", family, asked)
		}
		answer := reduceTaskAnswer(plan, Response{Answers: map[string]Answer{"modality": {
			Type: "choice", Choice: "DIFFUSION", Probabilities: map[string]float64{"AR": .2, "DIFFUSION": .7, "BOTH": .1},
		}}})
		if answer.Error != "" || answer.Choice != "image" || answer.Probabilities["image"] != .7 || answer.Probabilities["text"] != .2 || answer.Probabilities["text_image"] != .1 {
			t.Fatalf("%s: the trained answer reads as the canonical options: %+v", family, answer)
		}
	}
	plan, err := CompileTask(definition, question, presetFamilyCards["decision3"])
	if err != nil || plan.Questions[0].ID != question.ID || !slices.Equal(optionKeys(plan.Questions[0].Choices), []string{"text", "image", "text_image"}) {
		t.Fatalf("zero-shot modality question: %+v %v", plan, err)
	}
}

// A trained form answers the options it covers. A question that needs
// another one is asked in its canonical form, under the Router's ID: on a
// vela3_d3 size with the opt-in decision adapter, which a package without
// one ignores.
func TestUncoveredOptionsAskTheCanonicalQuestion(t *testing.T) {
	modality, _ := presets.Lookup(presets.Modality, presets.StageRequest)
	definition, _ := BuiltinTask("modality")
	question := PresetQuestion("systemone:wants_speech", modality, "image", "speech")
	for family, adapter := range map[string]bool{"vela3": false, "vela3_d3": true, "decision3": false} {
		plan, err := CompileTask(definition, question, presetFamilyCards[family])
		if err != nil {
			t.Fatal(err)
		}
		asked := plan.Questions[0]
		if plan.Implementation != implementationNative || asked.ID != question.ID || !slices.Equal(optionKeys(asked.Choices), []string{"image", "speech"}) || asked.DecisionAdapter != adapter {
			t.Fatalf("%s: %+v (%s)", family, asked, plan.Implementation)
		}
	}
}

// vela3-fast keeps the canonical options its thresholds were calibrated
// with; the vela3_d3 sizes get the NO_FEEDBACK description they were trained on.
func TestTrainedFeedbackOptionsFollowTheFamily(t *testing.T) {
	definition, _ := BuiltinTask("user_feedback")
	question := definition.Question
	question.ID = "feedback_detector:feedback"
	trained, _ := presets.Trained(presets.FamilyVela3D3, presets.Feedback, presets.StageRequest)
	longer := trained.Options[4].Description
	for family, want := range map[string]string{"vela3": "the user gives no feedback on the previous answer", "vela3_d3": longer} {
		plan, err := CompileTask(definition, question, familyCard(family, "choice", "noul"))
		if err != nil {
			t.Fatal(err)
		}
		index := slices.IndexFunc(plan.Questions[0].Choices, func(c Choice) bool { return c.Key == "NO_FEEDBACK" })
		if index < 0 || plan.Questions[0].Choices[index].Description != want {
			t.Fatalf("%s: NO_FEEDBACK %+v", family, plan.Questions[0].Choices)
		}
	}
	if definition.Question.Choices[4].Description == longer {
		t.Fatal("the canonical question is not changed by a family's trained form")
	}
}

// A consumer that specializes a built-in question (other options) asks it as
// written, under its own ID, on every family: on a vela3_d3 size with the
// decision adapter, since no adapter of it was trained on the question.
func TestSpecializedBuiltinQuestionsKeepTheRouterID(t *testing.T) {
	definition, _ := BuiltinTask("safety")
	question := definition.Question
	question.ID = "safety.custom:safety"
	question.Choices = []Choice{{"benign", "harmless"}, {"harmful", "harmful"}}
	plan, err := CompileTask(definition, question, familyCard("vela3_d3", "choice", "noul"))
	if err != nil || plan.Questions[0].ID != question.ID || plan.Implementation != implementationNative || !plan.Questions[0].DecisionAdapter {
		t.Fatalf("specialized question: %+v %v", plan, err)
	}
}

// Harm categories: vela3-fast is asked its trained 34-label Set when every
// label is one of them, a vela3_d3 size one trained hazard Noul per label,
// and every other model the generic composition.
func TestHarmCategoriesUseTheTrainedForm(t *testing.T) {
	definition, _ := BuiltinTask("safety_categories")
	categories, _ := presets.Lookup(presets.HarmCategories, presets.StageRequest)
	question := definition.Question
	question.ID = "safety.unsafe.hazard:safety_categories"
	question.Instructions = "Which harmful categories apply to this request?"
	question.Labels = []Choice{{"Violence", "Violence"}, {"S7", "S7"}}

	plan, err := CompileTask(definition, question, familyCard("vela3", "choice", "noul", "span"))
	if err != nil || plan.Implementation != implementationTrained || len(plan.Questions) != 1 {
		t.Fatalf("vela3: %+v %v", plan, err)
	}
	set := plan.Questions[0]
	if set.ID != "p_cats" || set.Type != "set" || set.Instructions != categories.Instructions || set.Labels[1].Description != "privacy violations, exposing personal information" {
		t.Fatalf("vela3 trained set: %+v", set)
	}
	answer := reduceTaskAnswer(plan, Response{Answers: map[string]Answer{"p_cats": {
		Type: "set", Probabilities: map[string]float64{"Violence": .9, "S7": .1}, Selected: []string{"Violence"}, InputCoverage: "complete",
	}}})
	if answer.Error != "" || answer.Probabilities["Violence"] != .9 {
		t.Fatalf("vela3 set answer: %+v", answer)
	}

	untrained := question
	untrained.Labels = []Choice{{"violence", "violence"}, {"privacy", "privacy"}}
	plan, err = CompileTask(definition, untrained, familyCard("vela3", "choice", "noul", "span"))
	if err != nil || plan.Implementation != implementationComposedNoul {
		t.Fatalf("vela3 with labels it was not trained on: %+v %v", plan, err)
	}

	plan, err = CompileTask(definition, untrained, familyCard("vela3_d3", "choice", "noul", "score"))
	if err != nil || plan.Implementation != implementationTrainedNoul || len(plan.Questions) != 2 {
		t.Fatalf("vela3_d3: %+v %v", plan, err)
	}
	for i, noul := range plan.Questions {
		label := untrained.Labels[i].Key
		if noul.ID != "hazard:"+label || noul.Type != "noul" || !strings.Contains(noul.Instructions, "does its harm involve "+label+"?") || len(noul.Choices) != 2 || noul.DecisionAdapter {
			t.Fatalf("vela3_d3 hazard noul %d: %+v", i, noul)
		}
	}
	answer = reduceTaskAnswer(plan, Response{Answers: map[string]Answer{
		"hazard:violence": {Type: "noul", Noul: .8, InputCoverage: "complete"},
		"hazard:privacy":  {Type: "noul", Noul: .2, InputCoverage: "complete"},
	}})
	if answer.Error != "" || !slices.Equal(answer.Selected, []string{"violence"}) {
		t.Fatalf("vela3_d3 hazard answer: %+v", answer)
	}

	plan, err = CompileTask(definition, untrained, familyCard("decision3", "choice", "noul", "score"))
	if err != nil || plan.Implementation != implementationComposedNoul || !strings.HasPrefix(plan.Questions[0].ID, question.ID+":label:") || plan.Questions[0].DecisionAdapter {
		t.Fatalf("decision3: %+v %v", plan, err)
	}
	plan, err = CompileTask(definition, question, familyCard("vela2", "choice", "noul", "score", "set", "span"))
	if err != nil || plan.Implementation != implementationNative || plan.Questions[0].ID != question.ID {
		t.Fatalf("vela2 answers a set natively, under the Router's ID: %+v %v", plan, err)
	}
}

// A span preset goes to the model's own preset question when the family was
// trained on it and the card defines it, else it is the canonical span
// question; a model that answers no span question cannot ask it.
func TestSpanPresetsAskTheModelPreset(t *testing.T) {
	definition, _ := BuiltinTask("pii_spans")
	question := definition.Question
	question.ID = "pii_classifier:pii_spans"
	withPreset := familyCard("vela3", "choice", "noul", "span")
	withPreset.Presets = []string{"pii", "halu"}
	plan, err := CompileTask(definition, question, withPreset)
	if err != nil || plan.Implementation != implementationTrained || plan.Questions[0].Preset != "pii" || plan.Questions[0].ID != question.ID {
		t.Fatalf("model preset: %+v %v", plan, err)
	}
	plan, err = CompileTask(definition, question, familyCard("vela3", "choice", "noul", "span"))
	if err != nil || plan.Implementation != implementationNative || plan.Questions[0].Type != "span" || len(plan.Questions[0].Labels) != len(PIICategoryLabels()) {
		t.Fatalf("canonical span question: %+v %v", plan, err)
	}
	if _, rejected := CompileTask(definition, question, presetFamilyCards["decision3"]); !errors.Is(rejected, ErrRejected) {
		t.Fatalf("a model without span answers: %v", rejected)
	}
	halu, _ := BuiltinTask("hallucination_spans")
	plan, err = CompileTask(halu, halu.Question, withPreset)
	if err != nil || plan.Questions[0].Preset != "halu" {
		t.Fatalf("halu preset: %+v %v", plan, err)
	}
}

// A custom systemone question no adapter was trained on goes out with
// decision_adapter on the vela3_d3 family only, as one question or as the
// Nouls a Set is composed of; a span question never takes it.
func TestDecisionAdapterMarksUntrainedQuestionsOnVela3D3(t *testing.T) {
	noul := Question{ID: "systemone:needs_tools", Type: "noul", Instructions: "Does this need a tool?"}
	set := Question{ID: "systemone:topics", Type: "set", Instructions: "Which topics?", Labels: []Choice{{"billing", ""}, {"shipping", ""}}}
	span := Question{ID: "systemone:accounts", Type: "span", Instructions: "Account numbers?", Labels: []Choice{{"account", ""}}}
	for family, card := range presetFamilyCards {
		response, err := ExecuteQuestions(context.Background(), taskDeciderFunc(func(_ context.Context, _ string, request Request) (Response, error) {
			for _, question := range request.Questions {
				adapter := family == "vela3_d3" && question.Type != "span"
				if question.DecisionAdapter != adapter {
					t.Errorf("%s: %s decision_adapter %v, want %v", family, question.ID, question.DecisionAdapter, adapter)
				}
			}
			return Response{Answers: map[string]Answer{}}, nil
		}), "d", card, Request{State: "text", Questions: questionsFor(card, noul, set, span)})
		if err != nil || len(response.Answers) == 0 {
			t.Fatalf("%s: %+v %v", family, response, err)
		}
	}
	encoded, err := json.Marshal(encodeQuestion(Question{ID: "q", Type: "noul", Instructions: "?", DecisionAdapter: true}))
	if err != nil || !strings.Contains(string(encoded), `"decision_adapter":true`) {
		t.Fatalf("encoded: %s %v", encoded, err)
	}
	if encoded, _ := json.Marshal(encodeQuestion(noul)); strings.Contains(string(encoded), "decision_adapter") {
		t.Fatalf("a question without the adapter leaves the field out: %s", encoded)
	}
}

func questionsFor(card ModelCard, questions ...Question) []Question {
	var answerable []Question
	for _, question := range questions {
		if question.Type != "span" || card.Answers("span") {
			answerable = append(answerable, question)
		}
	}
	return answerable
}

// Two consumers of one trained question in one call ask it once and both
// read its answer; two different questions under one ID are refused.
func TestPlansShareOneTrainedQuestion(t *testing.T) {
	definition, _ := BuiltinTask("safety")
	card := familyCard("vela3_d3", "choice", "noul")
	var plans []TaskPlan
	for _, consumer := range []string{"safety.a", "safety.b"} {
		question := definition.Question
		question.ID = consumer + ":p_harm"
		plan, err := CompileTask(definition, question, card)
		if err != nil {
			t.Fatal(err)
		}
		plans = append(plans, plan)
	}
	var asked []string
	decider := taskDeciderFunc(func(_ context.Context, _ string, request Request) (Response, error) {
		for _, question := range request.Questions {
			asked = append(asked, question.ID)
		}
		return Response{Answers: map[string]Answer{"p_harm": answerFor(request.Questions[0])}}, nil
	})
	response, results, err := ExecuteTaskPlans(context.Background(), decider, "lite", Request{State: "text"}, plans)
	if err != nil || !slices.Equal(asked, []string{"p_harm"}) {
		t.Fatalf("asked %v: %v", asked, err)
	}
	for _, consumer := range []string{"safety.a:p_harm", "safety.b:p_harm"} {
		if results[consumer].Status != "ok" || response.Answers[consumer].Choice != "safe" {
			t.Fatalf("%s: %+v %+v", consumer, results[consumer], response.Answers[consumer])
		}
	}

	conflicting := plans[1]
	conflicting.Questions = []Question{plans[1].Questions[0]}
	conflicting.Questions[0].Instructions = "Is this request dangerous?"
	if _, _, err := ExecuteTaskPlans(context.Background(), decider, "lite", Request{State: "text"}, []TaskPlan{plans[0], conflicting}); !errors.Is(err, ErrRejected) {
		t.Fatalf("two different questions under one ID: %v", err)
	}
}

// Images and videos travel with the state every question of it reads, and
// keep calls about the same text with other media apart.
func TestMediaTravelWithTheirState(t *testing.T) {
	body, err := encodeDecisionRequest(context.Background(), Request{State: "describe", Images: []string{"data:image/png;base64,AAAA"}, Videos: []string{"data:video/mp4;base64,BBBB"}})
	if err != nil || body.Images == nil || (*body.Images)[0] != "data:image/png;base64,AAAA" || body.Videos == nil || len(*body.Videos) != 1 {
		t.Fatalf("encoded: %+v %v", body, err)
	}
	if body, _ := encodeDecisionRequest(context.Background(), Request{State: "describe"}); body.Images != nil || body.Videos != nil {
		t.Fatal("a state without media carries none")
	}
	text := Request{State: "describe"}
	image := Request{State: "describe", Images: []string{"data:image/png;base64,AAAA"}}
	if stateText(text) == stateText(image) || decideKey(text) == decideKey(image) {
		t.Fatal("the same text with an image is another state")
	}
}
