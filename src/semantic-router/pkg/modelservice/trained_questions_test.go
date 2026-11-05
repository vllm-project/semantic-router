package modelservice

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

func familyCard(family string, types ...string) ModelCard {
	return ModelCard{ID: family + "-model", Family: family, Surfaces: []string{"decisions"}, QuestionTypes: types}
}

// The Vela 3.0 packages are asked a built-in question under the ID they were
// trained on (vela3_d3 turns its adapter on by ID); every other family keeps
// the Router's own ID. The answer comes back under the Router's ID.
func TestBuiltinQuestionsGoOutUnderTheTrainedIDPerFamily(t *testing.T) {
	trained := map[string]string{
		"domain": "domain", "fact_check": "factcheck", "user_feedback": "feedback",
		"modality": "modality", "jailbreak": "attack", "safety": "p_harm",
	}
	cards := map[string]ModelCard{
		"vela3":     familyCard("vela3", "choice", "noul", "span"),
		"vela3_d3":  familyCard("vela3_d3", "choice", "noul", "score", "span"),
		"vela2":     familyCard("vela2", "choice", "noul", "score", "set", "span"),
		"decision2": familyCard("decision2", "choice", "noul", "score"),
		"decision3": familyCard("decision3", "choice", "noul", "score"),
	}
	for family, card := range cards {
		for task, id := range trained {
			definition, _ := BuiltinTask(task)
			question := definition.Question
			question.ID = "consumer:" + task
			plan, err := CompileTask(definition, question, card)
			if err != nil {
				t.Fatalf("%s/%s: %v", family, task, err)
			}
			want, implementation := question.ID, implementationNative
			if family == "vela3" || family == "vela3_d3" {
				want, implementation = id, implementationTrained
			}
			if len(plan.Questions) != 1 || plan.Questions[0].ID != want || plan.Implementation != implementation || plan.Question.ID != question.ID {
				t.Fatalf("%s/%s: asked %+v (%s), want ID %q", family, task, plan.Questions, plan.Implementation, want)
			}
			if plan.Questions[0].Instructions != definition.Question.Instructions {
				t.Fatalf("%s/%s: the trained wording is the built-in wording: %q", family, task, plan.Questions[0].Instructions)
			}
			answered := reduceTaskAnswer(plan, Response{Answers: map[string]Answer{want: answerFor(plan.Questions[0])}})
			if answered.Error != "" {
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

// vela3-fast keeps the built-in options its thresholds were calibrated with;
// the vela3_d3 sizes get the NO_FEEDBACK description they were trained on.
func TestTrainedFeedbackOptionsFollowTheFamily(t *testing.T) {
	definition, _ := BuiltinTask("user_feedback")
	question := definition.Question
	question.ID = "feedback_detector:feedback"
	for family, want := range map[string]string{"vela3": "the user gives no feedback on the previous answer", "vela3_d3": trainedFeedbackNone} {
		plan, err := CompileTask(definition, question, familyCard(family, "choice", "noul"))
		if err != nil {
			t.Fatal(err)
		}
		index := slices.IndexFunc(plan.Questions[0].Choices, func(c Choice) bool { return c.Key == "NO_FEEDBACK" })
		if index < 0 || plan.Questions[0].Choices[index].Description != want {
			t.Fatalf("%s: NO_FEEDBACK %+v", family, plan.Questions[0].Choices)
		}
	}
	if definition.Question.Choices[4].Description == trainedFeedbackNone {
		t.Fatal("the canonical question is not changed by a family's trained form")
	}
}

// A consumer that specializes a built-in question (other options) asks it as
// written, under its own ID, on every family.
func TestSpecializedBuiltinQuestionsKeepTheRouterID(t *testing.T) {
	definition, _ := BuiltinTask("safety")
	question := definition.Question
	question.ID = "safety.custom:safety"
	question.Choices = []Choice{{"benign", "harmless"}, {"harmful", "harmful"}}
	plan, err := CompileTask(definition, question, familyCard("vela3_d3", "choice", "noul"))
	if err != nil || plan.Questions[0].ID != question.ID || plan.Implementation != implementationNative {
		t.Fatalf("specialized question: %+v %v", plan, err)
	}
}

// Harm categories: vela3-fast is asked its trained 34-label Set when every
// label is one of them, a vela3_d3 size one trained hazard Noul per label,
// and every other model the generic composition.
func TestHarmCategoriesUseTheTrainedForm(t *testing.T) {
	definition, _ := BuiltinTask("safety_categories")
	question := definition.Question
	question.ID = "safety.unsafe.hazard:safety_categories"
	question.Labels = []Choice{{"Violence", "Violence"}, {"S7", "S7"}}

	plan, err := CompileTask(definition, question, familyCard("vela3", "choice", "noul", "span"))
	if err != nil || plan.Implementation != implementationTrained || len(plan.Questions) != 1 {
		t.Fatalf("vela3: %+v %v", plan, err)
	}
	set := plan.Questions[0]
	if set.ID != trainedHazardSetID || set.Type != "set" || set.Instructions != trainedHazardSet || set.Labels[1].Description != trainedHazardCategories["S7"] {
		t.Fatalf("vela3 trained set: %+v", set)
	}
	answer := reduceTaskAnswer(plan, Response{Answers: map[string]Answer{trainedHazardSetID: {
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
		if noul.ID != trainedHazardNoulPrefix+label || noul.Type != "noul" || !strings.Contains(noul.Instructions, "does its harm involve "+label+"?") || len(noul.Choices) != 2 {
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
	if err != nil || plan.Implementation != implementationComposedNoul || !strings.HasPrefix(plan.Questions[0].ID, question.ID+":label:") {
		t.Fatalf("decision3: %+v %v", plan, err)
	}
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
