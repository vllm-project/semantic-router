package modelservice

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

type taskDeciderFunc func(context.Context, string, Request) (Response, error)

func (f taskDeciderFunc) Decide(ctx context.Context, deployment string, request Request) (Response, error) {
	return f(ctx, deployment, request)
}

func TestCompileDecisionTaskUsesStructureNotFamily(t *testing.T) {
	for _, family := range []string{"decision1", "decision2", "vela2", "future-provider"} {
		card := ModelCard{ID: family, Family: family, Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice", "noul", "score"}}
		for _, task := range BuiltinTasks() {
			capability := CapabilityForTask(task, card)
			if capability.Supported != (task.Output != "span") {
				t.Fatalf("%s/%s: %+v", family, task.ID, capability)
			}
			if task.Output == "set" && capability.Implementation != "composed_noul" {
				t.Fatalf("%s/%s must expose composition", family, task.ID)
			}
		}
		if card.Answers("set") || card.Answers("span") {
			t.Fatal("task composition changed native capabilities")
		}
	}
}

func TestComposedDecisionTaskBatchesLabelsAndRetainsUnknown(t *testing.T) {
	definition, _ := BuiltinTask("pii_categories")
	definition.Question.Labels = []Choice{{"name", "a person's name"}, {"email", "an email address"}}
	card := ModelCard{ID: "decision", Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul"}}
	plan, err := CompileTask(definition, definition.Question, card)
	if err != nil {
		t.Fatal(err)
	}
	for _, partial := range []bool{false, true} {
		calls := 0
		decider := taskDeciderFunc(func(_ context.Context, deployment string, request Request) (Response, error) {
			calls++
			if deployment != "selected" || request.State != "Alex" || len(request.Questions) != 2 || request.Questions[0].Type != "noul" {
				t.Fatalf("unexpected execution: %s %+v", deployment, request)
			}
			answers := map[string]Answer{request.Questions[0].ID: {Type: "noul", Noul: .9}}
			if !partial {
				answers[request.Questions[1].ID] = Answer{Type: "noul", Noul: .1}
			}
			return Response{Answers: answers}, nil
		})
		response, results, err := ExecuteTaskPlans(context.Background(), decider, "selected", Request{State: "Alex"}, []TaskPlan{plan})
		if err != nil || calls != 1 {
			t.Fatalf("calls=%d error=%v", calls, err)
		}
		result, answer := results[plan.Question.ID], response.Answers[plan.Question.ID]
		if partial {
			if result.Status != "unknown" || result.Coverage != "unknown" || answer.Error == "" {
				t.Fatalf("missing label incorrectly certified complete: %+v", result)
			}
		} else if result.Status != "ok" || len(answer.Selected) != 1 || answer.Selected[0] != "name" || answer.Probabilities["email"] != .1 {
			t.Fatalf("bad set reduction: %+v", result)
		}
	}
}

func TestComposedDecisionTaskDeadlineAndDuplicateIDs(t *testing.T) {
	definition, _ := BuiltinTask("reask")
	plan, _ := CompileTask(definition, definition.Question, ModelCard{Surfaces: []string{"decisions"}})
	decider := taskDeciderFunc(func(context.Context, string, Request) (Response, error) { return Response{}, context.DeadlineExceeded })
	_, results, err := ExecuteTaskPlans(context.Background(), decider, "selected", Request{State: "request"}, []TaskPlan{plan})
	if !errors.Is(err, context.DeadlineExceeded) || results[plan.Question.ID].Status != "error" || results[plan.Question.ID].Coverage != "unknown" {
		t.Fatalf("deadline became a negative result: %+v %v", results, err)
	}
	_, _, err = ExecuteTaskPlans(context.Background(), decider, "selected", Request{State: "request"}, []TaskPlan{plan, plan})
	if !errors.Is(err, ErrRejected) {
		t.Fatalf("duplicate compiled IDs accepted: %v", err)
	}
}

func TestTaskOrdinalScoreAndUnknownValues(t *testing.T) {
	definition, _ := BuiltinTask("complexity")
	plan, _ := CompileTask(definition, definition.Question, ModelCard{Surfaces: []string{"decisions"}})
	for _, value := range []float64{0, 1.5, 2} {
		response := Response{Answers: map[string]Answer{plan.Question.ID: {Type: "score", Score: value}}}
		if got := reduceTaskAnswer(plan, response); got.Error != "" || got.Score != value {
			t.Fatalf("valid ordinal %v rejected: %+v", value, got)
		}
	}
	definition, _ = BuiltinTask("pii_presence")
	plan, _ = CompileTask(definition, definition.Question, ModelCard{Surfaces: []string{"decisions"}})
	for _, code := range []string{"unknown", "missing_answer_value", "max_length_exceeded"} {
		decider := taskDeciderFunc(func(context.Context, string, Request) (Response, error) {
			return Response{Answers: map[string]Answer{plan.Question.ID: {Type: "noul", Error: code}}}, nil
		})
		_, results, err := ExecuteTaskPlans(t.Context(), decider, "selected", Request{State: "private"}, []TaskPlan{plan})
		if err != nil || results[plan.Question.ID].Status != "unknown" || results[plan.Question.ID].Coverage != "unknown" {
			t.Fatalf("%s certified input: %+v", code, results)
		}
	}
}

func TestChoiceTaskRejectsMalformedDistributionBeforeSuccessMetrics(t *testing.T) {
	definition, _ := BuiltinTask("preference")
	plan, err := CompileTask(definition, definition.Question, ModelCard{Surfaces: []string{"decisions"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name          string
		probabilities map[string]float64
	}{
		{"missing", nil},
		{"partial", map[string]float64{"concise": 1}},
		{"unknown key", map[string]float64{"concise": .8, "invented": .2}},
		{"nan", map[string]float64{"concise": math.NaN(), "detailed": .2}},
		{"infinite", map[string]float64{"concise": math.Inf(1), "detailed": .2}},
		{"negative", map[string]float64{"concise": 1.1, "detailed": -.1}},
		{"unnormalized", map[string]float64{"concise": .8, "detailed": .4}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			deployment := "malformed-choice-" + tc.name
			decider := taskDeciderFunc(func(context.Context, string, Request) (Response, error) {
				return Response{Answers: map[string]Answer{plan.Question.ID: {Type: "choice", Choice: "concise", Probabilities: tc.probabilities}}}, nil
			})
			_, results, err := ExecuteTaskPlans(t.Context(), decider, deployment, Request{State: "Be brief"}, []TaskPlan{plan})
			if err != nil || results[plan.Question.ID].Status != "error" || results[plan.Question.ID].Coverage != "unknown" {
				t.Fatalf("malformed distribution certified: %+v %v", results, err)
			}
			if got := testutil.ToFloat64(taskCalls.WithLabelValues(deployment, "preference", "request", "native", "error")); got != 1 {
				t.Fatalf("failed task count = %v", got)
			}
			if got := testutil.ToFloat64(taskResults.WithLabelValues(deployment, "preference", "request", "native", "choice:concise")); got != 0 {
				t.Fatalf("malformed answer recorded as successful result = %v", got)
			}
		})
	}
	for _, probabilities := range []map[string]float64{{"concise": .8, "detailed": .2}, {"concise": .8, "detailed": .20001}} {
		if code := validateTaskAnswer(plan.Question, Answer{Type: "choice", Choice: "concise", Probabilities: probabilities}); code != "" {
			t.Fatalf("native floating-point distribution rejected: %s", code)
		}
	}
}

func TestSetTaskRejectsInventedAndDuplicateSelectedLabels(t *testing.T) {
	question := Question{Type: "set", Labels: []Choice{{Key: "name"}, {Key: "email"}}}
	for _, selected := range [][]string{{"invented"}, {"name", "name"}} {
		if code := validateTaskAnswer(question, Answer{Type: "set", Selected: selected, Probabilities: map[string]float64{"name": .9, "email": .1}}); code != "invalid_selected_labels" {
			t.Fatalf("invalid selected labels accepted: %v (%s)", selected, code)
		}
	}
	if code := validateTaskAnswer(Question{Preset: "labels"}, Answer{Type: "set", Selected: []string{"name"}, Probabilities: map[string]float64{"name": .9}}); code != "" {
		t.Fatalf("valid preset-owned labels rejected: %s", code)
	}
}

func TestComposedSetMatchesNativeStrictThreshold(t *testing.T) {
	definition, _ := BuiltinTask("pii_categories")
	definition.Question.Labels = []Choice{{Key: "equal"}, {Key: "above"}}
	plan, err := CompileTask(definition, definition.Question, ModelCard{Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul"}})
	if err != nil {
		t.Fatal(err)
	}
	answer := reduceTaskAnswer(plan, Response{Answers: map[string]Answer{
		plan.Questions[0].ID: {Type: "noul", Noul: .5},
		plan.Questions[1].ID: {Type: "noul", Noul: .5001},
	}})
	if answer.Error != "" || len(answer.Selected) != 1 || answer.Selected[0] != "above" {
		t.Fatalf("composed selection differs from native p > threshold: %+v", answer)
	}
}
