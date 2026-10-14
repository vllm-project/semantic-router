package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

const (
	singleRequest     = `{"model":"vllm-sr/auto","state":"hello","questions":{"task":{"type":"choice","instructions":"Type?","criteria":{"code":"Code","chat":"Chat"}}}}`
	uncertainResponse = `{"model":"fast-actual","answers":{"task":{"type":"choice","choice":"chat","confidence":0.01,"probabilities":{"code":0.4,"chat":0.6}}},"usage":{"input_tokens":11,"output_tokens":0},"extension":{"keep":true}}`
	certainResponse   = `{"model":"strong-actual","answers":{"task":{"type":"choice","choice":"chat","confidence":0.99,"probabilities":{"code":0.001,"chat":0.999}}},"usage":{"input_tokens":11,"output_tokens":0}}`
)

func gate(threshold float64) *config.NativeAcceptance {
	return &config.NativeAcceptance{Rules: []config.NativeAcceptanceRule{{QuestionType: "choice", Field: "top_probability", Predicate: config.NumericPredicate{GTE: &threshold}}}}
}

func cascadePlan() *config.AlgorithmConfig {
	return &config.AlgorithmConfig{
		Type:    config.DecisionAlgorithmCascade,
		Budget:  &config.AlgorithmBudget{Deadline: "3s", MaxCalls: 3},
		Quality: &config.NativeQualityConfig{Type: "uncalibrated", Acceptance: gate(0)},
		Stages:  []config.CascadeStage{{Name: "fast", Kind: "native", Model: "kai", Accept: gate(0.9)}, {Name: "strong", Kind: "native", Model: "nox"}},
	}
}

func TestCascadeFastExitEscalationAndSharedBudget(t *testing.T) {
	request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
	for _, tc := range []struct {
		name, fast string
		limit      int
		want       []string
		unresolved bool
	}{
		{"fast", certainResponse, 2, []string{"kai"}, false},
		{"escalate", uncertainResponse, 2, []string{"kai", "nox"}, false},
		{"budget", uncertainResponse, 1, []string{"kai"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			executor, err := NewExecutor(cascadePlan(), nil)
			if err != nil {
				t.Fatal(err)
			}
			ctx, ledger := budget.WithLimit(context.Background(), tc.limit)
			var calls []string
			invoke := func(ctx context.Context, model string, body json.RawMessage) (int, []byte, error) {
				if budgetErr := budget.Consume(ctx); budgetErr != nil {
					return 0, nil, budgetErr
				}
				calls = append(calls, model)
				if string(body) != singleRequest {
					t.Fatal("native task rewritten")
				}
				if model == "kai" {
					return 200, []byte(tc.fast), nil
				}
				return 200, []byte(certainResponse), nil
			}
			result, err := executor.Execute(ctx, request, invoke)
			if tc.unresolved != errors.Is(err, ErrUnresolved) || (!tc.unresolved && err != nil) {
				t.Fatalf("result=%v error=%v", result, err)
			}
			if !reflect.DeepEqual(calls, tc.want) || ledger.Used() != len(tc.want) {
				t.Fatalf("calls=%v ledger=%d", calls, ledger.Used())
			}
		})
	}
}

func TestJudgeSelectsAnIntactBundleWithoutInventingConfidence(t *testing.T) {
	plan := cascadePlan()
	plan.Stages[1].Accept = gate(1)
	plan.Stages = append(plan.Stages, config.CascadeStage{Name: "review", Kind: "judge", Model: "reviewer", Generation: &config.NativeGenerationConfig{MaxOutputTokens: 128}})
	executor, err := NewExecutor(plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
	result, err := executor.Execute(context.Background(), request, func(_ context.Context, model string, body json.RawMessage) (int, []byte, error) {
		switch model {
		case "kai":
			return 200, []byte(uncertainResponse), nil
		case "nox":
			return 200, []byte(certainResponse), nil
		}
		var prompt map[string]json.RawMessage
		if json.Unmarshal(body, &prompt) != nil || len(prompt["response_format"]) == 0 {
			t.Fatal("unconstrained judge")
		}
		return 200, []byte(`{"choices":[{"finish_reason":"stop","message":{"content":"{\"selected\":\"fast\"}"}}]}`), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Stage != "review" || result.Model != "kai" || string(result.Body) != uncertainResponse {
		t.Fatalf("judge changed native evidence: %+v", result)
	}
	if len(result.History) != 3 || result.History[2].Model != "reviewer" || string(result.History[2].Response) == string(result.Body) {
		t.Fatal("judge invocation was confused with the selected native response")
	}
}

func TestCanceledCascadeDoesNotLaunchAnotherStage(t *testing.T) {
	executor, _ := NewExecutor(cascadePlan(), nil)
	request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	_, err := executor.Execute(ctx, request, func(context.Context, string, json.RawMessage) (int, []byte, error) {
		calls++
		cancel()
		return 0, nil, context.Canceled
	})
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("calls=%d error=%v", calls, err)
	}
}

const judgeFastResponse = `{"choices":[{"finish_reason":"stop","message":{"content":"{\"selected\":\"fast\"}"}}]}`

func TestCascadeRejectsSuccessfulResponseAfterCancellation(t *testing.T) {
	for _, kind := range []string{"native", "judge", "quality"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			plan := cascadePlan()
			var quality QualityEvaluator
			wantCalls := 1
			switch kind {
			case "judge":
				plan.Stages[1].Accept = gate(1)
				plan.Stages = append(plan.Stages, config.CascadeStage{Name: "review", Kind: "judge", Model: "reviewer", Generation: &config.NativeGenerationConfig{MaxOutputTokens: 128}})
				wantCalls = 3
			case "quality":
				plan.Quality.Type = "calibrated"
				quality = func(*NativeRequest, Candidate) (bool, error) {
					cancel()
					return true, nil
				}
			}
			executor, err := NewExecutor(plan, quality)
			if err != nil {
				t.Fatal(err)
			}
			request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
			calls := 0
			result, err := executor.Execute(ctx, request, func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
				calls++
				if kind == "native" || model == "reviewer" {
					cancel()
				}
				if model == "reviewer" {
					return 200, []byte(judgeFastResponse), nil
				}
				if kind == "judge" && model == "kai" {
					return 200, []byte(uncertainResponse), nil
				}
				return 200, []byte(certainResponse), nil
			})
			if !errors.Is(err, context.Canceled) || result.Body != nil || calls != wantCalls {
				t.Fatalf("accepted canceled execution: stage=%q calls=%d error=%v", result.Stage, calls, err)
			}
		})
	}
}

func TestCascadeStageTimeoutRejectsLateResponseAndAllowsUpgrade(t *testing.T) {
	for _, kind := range []string{"native", "judge"} {
		t.Run(kind, func(t *testing.T) {
			plan := cascadePlan()
			lateModel, wantModel := "kai", "nox"
			wantCalls := 2
			if kind == "native" {
				plan.Stages[0].Timeout = "1ms"
			} else {
				plan.Stages[0].Accept, plan.Stages[1].Accept = gate(1), gate(1)
				plan.Stages = append(plan.Stages,
					config.CascadeStage{Name: "review", Kind: "judge", Model: "reviewer", Timeout: "1ms", Generation: &config.NativeGenerationConfig{MaxOutputTokens: 128}},
					config.CascadeStage{Name: "final", Kind: "native", Model: "vega"})
				lateModel, wantModel, wantCalls = "reviewer", "vega", 4
			}
			executor, err := NewExecutor(plan, nil)
			if err != nil {
				t.Fatal(err)
			}
			request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
			calls := 0
			result, err := executor.Execute(t.Context(), request, func(ctx context.Context, model string, _ json.RawMessage) (int, []byte, error) {
				calls++
				if model == lateModel {
					<-ctx.Done()
				}
				if model == "reviewer" {
					return 200, []byte(judgeFastResponse), nil
				}
				return 200, []byte(certainResponse), nil
			})
			if err != nil || result.Model != wantModel || calls != wantCalls {
				t.Fatalf("late stage was accepted: model=%q calls=%d error=%v", result.Model, calls, err)
			}
		})
	}
}
