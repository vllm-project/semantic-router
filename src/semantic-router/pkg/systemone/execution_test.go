package systemone

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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
		Quality: &config.NativeQualityConfig{Type: "uncalibrated", Acceptance: gate(0)},
		Stages:  []config.CascadeStage{{Name: "fast", Kind: "native", Model: "kai", Accept: gate(0.9)}, {Name: "strong", Kind: "native", Model: "nox"}},
	}
}

func policyIdentity(model string) InferenceIdentity {
	return InferenceIdentity{ModelID: model, Revision: strings.Repeat("a", 40), ModelSHA256: strings.Repeat("b", 64), Engine: "native", Profile: "exact", Numerics: "exact", Accelerator: "cpu"}
}

func policyResponse(body string) []byte {
	var response map[string]any
	_ = json.Unmarshal([]byte(body), &response)
	response["meta"] = policyIdentity(response["model"].(string))
	encoded, _ := json.Marshal(response)
	return encoded
}

func policyBindings(fast string) map[string]PolicyActionBinding {
	return map[string]PolicyActionBinding{
		"fast":   {Model: "kai", Identity: policyIdentity(fast)},
		"strong": {Model: "nox", Identity: policyIdentity("strong-actual")},
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

func TestPolicyUnresolvedAnswerCanReachExplicitJudge(t *testing.T) {
	plan := cascadePlan()
	plan.Type = config.DecisionAlgorithmPolicy
	plan.Stages = append(plan.Stages, config.CascadeStage{Name: "review", Kind: "judge", Model: "reviewer", Generation: &config.NativeGenerationConfig{MaxOutputTokens: 128}})
	weights := make([]float64, len(FeatureNames))
	weights[0] = -1 // No positive native upgrade is predicted.
	policy := LearnedPolicy{SchemaVersion: "systemone-policy/v1", FeatureNames: FeatureNames, Actions: policyBindings("fast-actual"), Heads: map[string]map[string]policyAction{"fast": {"strong": {Weights: weights, CostMS: 1}}}}
	data, _ := json.Marshal(policy)
	sum := sha256.Sum256(data)
	path := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	plan.Policy = &config.PolicyAlgorithmConfig{Source: path, SHA256: hex.EncodeToString(sum[:])}
	executor, err := NewExecutor(plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
	var calls []string
	result, err := executor.Execute(context.Background(), request, func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
		calls = append(calls, model)
		if model == "kai" {
			return 200, policyResponse(uncertainResponse), nil
		}
		return 200, []byte(`{"choices":[{"finish_reason":"stop","message":{"content":"{\"selected\":\"fast\"}"}}]}`), nil
	})
	if err != nil || !reflect.DeepEqual(calls, []string{"kai", "reviewer"}) || result.Stage != "review" || result.Model != "kai" {
		t.Fatalf("calls=%v result=%v error=%v", calls, result, err)
	}
}

func TestLearnedPolicyUsesDeclaredActionsAndFrozenBytes(t *testing.T) {
	plan := cascadePlan()
	plan.Type = config.DecisionAlgorithmPolicy
	plan.Stages[0].Accept = nil
	weights := make([]float64, len(FeatureNames))
	weights[0] = 1
	policy := LearnedPolicy{SchemaVersion: "systemone-policy/v1", FeatureNames: FeatureNames, Actions: policyBindings("strong-actual"), Heads: map[string]map[string]policyAction{"fast": {"strong": {Weights: weights, CostMS: 1}}}}
	data, _ := json.Marshal(policy)
	sum := sha256.Sum256(data)
	path := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	plan.Policy = &config.PolicyAlgorithmConfig{Source: path, SHA256: hex.EncodeToString(sum[:])}
	executor, err := NewExecutor(plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := os.WriteFile(path, []byte(`{"mutated":true}`), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
	var calls []string
	result, err := executor.Execute(context.Background(), request, func(_ context.Context, model string, body json.RawMessage) (int, []byte, error) {
		if !strings.Contains(string(body), `"return_meta":true`) {
			t.Fatal("learned policy failed to request runtime provenance")
		}
		calls = append(calls, model)
		return 200, policyResponse(certainResponse), nil
	})
	if err != nil || !reflect.DeepEqual(calls, []string{"kai", "nox"}) || result.Stage != "strong" {
		t.Fatalf("calls=%v result=%v error=%v", calls, result, err)
	}
	calls = nil
	_, err = executor.Execute(context.Background(), request, func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
		calls = append(calls, model)
		// A remote operator repointed the same provider alias to another model.
		return 200, policyResponse(uncertainResponse), nil
	})
	if err == nil || !reflect.DeepEqual(calls, []string{"kai"}) {
		t.Fatalf("policy used evidence from a changed model: calls=%v error=%v", calls, err)
	}
	if _, err := NewExecutor(plan, nil); err == nil {
		t.Fatal("modified policy bytes accepted")
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
