package classification

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// fakeSystemOne stands in for the remote API of a systemone deployment.
type fakeSystemOne struct {
	fakeDecider
	card  modelservice.ModelCard
	block bool
}

func (f *fakeSystemOne) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	if f.block {
		<-ctx.Done()
		return modelservice.Response{}, ctx.Err()
	}
	return f.fakeDecider.Decide(ctx, deployment, request)
}

func (f *fakeSystemOne) Card(deployment string) (modelservice.ModelCard, bool) {
	return f.card, deployment == "hosted-d3"
}

func (f *fakeSystemOne) PrepareCard(_ context.Context, deployment string) (modelservice.ModelCard, error) {
	if deployment != "hosted-d3" {
		return modelservice.ModelCard{}, modelservice.ErrUnknownDeployment
	}
	return f.card, nil
}

func newFakeSystemOne() *fakeSystemOne {
	return &fakeSystemOne{
		fakeDecider: fakeDecider{calls: map[string]int{}, requests: map[string]modelservice.Request{}, failures: map[string]error{}, answers: map[string]modelservice.Answer{
			"support_team": {Type: "choice", Choice: "engineering", Probabilities: map[string]float64{"billing": 0.1, "engineering": 0.9}, Confidence: 0.8},
			"refund":       {Type: "noul", Noul: 0.2},
			"urgency":      {Type: "score", Score: 1.5, Probabilities: map[string]float64{"0": 0.1, "1": 0.3, "2": 0.6}},
		}},
		card: modelservice.RemoteCard("vllm-sr/d3-ultra", nil, nil),
	}
}

// systemOneTestClassifier asks the systemone decision model hosted-d3 three
// questions, one without a deployment, and a model_runtime deployment one.
func systemOneTestClassifier(local modelservice.Decider, remote modelservice.Remote, ruleTimeoutMs, deploymentTimeoutMs int) *Classifier {
	level := 1.0
	cfg := &config.RouterConfig{}
	cfg.DecisionModel = "hosted-d3"
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"hosted-d3": {Provider: config.SystemOneProvider, Endpoint: "https://systemone.example.com/v1/systemone", Model: "vllm-sr/d3-ultra", TimeoutMs: deploymentTimeoutMs},
		"kai":       {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1"},
	}
	cfg.DecisionRules = []config.DecisionSignalRule{
		{Name: "support_team", Deployment: "hosted-d3", TimeoutMs: ruleTimeoutMs, Question: config.DecisionQuestion{Type: "choice", Instructions: "Which team?", Choices: []config.DecisionChoice{{Key: "billing"}, {Key: "engineering"}}}},
		{Name: "refund", TimeoutMs: ruleTimeoutMs, Question: config.DecisionQuestion{Type: "noul", Instructions: "A refund?"}},
		{Name: "urgency", Deployment: "hosted-d3", TimeoutMs: ruleTimeoutMs, Predicate: &config.NumericPredicate{GTE: &level}, Question: config.DecisionQuestion{Type: "score", Instructions: "How urgent?", Levels: []string{"low", "medium", "high"}}},
		{Name: "local", Deployment: "kai", Question: config.DecisionQuestion{Type: "noul", Instructions: "Local?"}},
	}
	classifier := &Classifier{Config: cfg}
	classifier.SetDecisionDecider(local)
	if remote != nil {
		classifier.SetSystemOneDecider(remote)
	}
	return classifier
}

func allDecisionRulesUsed(classifier *Classifier) map[string]bool {
	used := map[string]bool{}
	for _, rule := range classifier.Config.DecisionRules {
		used["decision:"+rule.Name] = true
	}
	return used
}

func newLocalDecider() *fakeDecider {
	return &fakeDecider{calls: map[string]int{}, requests: map[string]modelservice.Request{}, answers: map[string]modelservice.Answer{"local": {Type: "noul", Noul: 0.9}}}
}

func TestSystemOneDeploymentAnswersSignalsAsAModelRuntimeOneDoes(t *testing.T) {
	local, remote := newLocalDecider(), newFakeSystemOne()
	classifier := systemOneTestClassifier(local, remote, 3000, 0)
	results := newSignalResults()
	classifier.evaluateDecisionModelSignals(context.Background(), results, &sync.Mutex{}, "refund my invoice", "refund my invoice", allDecisionRulesUsed(classifier))

	if remote.calls["hosted-d3"] != 1 || len(remote.requests["hosted-d3"].Questions) != 3 || remote.requests["hosted-d3"].State != "refund my invoice" {
		t.Fatalf("one remote call with every question to hosted-d3, got %v %+v", remote.calls, remote.requests["hosted-d3"])
	}
	if local.calls["kai"] != 1 || local.calls["hosted-d3"] != 0 {
		t.Fatalf("model_runtime questions stay with the model runtime: %v", local.calls)
	}
	for _, want := range []string{"support_team:engineering", "urgency", "local"} {
		if !slices.Contains(results.MatchedDecisionRules, want) {
			t.Fatalf("missing match %q in %v", want, results.MatchedDecisionRules)
		}
	}
	if slices.Contains(results.MatchedDecisionRules, "refund") {
		t.Fatalf("a noul below the default threshold does not match: %v", results.MatchedDecisionRules)
	}
	if results.SignalValues["decision:support_team:billing"] != 0.1 || results.SignalConfidences["decision:support_team"] != 0.9 ||
		results.SignalValues["decision:refund"] != 0.2 || results.SignalValues["decision:urgency"] != 1.5 || results.SignalConfidences["decision:urgency"] != 0.6 {
		t.Fatalf("values %v confidences %v", results.SignalValues, results.SignalConfidences)
	}
	if len(results.SignalErrors) != 0 {
		t.Fatalf("answered signals carry no error: %v", results.SignalErrors)
	}
}

func TestSystemOneDeploymentFailuresLeaveItsSignalsUnknown(t *testing.T) {
	evaluate := func(remote modelservice.Remote, ruleTimeoutMs, deploymentTimeoutMs int) (*SignalResults, time.Duration) {
		classifier := systemOneTestClassifier(newLocalDecider(), remote, ruleTimeoutMs, deploymentTimeoutMs)
		results := newSignalResults()
		started := time.Now()
		classifier.evaluateDecisionModelSignals(context.Background(), results, &sync.Mutex{}, "x", "x", allDecisionRulesUsed(classifier))
		return results, time.Since(started)
	}
	expectUnknown := func(results *SignalResults, reason string) {
		t.Helper()
		for _, name := range []string{"support_team", "refund", "urgency"} {
			if got := results.SignalErrors["decision:"+name]; got != reason {
				t.Fatalf("%s: %q, want %q (errors %v)", name, got, reason, results.SignalErrors)
			}
		}
		if !slices.Contains(results.MatchedDecisionRules, "local") {
			t.Fatalf("a failing remote does not affect model_runtime signals: %v", results.MatchedDecisionRules)
		}
	}

	results, _ := evaluate(nil, 3000, 0)
	expectUnknown(results, "decision_unavailable")

	rejected := newFakeSystemOne()
	rejected.failures["hosted-d3"] = modelservice.ErrRejected
	results, _ = evaluate(rejected, 3000, 0)
	expectUnknown(results, "decision_rejected")

	slow := newFakeSystemOne()
	slow.block = true
	results, took := evaluate(slow, 100, 0)
	expectUnknown(results, "decision_timeout")
	if took > 2*time.Second {
		t.Fatalf("a slow remote holds the stage only for its rules' timeout, took %s", took)
	}
	// Questions without a timeout take the deployment's timeout_ms, else the
	// default; a question to the decision model joins no built-in call.
	results, took = evaluate(slow, 0, 300)
	expectUnknown(results, "decision_timeout")
	if took < 250*time.Millisecond || took > 900*time.Millisecond {
		t.Fatalf("the deployment's timeout bounds its call, took %s", took)
	}
	results, took = evaluate(slow, 0, 0)
	expectUnknown(results, "decision_timeout")
	if took < time.Duration(config.DefaultDecisionTimeoutMs)*time.Millisecond/2 || took > 3*time.Second {
		t.Fatalf("the default timeout bounds a remote call, took %s", took)
	}
}
