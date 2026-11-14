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
			"systemone:support_team": {Type: "choice", Choice: "engineering", Probabilities: map[string]float64{"billing": 0.1, "engineering": 0.9}, Confidence: 0.8},
			"systemone:refund":       {Type: "noul", Noul: 0.2},
			"systemone:urgency":      {Type: "score", Score: 1.5, Probabilities: map[string]float64{"0": 0.1, "1": 0.3, "2": 0.6}},
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
		"hosted-d3": {Provider: config.SystemOneProvider, Schema: "vllm-sr/4.0", Endpoint: "https://systemone.example.com/v1/systemone", Model: "vllm-sr/d3-ultra", TimeoutMs: deploymentTimeoutMs},
		"kai":       {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1"},
	}
	cfg.SystemOneRules = []config.SystemOneRule{
		{Name: "support_team", Deployment: "hosted-d3", TimeoutMs: ruleTimeoutMs, Question: config.SystemOneQuestion{Type: "choice", Instructions: "Which team?", Choices: []config.SystemOneOption{{Key: "billing"}, {Key: "engineering"}}}},
		{Name: "refund", TimeoutMs: ruleTimeoutMs, Question: config.SystemOneQuestion{Type: "noul", Instructions: "A refund?"}},
		{Name: "urgency", Deployment: "hosted-d3", TimeoutMs: ruleTimeoutMs, Predicate: &config.NumericPredicate{GTE: &level}, Question: config.SystemOneQuestion{Type: "score", Instructions: "How urgent?", Levels: []string{"low", "medium", "high"}}},
		{Name: "local", Deployment: "kai", Question: config.SystemOneQuestion{Type: "noul", Instructions: "Local?"}},
	}
	classifier := &Classifier{Config: cfg}
	classifier.SetDecisionDecider(local)
	if remote != nil {
		classifier.SetSystemOneDecider(remote)
	}
	return classifier
}

func newLocalDecider() *fakeDecider {
	return newFakeDecider(map[string]modelservice.Answer{"systemone:local": {Type: "noul", Noul: 0.9}})
}

func TestSystemOneDeploymentAnswersSignalsAsAModelRuntimeOneDoes(t *testing.T) {
	local, remote := newLocalDecider(), newFakeSystemOne()
	classifier := systemOneTestClassifier(local, remote, 3000, 0)
	results := newSignalResults()
	classifier.evaluateSystemOneSignals(context.Background(), results, &sync.Mutex{}, systemOneReading{request: "refund my invoice"}, usedRules(classifier.Config.SystemOneRules))

	if remote.calls["hosted-d3"] != 1 || len(remote.requests["hosted-d3"].Questions) != 3 || remote.requests["hosted-d3"].State != "refund my invoice" {
		t.Fatalf("one remote call with every question to hosted-d3, got %v %+v", remote.calls, remote.requests["hosted-d3"])
	}
	if local.calls["kai"] != 1 || local.calls["hosted-d3"] != 0 {
		t.Fatalf("model_runtime questions stay with the model runtime: %v", local.calls)
	}
	for _, want := range []string{"support_team:engineering", "urgency", "local"} {
		if !slices.Contains(results.MatchedSystemOneRules, want) {
			t.Fatalf("missing match %q in %v", want, results.MatchedSystemOneRules)
		}
	}
	if slices.Contains(results.MatchedSystemOneRules, "refund") {
		t.Fatalf("a noul below the default threshold does not match: %v", results.MatchedSystemOneRules)
	}
	if results.SignalValues["systemone:support_team:billing"] != 0.1 || results.SignalConfidences["systemone:support_team"] != 0.9 ||
		results.SignalValues["systemone:refund"] != 0.2 || results.SignalValues["systemone:urgency"] != 1.5 || results.SignalConfidences["systemone:urgency"] != 0.6 {
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
		classifier.evaluateSystemOneSignals(context.Background(), results, &sync.Mutex{}, systemOneReading{request: "x"}, usedRules(classifier.Config.SystemOneRules))
		return results, time.Since(started)
	}
	expectUnknown := func(results *SignalResults, reason string) {
		t.Helper()
		for _, name := range []string{"support_team", "refund", "urgency"} {
			if got := results.SignalErrors["systemone:"+name]; got != reason {
				t.Fatalf("%s: %q, want %q (errors %v)", name, got, reason, results.SignalErrors)
			}
		}
		if !slices.Contains(results.MatchedSystemOneRules, "local") {
			t.Fatalf("a failing remote does not affect model_runtime signals: %v", results.MatchedSystemOneRules)
		}
	}

	results, _ := evaluate(nil, 3000, 0)
	expectUnknown(results, "systemone_unavailable")

	rejected := newFakeSystemOne()
	rejected.failures["hosted-d3"] = modelservice.ErrRejected
	results, _ = evaluate(rejected, 3000, 0)
	expectUnknown(results, "systemone_rejected")

	slow := newFakeSystemOne()
	slow.block = true
	results, took := evaluate(slow, 100, 0)
	expectUnknown(results, "systemone_timeout")
	if took > 2*time.Second {
		t.Fatalf("a slow remote holds the stage only for its rules' timeout, took %s", took)
	}
	// Questions without a timeout take the deployment's timeout_ms, else the
	// default; a question to the decision model joins no built-in call.
	results, took = evaluate(slow, 0, 300)
	expectUnknown(results, "systemone_timeout")
	if took < 250*time.Millisecond || took > 900*time.Millisecond {
		t.Fatalf("the deployment's timeout bounds its call, took %s", took)
	}
	results, took = evaluate(slow, 0, 0)
	expectUnknown(results, "systemone_timeout")
	if took < time.Duration(config.DefaultSystemOneTimeoutMs)*time.Millisecond/2 || took > 3*time.Second {
		t.Fatalf("the default timeout bounds a remote call, took %s", took)
	}
}

// A systemone span rule masking reads asks for the whole text, whatever
// coverage the rule asks for a decision, and its spans come back under the
// rule's name.
func TestSystemOneSpanRuleSpansReadTheWholeText(t *testing.T) {
	remote := newFakeSystemOne()
	remote.card = modelservice.RemoteCard("vllm-sr/d3-ultra", []string{"choice", "noul", "score", "span"}, nil)
	remote.answers["systemone:customer_account"] = modelservice.Answer{
		Type: "span", InputCoverage: "complete",
		Spans: []modelservice.Span{{Label: "account number", Start: 12, End: 19, Text: "88-1910", Probability: 0.9}},
	}
	classifier := systemOneTestClassifier(newLocalDecider(), remote, 0, 0)
	classifier.Config.SystemOneRules = append(classifier.Config.SystemOneRules, config.SystemOneRule{
		Name: "customer_account", Deployment: "hosted-d3",
		Question: config.SystemOneQuestion{Type: "span", Instructions: "Which spans are account numbers?", Labels: []config.SystemOneOption{{Key: "account number"}}},
	})
	spans, err := classifier.RuleSpans(t.Context(), config.SignalTypeSystemOne, "customer_account", "Paid for #  88-1910 today")
	if err != nil {
		t.Fatal(err)
	}
	if len(spans) != 1 || spans[0].Label != "customer_account" || spans[0].Text != "88-1910" {
		t.Fatalf("spans = %+v, want the account number under the rule's name", spans)
	}
	question := remote.asked[len(remote.asked)-1].Questions[0]
	if question.ID != "systemone:customer_account" || question.Truncate {
		t.Fatalf("masking asked %+v, want the rule's question over the whole text", question)
	}
	// A span outside the text fails the read rather than being clamped onto
	// a range that leaves part of the value in clear text.
	for _, bad := range [][2]int{{-1, 3}, {12, 99}, {5, 5}} {
		remote.answers["systemone:customer_account"] = modelservice.Answer{
			Type: "span", InputCoverage: "complete",
			Spans: []modelservice.Span{{Label: "account number", Start: bad[0], End: bad[1], Text: "88-1910", Probability: 0.9}},
		}
		if spans, err := classifier.RuleSpans(t.Context(), config.SignalTypeSystemOne, "customer_account", "Paid for #  88-1910 today"); err == nil {
			t.Fatalf("span %v read as %+v, want an error", bad, spans)
		}
	}
}

// A guard rule the plugins ask reads the whole command, and matches only by
// its own name: a choice answer that names one of its options without the
// rule matching is no match.
func TestSystemOneGuardRuleMatchesReadTheWholeText(t *testing.T) {
	remote := newFakeSystemOne()
	remote.answers["systemone:deletes_data"] = modelservice.Answer{Type: "noul", Noul: 0.9, InputCoverage: "complete"}
	remote.answers["systemone:support_team"] = modelservice.Answer{Type: "choice", Choice: "engineering", Probabilities: map[string]float64{"billing": 0.1, "engineering": 0.9}, InputCoverage: "complete"}
	classifier := systemOneTestClassifier(newLocalDecider(), remote, 0, 0)
	classifier.Config.SystemOneRules = append(classifier.Config.SystemOneRules, config.SystemOneRule{
		Name: "deletes_data", Deployment: "hosted-d3", Question: config.SystemOneQuestion{Type: "noul", Instructions: "Does this delete data?"},
	})
	matched, err := classifier.RuleMatches(t.Context(), config.SignalTypeSystemOne, "deletes_data", "rm -rf /srv/data")
	if err != nil || !matched {
		t.Fatalf("matched = %v, %v; want the noul rule to match", matched, err)
	}
	if question := remote.asked[len(remote.asked)-1].Questions[0]; question.Truncate {
		t.Fatalf("guard asked %+v, want the whole command", question)
	}
	// support_team has no labels: its answer names an option, never the rule.
	if matched, err := classifier.RuleMatches(t.Context(), config.SignalTypeSystemOne, "support_team", "fix the login"); err != nil || matched {
		t.Fatalf("a choice rule without labels matched = %v, %v", matched, err)
	}
	delete(remote.answers, "systemone:deletes_data")
	if _, err := classifier.RuleMatches(t.Context(), config.SignalTypeSystemOne, "deletes_data", "rm -rf /srv/data"); err == nil {
		t.Fatal("a missing answer read as no match")
	}
}
