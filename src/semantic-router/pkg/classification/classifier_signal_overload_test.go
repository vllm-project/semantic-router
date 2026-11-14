package classification

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// busyDecider refuses every call as overloaded until free, as a decision
// runtime whose admission queue is full does, and then answers as its
// fakeDecider does.
type busyDecider struct {
	*fakeDecider
	lock  sync.Mutex
	free  time.Time
	asked int
}

func (d *busyDecider) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	d.lock.Lock()
	d.asked++
	busy := time.Now().Before(d.free)
	d.lock.Unlock()
	if busy {
		return modelservice.Response{}, modelservice.ErrOverloaded
	}
	return d.fakeDecider.Decide(ctx, deployment, request)
}

func (d *busyDecider) calls() int {
	d.lock.Lock()
	defer d.lock.Unlock()
	return d.asked
}

func benignJailbreakDecider(busy time.Duration) *busyDecider {
	return &busyDecider{fakeDecider: newFakeDecider(map[string]modelservice.Answer{
		presetQuestionID(presets.Jailbreak): {Type: "choice", Choice: "benign", Probabilities: map[string]float64{"benign": 0.97, "jailbreak": 0.03}},
	}), free: time.Now().Add(busy)}
}

// evaluateJailbreakWaiting evaluates cfg's jailbreak rules in a stage that
// waits up to wait for capacity (none for 0), and how long that took.
func evaluateJailbreakWaiting(t *testing.T, cfg *config.RouterConfig, decider *busyDecider, wait time.Duration) (*SignalResults, time.Duration) {
	t.Helper()
	return evaluateJailbreakWith(newApplicationTestClassifier(t, cfg, decider), cfg, "hello", wait)
}

// evaluateJailbreakWith is evaluateJailbreakWaiting with a classifier and a
// request of its own.
func evaluateJailbreakWith(classifier *Classifier, cfg *config.RouterConfig, text string, wait time.Duration) (*SignalResults, time.Duration) {
	results := newSignalResults()
	started := time.Now()
	classifier.evaluateApplicationSignals(withOverloadWait(context.Background(), wait), results, &sync.Mutex{}, config.SignalTypeJailbreak,
		systemOneReading{request: text}, usedApplicationRules(cfg))
	return results, time.Since(started)
}

// A call the decision runtime refuses as overloaded is asked again while its
// request waits for capacity: when capacity frees in time the rule reads its
// answer, and when it does not the rule fails as before, for the reason
// overloaded.
func TestAnOverloadedGuardCallIsAskedAgainWhileTheRequestWaitsForCapacity(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Deployment: "kai"}}

	decider := benignJailbreakDecider(150 * time.Millisecond)
	results, elapsed := evaluateJailbreakWaiting(t, cfg, decider, 3*time.Second)
	if results.SignalErrorMatches["jailbreak:attack"] || slices.Contains(results.MatchedJailbreakRules, "attack") || decider.calls() < 2 || elapsed > 2*time.Second {
		t.Fatalf("capacity freed after 150 ms of a 3 s wait: matched %v, errors %v, %d calls in %v",
			results.MatchedJailbreakRules, results.SignalErrors, decider.calls(), elapsed)
	}

	decider = benignJailbreakDecider(time.Hour)
	results, elapsed = evaluateJailbreakWaiting(t, cfg, decider, 300*time.Millisecond)
	if !results.SignalErrorMatches["jailbreak:attack"] || results.SignalErrors["jailbreak:attack"] != signalErrorOverloaded || decider.calls() < 2 {
		t.Fatalf("still overloaded after the wait: matched %v, errors %v, %d calls", results.MatchedJailbreakRules, results.SignalErrors, decider.calls())
	}
	if elapsed < 250*time.Millisecond || elapsed > 2*time.Second {
		t.Fatalf("the wait of 300 ms took %v", elapsed)
	}
}

// Without a wait (on_signal_overload absent, or refuse) an overloaded call
// is asked once, as before. With one, a rule that does not fail closed is
// asked again too, and stays unknown, unmatched, when the wait runs out.
func TestAnOverloadedCallIsAskedOnceWithoutAWait(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Deployment: "kai"}}
	decider := benignJailbreakDecider(time.Hour)
	results, _ := evaluateJailbreakWaiting(t, cfg, decider, 0)
	if decider.calls() != 1 || !results.SignalErrorMatches["jailbreak:attack"] || results.SignalErrors["jailbreak:attack"] != signalErrorOverloaded {
		t.Fatalf("without a wait: %d calls, errors %v", decider.calls(), results.SignalErrors)
	}

	cfg.JailbreakRules = []config.JailbreakRule{{Name: "lenient", Deployment: "kai", OnUnknown: string(config.RuleOnUnknownNoMatch)}}
	decider = benignJailbreakDecider(time.Hour)
	results, _ = evaluateJailbreakWaiting(t, cfg, decider, 300*time.Millisecond)
	if decider.calls() < 2 || results.SignalErrorMatches["jailbreak:lenient"] || results.SignalErrors["jailbreak:lenient"] != signalErrorOverloaded {
		t.Fatalf("a rule that does not fail closed: %d calls, matches %v, errors %v", decider.calls(), results.SignalErrorMatches, results.SignalErrors)
	}
}

// A request asks an overloaded deployment again in at most three rounds,
// spread over its wait: one call is asked four times in all, the last near
// the wait's end.
func TestARequestAsksAnOverloadedDeploymentAgainInAtMostThreeRounds(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Deployment: "kai"}}
	decider := benignJailbreakDecider(time.Hour)
	results, elapsed := evaluateJailbreakWaiting(t, cfg, decider, 2*time.Second)
	if decider.calls() != 1+overloadRounds || results.SignalErrors["jailbreak:attack"] != signalErrorOverloaded {
		t.Fatalf("a call to a deployment that stays overloaded: %d calls, errors %v", decider.calls(), results.SignalErrors)
	}
	if elapsed < 1500*time.Millisecond || elapsed > 3*time.Second {
		t.Fatalf("the rounds of a 2 s wait ended after %v", elapsed)
	}
}

// Requests that wait for the same overloaded deployment take turns: one
// request's round asks it per step, however many wait, and the first answer
// opens it to all of them at once.
func TestWaitingRequestsAskAnOverloadedDeploymentOneRoundAtATime(t *testing.T) {
	const requests = 8
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Deployment: "kai"}}
	wait := func(decider *busyDecider, wait time.Duration) ([]*SignalResults, time.Duration) {
		classifier := newApplicationTestClassifier(t, cfg, decider)
		results := make([]*SignalResults, requests)
		started := time.Now()
		var wg sync.WaitGroup
		for i := range requests {
			wg.Go(func() {
				results[i], _ = evaluateJailbreakWith(classifier, cfg, fmt.Sprintf("hello %d", i), wait)
			})
		}
		wg.Wait()
		return results, time.Since(started)
	}

	decider := benignJailbreakDecider(time.Hour)
	results, _ := wait(decider, time.Second)
	if asked := decider.calls() - requests; asked < 1 || asked > 6 {
		t.Fatalf("%d requests waiting 1 s for a deployment that stays overloaded asked it again %d times", requests, asked)
	}
	for _, result := range results {
		if result.SignalErrors["jailbreak:attack"] != signalErrorOverloaded {
			t.Fatalf("a request read its rule from a deployment that stayed overloaded: %v", result.SignalErrors)
		}
	}

	decider = benignJailbreakDecider(300 * time.Millisecond)
	results, elapsed := wait(decider, 3*time.Second)
	for _, result := range results {
		if len(result.SignalErrors) > 0 || slices.Contains(result.MatchedJailbreakRules, "attack") {
			t.Fatalf("capacity freed after 300 ms of a 3 s wait: matched %v, errors %v", result.MatchedJailbreakRules, result.SignalErrors)
		}
	}
	if decider.calls() > 2*requests || elapsed > 2*time.Second {
		t.Fatalf("once the deployment answered, %d requests took %d calls in all, in %v", requests, decider.calls(), elapsed)
	}
}

// A stage takes one round for a deployment at a time, whichever round its
// waiting calls reach: a call that reaches another round just after one was
// taken rides it, one that comes sooner than a step after it waits, and a
// round a step later takes another. So a deployment that answers some calls
// and refuses others does not spend a request's rounds at once.
func TestAStageTakesOneRoundForADeploymentAtATime(t *testing.T) {
	wait := &overloadWait{wait: 2 * time.Second, rounds: map[string]*deploymentRounds{}}
	gate := &overloadGate{}
	now := time.Now()
	ask := func(after time.Duration) bool {
		together, alone := wait.asks("kai", &overloadRound{at: now.Add(after)}, gate, now.Add(after))
		return together && !alone
	}
	taken := func() int { return wait.roundsOfLocked("kai").taken }
	if !ask(0) || taken() != 1 {
		t.Fatalf("the first round: taken %d", taken())
	}
	if !ask(10*time.Millisecond) || taken() != 1 {
		t.Fatalf("a call 10 ms after the round asks in it without taking another: taken %d", taken())
	}
	if ask(60*time.Millisecond) || taken() != 1 {
		t.Fatalf("a call 60 ms after the round waits for a later one: taken %d", taken())
	}
	if !ask(150*time.Millisecond) || taken() != 2 {
		t.Fatalf("a round 150 ms later takes the second: taken %d", taken())
	}
}

// A refusal is the overloaded rules' only when the decisions would not
// refuse had those rules stayed unknown: a genuine match of another rule, a
// failure for another reason, or a decision that does not refuse keeps the
// request's outcome.
func TestARefusalIsTheOverloadedRulesOnlyWhenNothingElseRefuses(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.FastResponsePluginConfig{Message: "blocked"})
	if err != nil {
		t.Fatal(err)
	}
	refuse := []config.DecisionPlugin{{Type: config.DecisionPluginFastResponse, Configuration: payload}}
	guard := config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
		{Type: config.SignalTypeJailbreak, Name: "attack"}, {Type: config.SignalTypePII, Name: "personal"},
	}}
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Deployment: "kai"}}
	cfg.PIIRules = []config.PIIRule{{Name: "personal", Deployment: "kai"}}
	cfg.Decisions = []config.Decision{
		{Name: "guard", Priority: 100, Rules: guard, Plugins: refuse},
		{Name: "rest", Priority: 1, ModelRefs: []config.ModelRef{{Model: "m"}}},
	}
	classifier := newApplicationTestClassifier(t, cfg, newFakeDecider(nil))
	failed := func(reason string) *SignalResults {
		results := newSignalResults()
		results.MatchedJailbreakRules = []string{"attack"}
		results.SignalErrors["jailbreak:attack"] = systemOneSignalErrorPrefix + reason
		results.SignalErrorMatches = map[string]bool{"jailbreak:attack": true}
		return results
	}
	refusal := func(results *SignalResults) []string {
		t.Helper()
		result, evaluated := classifier.EvaluateDecisionWithEngine(results)
		if evaluated != nil || result == nil || result.Decision.Name != "guard" {
			t.Fatalf("the guard does not refuse: %+v, %v", result, evaluated)
		}
		return classifier.OverloadedRefusal(results, nil, result)
	}

	if rules := refusal(failed("overloaded")); !slices.Equal(rules, []string{"jailbreak:attack"}) {
		t.Fatalf("an overloaded rule that alone refuses: %v", rules)
	}
	if rules := refusal(failed("timeout")); rules != nil {
		t.Fatalf("a rule that timed out: %v", rules)
	}
	detected := failed("overloaded")
	detected.MatchedPIIRules = []string{"personal"}
	if rules := refusal(detected); rules != nil {
		t.Fatalf("another rule's match refuses anyway: %v", rules)
	}
	if !slices.Equal(detected.MatchedJailbreakRules, []string{"attack"}) || !detected.SignalErrorMatches["jailbreak:attack"] {
		t.Fatalf("asking what the decisions would say changed the request's results: %v %v", detected.MatchedJailbreakRules, detected.SignalErrorMatches)
	}

	cfg.Decisions[0].Plugins, cfg.Decisions[0].ModelRefs = nil, []config.ModelRef{{Model: "quarantine"}}
	results := failed("overloaded")
	result, err := classifier.EvaluateDecisionWithEngine(results)
	if err != nil || result == nil || classifier.OverloadedRefusal(results, nil, result) != nil {
		t.Fatalf("a decision that routes instead of refusing: %+v, %v", result, err)
	}
}

// A request no decision refuses, while a rule whose match would refuse it
// stayed unknown because the runtime was overloaded, is not forwarded with
// that rule unread: a rule that fails open (safety) is unread, not benign.
// So is one whose match alone would refuse it, though all of them matched
// would not. Another failure of the rule, or an unread rule whose match
// would not refuse, leaves the request to its policy.
func TestARequestIsNotForwardedWithAnOverloadedRefusingRuleUnread(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.FastResponsePluginConfig{Message: "blocked"})
	if err != nil {
		t.Fatal(err)
	}
	guard := config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
		{Type: config.SignalTypeJailbreak, Name: "attack"}, {Type: config.SignalTypeSafety, Name: "unsafe"},
	}}
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Deployment: "kai"}, {Name: "hint", Deployment: "kai"}}
	cfg.SafetyRules = []config.SafetyRule{{Name: "unsafe", Deployment: "kai"}}
	cfg.Decisions = []config.Decision{
		{Name: "review", Priority: 200, Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{{Type: config.SignalTypeJailbreak, Name: "hint"}}}, ModelRefs: []config.ModelRef{{Model: "reviewer"}}},
		{Name: "guard", Priority: 100, Rules: guard, Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginFastResponse, Configuration: payload}}},
		{Name: "rest", Priority: 1, ModelRefs: []config.ModelRef{{Model: "m"}}},
	}
	classifier := newApplicationTestClassifier(t, cfg, newFakeDecider(nil))
	unread := func(failed map[string]string) []string {
		t.Helper()
		results := newSignalResults()
		for key, reason := range failed {
			results.SignalErrors[key] = systemOneSignalErrorPrefix + reason
		}
		result, err := classifier.EvaluateDecisionWithEngine(results)
		if err != nil || result == nil || result.Decision.Name != "rest" {
			t.Fatalf("the request is not routed: %+v, %v", result, err)
		}
		return classifier.OverloadedUnread(results, nil, result)
	}
	if rules := unread(map[string]string{"safety:unsafe": "overloaded"}); !slices.Equal(rules, []string{"safety:unsafe"}) {
		t.Fatalf("a refusing rule unread for overload: %v", rules)
	}
	if rules := unread(map[string]string{"safety:unsafe": "timeout"}); rules != nil {
		t.Fatalf("a refusing rule that timed out: %v", rules)
	}
	if rules := unread(map[string]string{"jailbreak:hint": "overloaded"}); rules != nil {
		t.Fatalf("an unread rule whose match only routes: %v", rules)
	}
	both := unread(map[string]string{"jailbreak:hint": "overloaded", "safety:unsafe": "overloaded"})
	if !slices.Equal(both, []string{"jailbreak:hint", "safety:unsafe"}) {
		t.Fatalf("a refusing rule unread with one whose match would route instead: %v", both)
	}
}
