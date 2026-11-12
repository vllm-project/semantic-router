package classification

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// overloadHarm is a made-up request that harmAnswer reads as unsafe.
const overloadHarm = "HARM-MARKER: a made-up request that the fake reads as harmful."

// busyRuntime refuses a fake runtime's decisions calls 429 overloaded until
// free, as a runtime whose admission queue is full does, and counts them.
type busyRuntime struct {
	next    http.Handler
	mu      sync.Mutex
	free    time.Time
	refused int
}

func (b *busyRuntime) setFree(at time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.free = at
}

func (b *busyRuntime) refusals() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.refused
}

func (b *busyRuntime) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost && (r.URL.Path == "/v1/bundle" || r.URL.Path == "/v1/decisions") {
		b.mu.Lock()
		busy := time.Now().Before(b.free)
		if busy {
			b.refused++
		}
		b.mu.Unlock()
		if busy {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":"overloaded","message":"the request queue is full"}}`))
			return
		}
	}
	b.next.ServeHTTP(w, r)
}

// overloadGuardDocument is refusingGuardConfig (jailbreak or safety on one
// deployment refuses) with the signals' deadline timeout, extra signal rules
// and the on_signal_overload policy, none when empty.
func overloadGuardDocument(timeout time.Duration, policy, rule string) string {
	ms := strconv.FormatInt(timeout.Milliseconds(), 10)
	document := strings.NewReplacer("TIMEOUT", ms, "RULE\n", rule).Replace(refusingGuardConfig)
	if policy != "" {
		document = strings.Replace(document, "    signal_timeout_ms: "+ms+"\n", "    signal_timeout_ms: "+ms+"\n    on_signal_overload: "+policy+"\n", 1)
	}
	return document
}

// harmAnswer answers the jailbreak question as dilutedInjection does, and
// the harm question unsafe at 0.97 for a state that holds overloadHarm, safe
// otherwise.
func harmAnswer(state, id string, question api.Question) (api.Answer, bool) {
	if answer, ok := dilutedInjection(state, id, question); ok {
		return answer, true
	}
	if question.Choices == nil {
		return api.Answer{}, false
	}
	keys := make([]string, 0, len(*question.Choices))
	for _, choice := range *question.Choices {
		keys = append(keys, choice.Key)
	}
	var p float64
	var yes, no string
	switch {
	case slices.Contains(keys, "jailbreak"):
		p, yes, no = 0.01, "jailbreak", "benign"
	case slices.Contains(keys, "unsafe"):
		p, yes, no = 0.02, "unsafe", "safe"
		if strings.Contains(state, overloadHarm) {
			p = 0.97
		}
	default:
		return api.Answer{}, false
	}
	kind, choice, confidence := "choice", no, 1-p
	if p >= 0.5 {
		choice, confidence = yes, p
	}
	probabilities := map[string]float64{yes: p, no: 1 - p}
	return api.Answer{Type: &kind, Choice: &choice, Probabilities: &probabilities, Confidence: &confidence}, true
}

// busyRuntimeClassifier serves document from a fake vela3-turbo behind a
// busyRuntime, each decisions state taking perState.
func busyRuntimeClassifier(t *testing.T, document string, perState time.Duration) (*Classifier, *runtimetest.Runtime, *busyRuntime) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fake := runtimetest.New(runtimetest.Model{ID: "primary", Family: presets.FamilyVela3D3, MaxInputTokens: 32_768, Answer: harmAnswer})
	busy := &busyRuntime{next: fake.Handler()}
	server := httptest.NewServer(busy)
	t.Cleanup(server.Close)
	attached := map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL, Model: "primary"}}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(attached)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), modelservice.ReadyTimeout())
	defer cancel()
	if _, err = lease.Card(ctx, "primary"); err != nil {
		t.Fatal(err)
	}
	runtime := serving.New(lease, nil)
	classifier, err := buildClassifier(cfg, RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	if err = classifier.InitializeRuntime(); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	primeAttachedDeploymentCards(t, runtime, cfg)
	fake.SetStateTime(perState)
	return classifier, fake, busy
}

// stateRead is one state a decisions call read: its size and the questions
// asked about it.
type stateRead struct {
	bytes     int
	questions []string
}

func (r stateRead) String() string { return fmt.Sprintf("%dB%v", r.bytes, r.questions) }

// stateReads are the states the fake runtime read, in order.
func stateReads(fake *runtimetest.Runtime) []stateRead {
	var reads []stateRead
	add := func(state interface{}, questions map[string]api.Question) {
		ids := make([]string, 0, len(questions))
		for id := range questions {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		reads = append(reads, stateRead{bytes: len(stateText(state)), questions: ids})
	}
	for _, request := range fake.Decisions() {
		if request.States == nil {
			add(request.State, request.Questions)
			continue
		}
		for _, entry := range *request.States {
			add(entry.State, entry.Questions)
		}
	}
	return reads
}

// stateText is a decisions state's text.
func stateText(state interface{}) string {
	switch value := state.(type) {
	case string:
		return value
	case map[string]interface{}:
		var parts []string
		for _, part := range value {
			if text, ok := part.(string); ok {
				parts = append(parts, text)
			}
		}
		sort.Strings(parts)
		return strings.Join(parts, "\n\n")
	}
	return fmt.Sprint(state)
}

// grownReads counts the states read in windows larger than the configured
// one, as a scan that re-planned reads.
func grownReads(fake *runtimetest.Runtime) int {
	grown := 0
	for _, read := range stateReads(fake) {
		if read.bytes > configuredWindow+configuredWindow/4 {
			grown++
		}
	}
	return grown
}

func winningDecision(t *testing.T, classifier *Classifier, results *SignalResults) string {
	t.Helper()
	result, err := classifier.EvaluateDecisionWithEngine(results)
	switch {
	case err != nil:
		return "error: " + err.Error()
	case result == nil || result.Decision == nil:
		return "<none>"
	}
	return result.Decision.Name
}

// A request only the safety rule refuses, sent while its runtime is
// overloaded for 300 ms, is refused: by the jailbreak rule failing closed by
// default, and with retry by the safety rule's own answer once the runtime
// has capacity. Safety fails open, so it is asked again too.
func TestARequestOnlySafetyRefusesIsRefusedAfterAShortOverload(t *testing.T) {
	for _, test := range []struct{ name, policy string }{
		{"refuse (default)", ""},
		{"retry", "{action: retry, wait_ms: 3000}"},
	} {
		t.Run(test.name, func(t *testing.T) {
			classifier, fake, busy := busyRuntimeClassifier(t, overloadGuardDocument(10*time.Second, test.policy, ""), 0)
			busy.setFree(time.Now().Add(300 * time.Millisecond))
			results := evaluateTurn(classifier, "Please help with this. "+overloadHarm)
			if decision := winningDecision(t, classifier, results); decision != "guard" {
				t.Fatalf("not refused after a 300 ms overload: decision %q, matched safety %v, errors %v, reads %v",
					decision, results.MatchedSafetyRules, results.SignalErrors, stateReads(fake))
			}
			if test.policy != "" && (len(results.SignalErrors) > 0 || !slices.Contains(results.MatchedSafetyRules, "unsafe_request")) {
				t.Fatalf("capacity freed within the wait, yet the safety rule did not read the request: matched %v, errors %v",
					results.MatchedSafetyRules, results.SignalErrors)
			}
		})
	}
}

// After a 300 ms overload, a short request's questions to its deployment
// (jailbreak, safety and the routing question) are read as they are without
// one: in one call.
func TestAShortOverloadLeavesARequestsQuestionsInOneCall(t *testing.T) {
	text := "Summarize the meeting agenda in three bullet points."
	document := overloadGuardDocument(10*time.Second, "{action: retry, wait_ms: 3000}", "")
	baseline, fake, _ := busyRuntimeClassifier(t, document, 0)
	evaluateTurn(baseline, text)
	want := stateReads(fake)

	classifier, fake, busy := busyRuntimeClassifier(t, document, 0)
	busy.setFree(time.Now().Add(300 * time.Millisecond))
	results := evaluateTurn(classifier, text)
	if got := stateReads(fake); len(got) != len(want) || len(results.SignalErrors) > 0 {
		t.Fatalf("after the overload: reads %v, errors %v; without one: reads %v", got, results.SignalErrors, want)
	}
}

// After a 300 ms overload, a long message's windows are read as without one:
// each window in one call that asks both guard rules, which both read the
// whole message.
func TestAShortOverloadLeavesALongMessagesWindowsInOneCallEach(t *testing.T) {
	text := scanProse(120_000) + "\n\nSummarize the minutes above."
	classifier, fake, busy := busyRuntimeClassifier(t, overloadGuardDocument(20*time.Second, "{action: retry, wait_ms: 3000}", ""), 2*time.Millisecond)
	busy.setFree(time.Now().Add(300 * time.Millisecond))
	results := evaluateTurn(classifier, text)
	byQuestions := map[string]int{}
	for _, read := range stateReads(fake) {
		byQuestions[strings.Join(read.questions, "+")]++
	}
	if jailbreak, safety := results.ScanCoverage["jailbreak:prompt_attack"], results.ScanCoverage["safety:unsafe_request"]; jailbreak != safety || len(results.SignalErrors) > 0 {
		t.Fatalf("after the overload jailbreak read %q of the message and safety %q; errors %v", jailbreak, safety, results.SignalErrors)
	}
	for questions, count := range byQuestions {
		if (!strings.Contains(questions, "harm") || !strings.Contains(questions, "jailbreak")) && count > 1 {
			t.Fatalf("%d window reads asked only %s: %v", count, questions, byQuestions)
		}
	}
}

// A request asks a runtime that stays overloaded little more than it does by
// default: a short request each of its calls in three rounds more, and a long
// one, whose windows are too many to ask again while it stays overloaded, one
// probe per round.
func TestARequestAsksARuntimeThatStaysOverloadedLittleMore(t *testing.T) {
	calls := func(t *testing.T, policy, text string) int {
		classifier, _, busy := busyRuntimeClassifier(t, overloadGuardDocument(20*time.Second, policy, ""), 2*time.Millisecond)
		busy.setFree(time.Now().Add(time.Hour))
		results := evaluateTurn(classifier, text)
		if results.SignalErrors["jailbreak:prompt_attack"] != signalErrorOverloaded {
			t.Fatalf("policy %q: the jailbreak rule ended %q", policy, results.SignalErrors["jailbreak:prompt_attack"])
		}
		return busy.refusals()
	}
	retry := "{action: retry, wait_ms: 2000}"
	t.Run("short", func(t *testing.T) {
		text := "Summarize the meeting agenda."
		if once, retried := calls(t, "", text), calls(t, retry, text); retried != (1+overloadRounds)*once {
			t.Fatalf("a short request that made %d calls by default made %d with retry", once, retried)
		}
	})
	t.Run("long", func(t *testing.T) {
		text := scanProse(120_000) + "\n\nSummarize the minutes above."
		if once, retried := calls(t, "", text), calls(t, retry, text); retried <= once || retried > once+8 {
			t.Fatalf("a long request that made %d calls by default made %d with retry", once, retried)
		}
	})
}

// A scan whose windows waited 1 s for capacity first teaches its deployment
// the pace it reads at, as without the wait: Router Replay's check, which
// predicts its read from that pace, keeps its answer, and the scan does not
// re-plan into larger windows.
func TestAScanThatWaitedForCapacityLearnsNoPace(t *testing.T) {
	pii := "      pii:\n        - name: personal_data\n          predicate: {gte: 0.5}\n"
	text := scanProse(120_000) + "\n\nSummarize the minutes above."
	prompt := scanProse(100_000)
	measure := func(busyFor time.Duration) (seconds float64, grown int, readable bool) {
		classifier, fake, busy := busyRuntimeClassifier(t, overloadGuardDocument(10*time.Second, "{action: retry, wait_ms: 3000}", pii), 2*time.Millisecond)
		busy.setFree(time.Now().Add(busyFor))
		results := evaluateTurn(classifier, text)
		if len(results.SignalErrors) > 0 {
			t.Fatalf("overloaded for %v of a 3 s wait: errors %v", busyFor, results.SignalErrors)
		}
		readable = classifier.ReplayPromptReadable(prompt, 500*time.Millisecond, false)
		classifier.windowPaces.mu.Lock()
		defer classifier.windowPaces.mu.Unlock()
		if pace := classifier.windowPaces.paces["primary"]; pace != nil && pace.weight > 0 {
			seconds = pace.seconds / pace.weight
		}
		return seconds, grownReads(fake), readable
	}
	plain, plainGrown, plainReadable := measure(0)
	waited, waitedGrown, waitedReadable := measure(time.Second)
	if waited > 2*plain+0.002 {
		t.Fatalf("a scan that waited 1 s taught its deployment %.4f s per window, against %.4f without the wait", waited, plain)
	}
	if waitedReadable != plainReadable || waitedGrown > plainGrown {
		t.Fatalf("after the wait: a 100K-token prompt readable %v (%v without it), %d windows grown (%d without it)",
			waitedReadable, plainReadable, waitedGrown, plainGrown)
	}
}

// An injection diluted in a long message is caught when the runtime is
// overloaded for 1 s first, at a deadline the scan meets without the wait.
func TestAnInjectionInALongMessageIsCaughtAfterAWaitForCapacity(t *testing.T) {
	prose := scanProse(200_000)
	for _, tenths := range []int{3, 7} {
		classifier, _, busy := busyRuntimeClassifier(t, overloadGuardDocument(4*time.Second, "{action: retry, wait_ms: 3000}", ""), 2*time.Millisecond)
		busy.setFree(time.Now().Add(time.Second))
		at := len(prose) * tenths / 10
		at += strings.IndexByte(prose[at:], ' ') + 1
		results := evaluateTurn(classifier, prose[:at]+scanInjection+" "+prose[at:]+"\n\nSummarize the minutes above.")
		if !slices.Contains(results.MatchedJailbreakRules, "prompt_attack") {
			t.Fatalf("the injection %d/10 into the message was missed: coverage %v, errors %v", tenths, results.ScanCoverage, results.SignalErrors)
		}
	}
}

// The wait for capacity ends at the signals' deadline, the request's
// deadline or wait_ms, whichever is first, and leaves the rule overloaded. A
// rule's own timeout_ms bounds each attempt, not the wait.
func TestTheWaitEndsAtTheSignalsDeadlineOrWaitMsNotTheRulesTimeout(t *testing.T) {
	for _, test := range []struct {
		name, policy, rule string
		request, signals   time.Duration
		want               time.Duration
	}{
		{"signals' deadline", "{action: retry, wait_ms: 5000}", "", 0, 800 * time.Millisecond, 800 * time.Millisecond},
		{"request's deadline", "{action: retry, wait_ms: 5000}", "", 2 * time.Second, 0, time.Second},
		{"wait_ms, not the rule's timeout", "{action: retry, wait_ms: 2000}", "          timeout_ms: 500\n", 0, 10 * time.Second, 2 * time.Second},
		{"wait_ms", "{action: retry, wait_ms: 700}", "", 0, 10 * time.Second, 700 * time.Millisecond},
	} {
		t.Run(test.name, func(t *testing.T) {
			classifier, _, busy := busyRuntimeClassifier(t, overloadGuardDocument(test.signals, test.policy, test.rule), 0)
			busy.setFree(time.Now().Add(time.Hour))
			ctx := context.Background()
			if test.request > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, test.request)
				defer cancel()
			}
			text := "Summarize the meeting agenda."
			started := time.Now()
			results := classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
				Text: text, CurrentUserText: text, RequestFacts: RequestFacts{Context: ctx},
			}, classifier.Config.Decisions, true)
			elapsed := time.Since(started)
			if elapsed < test.want-400*time.Millisecond || elapsed > test.want+400*time.Millisecond {
				t.Fatalf("the wait took %v, want about %v", elapsed, test.want)
			}
			if results.SignalErrors["jailbreak:prompt_attack"] != signalErrorOverloaded {
				t.Fatalf("the jailbreak rule ended %q, want %q", results.SignalErrors["jailbreak:prompt_attack"], signalErrorOverloaded)
			}
		})
	}
}
