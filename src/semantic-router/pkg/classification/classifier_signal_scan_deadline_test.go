package classification

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// configuredWindow is refusingGuardConfig's window, 2,048 tokens of prose,
// in bytes.
const configuredWindow = 2048 * routingViewBytesPerToken

// dilutedInjection answers the jailbreak question about a state as a
// decision model does: a state that holds scanInjection is an attack with a
// probability that falls as the state grows around it, 0.9995 less 0.44 per
// million bytes. A window of the configured size holds it at about 0.996,
// over the guard's 0.995; one of 13 KB, as a 700K-token message's windows
// grow to, at about 0.994, under it.
func dilutedInjection(state, _ string, question api.Question) (api.Answer, bool) {
	if question.Choices == nil || !slices.ContainsFunc(*question.Choices, func(choice api.ChoiceOption) bool { return choice.Key == "jailbreak" }) ||
		!strings.Contains(state, scanInjection) {
		return api.Answer{}, false
	}
	attack := 0.9995 - 0.44e-6*float64(len(state))
	kind, choice := "choice", "jailbreak"
	probabilities := map[string]float64{"benign": 1 - attack, "jailbreak": attack}
	return api.Answer{Type: &kind, Choice: &choice, Probabilities: &probabilities, Confidence: &attack}, true
}

// dilutingGuardClassifier serves refusingGuardConfig, with the signals'
// deadline timeout, from a fake vela3-turbo validated at 32,768 tokens that
// answers as dilutedInjection says and whose engine reads one state at a
// time, perState and perWord for each of its words.
func dilutingGuardClassifier(t *testing.T, timeout, perState, perWord time.Duration) (*Classifier, *runtimetest.Runtime) {
	t.Helper()
	document := strings.NewReplacer("TIMEOUT", fmt.Sprint(timeout.Milliseconds()), "RULE\n", "").Replace(refusingGuardConfig)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"primary": {Family: presets.FamilyVela3D3, MaxInputTokens: 32_768, Answer: dilutedInjection},
	})
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
	fake.SetWordTime(perWord)
	return classifier, fake
}

// distinctProse is benign prose of about tokens tokens whose every item names
// request, so no window of it is one another request already had read, which
// the result cache would answer at once.
func distinctProse(tokens, request int) string {
	var prose strings.Builder
	for item := 1; prose.Len() < tokens*9/2; item++ {
		fmt.Fprintf(&prose, "Item %d.%d: the committee reviewed the quarterly report on regional water supplies and agreed to meet again. ", request, item)
	}
	return prose.String()
}

// injectedAt is prose with scanInjection at share of it, between words.
func injectedAt(prose string, share float64) string {
	at := int(float64(len(prose)) * share)
	at += strings.IndexByte(prose[at:], ' ') + 1
	return prose[:at] + scanInjection + " " + prose[at:] + "\n\nSummarize the minutes above."
}

// unknownAnswersCounted is vsr_model_runtime_unknown_answers_total over every
// deployment and reason.
func unknownAnswersCounted(t *testing.T) float64 {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for _, family := range families {
		if family.GetName() == "vsr_model_runtime_unknown_answers_total" {
			for _, metric := range family.GetMetric() {
				total += metric.GetCounter().GetValue()
			}
		}
	}
	return total
}

// guardStatesSince are the states the fake read for the jailbreak question
// after the first from of them.
func guardStatesSince(fake *runtimetest.Runtime, from int) []string {
	return guardStates(fake)[from:]
}

// An injection at the very end of a 700,000-token message is refused however
// short the deadline: the tail is read first, at the configured size. Grown
// to fit the deadline, as windows were, the tail grew with the others: the
// first message showed the deployment's pace, 20 ms a window, at which the
// next one's windows would not all be read in 3 s at any size, so they grew
// to the largest the model takes, about 31 KB. The message is cut so that it
// ends in a nearly full one, which holds the injection at about 0.987, under
// the guard's 0.995.
func TestAnInjectionAtTheEndOfA700KMessageIsReadAtTheConfiguredSize(t *testing.T) {
	classifier, fake := dilutingGuardClassifier(t, 3*time.Second, 20*time.Millisecond, 0)
	evaluateTurn(classifier, distinctProse(700_000, 0)+"\n\nSummarize the minutes above.")
	budget := classifier.scanWindowBudget("primary")
	prose := distinctProse(700_000, 1)
	text := ""
	for end := len(prose); end > len(prose)-2*budget.most && text == ""; end -= 256 {
		candidate := prose[:strings.LastIndexByte(prose[:end], ' ')] + " " + scanInjection
		if windows := scanWindowsOf(candidate, budget.most, budget.overlap); windows[len(windows)-1].end-windows[len(windows)-1].start > budget.most*9/10 {
			text = candidate
		}
	}
	if text == "" {
		t.Fatal("no cut of the message ends in a nearly full window of the largest size")
	}
	before := len(guardStates(fake))

	results := evaluateTurn(classifier, text)

	if !slices.Contains(results.MatchedJailbreakRules, "prompt_attack") || results.SignalErrorMatches["jailbreak:prompt_attack"] {
		t.Fatalf("the injection at the end was not refused: jailbreak %v, coverage %v, errors %v",
			results.MatchedJailbreakRules, results.ScanCoverage, results.SignalErrors)
	}
	for _, state := range guardStatesSince(fake, before) {
		if strings.HasSuffix(state, scanInjection) && len(state) > configuredWindow {
			t.Fatalf("the tail was read in a window of %d bytes, larger than the configured %d", len(state), configuredWindow)
		}
	}
}

// A window larger than the configured size whose answer comes close to a
// match is read again in windows of that size: an injection at 60% of a
// 700,000-token message reads at about 0.994 in its 13 KB window, under the
// guard's 0.995, and at about 0.996 in the configured window that holds it.
// The windows of a benign message are read once each.
func TestAGrownWindowCloseToAMatchIsReadAgainAtTheConfiguredSize(t *testing.T) {
	classifier, fake := dilutingGuardClassifier(t, 20*time.Second, time.Millisecond, 0)

	results := evaluateTurn(classifier, injectedAt(distinctProse(700_000, 2), 0.6))

	if !slices.Contains(results.MatchedJailbreakRules, "prompt_attack") || results.SignalErrorMatches["jailbreak:prompt_attack"] {
		t.Fatalf("the injection at 60%% was not refused: jailbreak %v, coverage %v, errors %v",
			results.MatchedJailbreakRules, results.ScanCoverage, results.SignalErrors)
	}
	var large, configured int
	for _, state := range guardStates(fake) {
		switch {
		case !strings.Contains(state, scanInjection):
		case len(state) > configuredWindow:
			large++
		default:
			configured++
		}
	}
	if large == 0 || configured == 0 {
		t.Fatalf("the injection was read in %d large windows and %d of the configured size; want both", large, configured)
	}

	before := len(guardStates(fake))
	benign := evaluateTurn(classifier, distinctProse(700_000, 3)+"\n\nSummarize the minutes above.")
	states := guardStatesSince(fake, before)
	if benign.ScanCoverage["jailbreak:prompt_attack"] != scanCoverageFull || guarded(benign) || len(states) > maxScanWindows {
		t.Fatalf("benign: coverage %v, guarded %v, %d windows read", benign.ScanCoverage, guarded(benign), len(states))
	}
}

// An input whose windows of the configured size are all read by the
// signals' deadline keeps them, however little time that leaves: it is read
// as before windows could grow, as fast, and at the size its rules expect.
// The fake engine reads a window in 40 ms, one at a time, so the message's
// windows take five sixths of the deadline, more than the share a deadline
// once had to leave for larger windows to be chosen.
func TestAnInputReadWholeInTimeKeepsTheConfiguredWindows(t *testing.T) {
	const perWindow = 40 * time.Millisecond
	text := distinctProse(80_000, 4) + "\n\nSummarize the minutes above."
	windows := len(scanWindowsOf(text, configuredWindow, 256*routingViewBytesPerToken))
	deadline := time.Duration(windows) * perWindow * 6 / 5
	classifier, fake := dilutingGuardClassifier(t, deadline, perWindow, 0)
	evaluateTurn(classifier, distinctProse(80_000, 5)+"\n\nSummarize the minutes above.")

	before := len(guardStates(fake))
	started := time.Now()
	results := evaluateTurn(classifier, text)
	elapsed := time.Since(started)

	states := guardStatesSince(fake, before)
	if results.ScanCoverage["jailbreak:prompt_attack"] != scanCoverageFull || elapsed > deadline {
		t.Fatalf("coverage %v after %v: the message should be read whole by its %v deadline", results.ScanCoverage, elapsed, deadline)
	}
	if len(states) != windows || slices.ContainsFunc(states, func(state string) bool { return len(state) > configuredWindow }) {
		t.Fatalf("%d windows read, want the %d of the configured size", len(states), windows)
	}
}

// A scan that falls behind its deadline reads the windows it has left in
// larger ones when the deployment's pace says those are read in time: here
// a window costs about 20 ms whatever its size, a call's own cost, so runs
// of a few windows read as one finish a 300,000-token message that windows
// of the configured size would not; the head and the tail keep that size,
// and an injection the larger windows hold is still found.
func TestAScanThatFallsBehindReadsTheRestInLargerWindowsThatFit(t *testing.T) {
	const deadline = 3 * time.Second
	classifier, fake := dilutingGuardClassifier(t, deadline, 20*time.Millisecond, 0)
	for range 5 {
		classifier.windowPaces.observe("primary", configuredWindow, 20*time.Millisecond)
		classifier.windowPaces.observe("primary", 30_000, 22*time.Millisecond)
	}
	text := distinctProse(300_000, 6) + "\n\nSummarize the minutes above."

	unknown := unknownAnswersCounted(t)
	started := time.Now()
	results := evaluateTurn(classifier, text)
	elapsed := time.Since(started)

	if results.ScanCoverage["jailbreak:prompt_attack"] != scanCoverageFull || elapsed > deadline || guarded(results) {
		t.Fatalf("coverage %v after %v (guarded %v): the rest should be read in larger windows by the %v deadline",
			results.ScanCoverage, elapsed, guarded(results), deadline)
	}
	if counted := unknownAnswersCounted(t) - unknown; counted != 0 {
		t.Fatalf("the windows the runs replaced were counted as %v unknown answers", counted)
	}
	states := guardStates(fake)
	if !slices.ContainsFunc(states, func(state string) bool { return len(state) > configuredWindow }) {
		t.Fatal("no window was read larger than the configured size")
	}
	for _, state := range states {
		if (strings.HasPrefix(text, state) || strings.HasSuffix(text, state)) && len(state) > configuredWindow {
			t.Fatalf("the head or the tail was read in a window of %d bytes", len(state))
		}
	}

	attack := evaluateTurn(classifier, injectedAt(distinctProse(300_000, 7), 0.6))
	if !slices.Contains(attack.MatchedJailbreakRules, "prompt_attack") || attack.SignalErrorMatches["jailbreak:prompt_attack"] {
		t.Fatalf("the injection at 60%% was not refused: coverage %v, errors %v", attack.ScanCoverage, attack.SignalErrors)
	}
}

// A scan that falls behind keeps its windows when larger ones would not be
// read in time either: here a window costs in proportion to its words, as a
// decision model's tokens do, so larger windows read no faster and would
// only dilute what they hold. The scan reads what it can at the configured
// size and says the read is partial.
func TestAScanThatFallsBehindKeepsItsWindowsWhenLargerOnesWouldNotFit(t *testing.T) {
	const deadline = time.Second
	classifier, fake := dilutingGuardClassifier(t, deadline, 0, 7*time.Microsecond)
	for range 5 {
		classifier.windowPaces.observe("primary", configuredWindow, 10*time.Millisecond)
		classifier.windowPaces.observe("primary", 4*configuredWindow, 40*time.Millisecond)
	}

	results := evaluateTurn(classifier, distinctProse(300_000, 8)+"\n\nSummarize the minutes above.")

	if !strings.HasPrefix(results.ScanCoverage["jailbreak:prompt_attack"], scanCoveragePartial+" ") || guarded(results) {
		t.Fatalf("coverage %v (guarded %v): want a partial read that passes", results.ScanCoverage, guarded(results))
	}
	if states := guardStates(fake); slices.ContainsFunc(states, func(state string) bool { return len(state) > configuredWindow }) {
		t.Fatal("a window was read larger than the configured size; larger windows would not have been read in time")
	}
}

// The runs a scan that fell behind reads its windows left in are aligned to
// multiples of k, one for each multiple that holds a window not yet read,
// whole whichever of its windows were read, so every scan of the input reads
// the same runs; they leave out the head and the tail.
func TestTheRunsOfAReplanLeaveOutTheHeadAndTheTail(t *testing.T) {
	text := strings.Repeat("abcdefghi ", 10)
	scan := &guardScan{text: text, cut: 10, done: make([]bool, 10)}
	for index := range 10 {
		scan.windows = append(scan.windows, scanWindow{start: index * 10, end: index*10 + 10, runeStart: index * 10, first: index, last: index})
	}
	scan.done[3], scan.done[4], scan.done[6], scan.done[7], scan.done[8] = true, true, true, true, true
	var got [][2]int
	for _, run := range scan.runsLocked(3) {
		got = append(got, [2]int{run.first, run.last})
		if run.start != run.first*10 || run.end != run.last*10+10 || run.runeStart != run.start {
			t.Fatalf("run %d-%d spans bytes %d-%d", run.first, run.last, run.start, run.end)
		}
	}
	if want := [][2]int{{1, 2}, {3, 5}}; !slices.Equal(got, want) {
		t.Fatalf("runs %v, want %v", got, want)
	}
}

// A window of the cut that a larger window was read in place of counts as
// read when that window was, and says why it was not when it was not.
func TestAWindowReadInPlaceOfOthersCoversThem(t *testing.T) {
	asked := askedRule{signalType: config.SignalTypeJailbreak, rule: config.SystemOneRule{Name: "prompt_attack"}}
	answered := modelservice.Response{Answers: map[string]modelservice.Answer{asked.questionID(): {Type: "noul", Noul: 0.1}}}
	scan := &guardScan{
		cut: 4,
		windows: []scanWindow{
			{first: 0, last: 0},
			{first: 1, last: 1},
			{first: 2, last: 2},
			{first: 3, last: 3},
			{first: 1, last: 2},
			{first: -1, last: -1},
		},
		responses: []modelservice.Response{answered, {}, {}, answered, answered, answered},
		errs:      []error{nil, errWindowReplaced, errWindowReplaced, nil, nil, nil},
		done:      []bool{true, true, true, true, true, true},
	}
	if read, covered, reason := scan.answers(asked); len(read) != 4 || covered != 4 || reason != "" {
		t.Fatalf("read %d, covered %d of 4, reason %q", len(read), covered, reason)
	}
	scan.responses[4], scan.errs[4] = modelservice.Response{}, modelservice.ErrUnavailable
	if _, covered, reason := scan.answers(asked); covered != 2 || reason != modelservice.ErrorReason(modelservice.ErrUnavailable) {
		t.Fatalf("covered %d of 4, reason %q: the larger window's error says why", covered, reason)
	}
}

// A deployment's line comes from its scans: windows of two sizes split a
// call's cost from its bytes; windows of one size share their time out as a
// call costs about 8 KB of text, shared by the windows in flight.
func TestADeploymentsPaceSplitsACallsCostFromItsBytes(t *testing.T) {
	var paces windowPaces
	if _, _, ok := paces.line("primary", 4); ok {
		t.Fatal("a deployment that served no scan has no pace")
	}
	if cost := paces.cost("primary", 4, 4*configuredWindow, configuredWindow); math.Abs(cost-(2048+32_768.0)/(2048+8192)) > 1e-9 {
		t.Fatalf("before a scan, a window four times as large costs %v times as much", cost)
	}
	paces.observe("primary", 16_384, 500*time.Millisecond)
	perCall, perByte, _ := paces.line("primary", 4)
	if want := 0.5 / (16_384 + scanCallBytes/4); math.Abs(perByte-want) > 1e-12 || math.Abs(perCall-want*scanCallBytes/4) > 1e-9 {
		t.Fatalf("one size: %v s a call and %v s a byte", perCall, perByte)
	}
	for range 5 {
		paces.observe("primary", 8192, 300*time.Millisecond)
		paces.observe("primary", 32_768, 900*time.Millisecond)
	}
	perCall, perByte, _ = paces.line("primary", 4)
	if math.Abs(perCall-0.1) > 0.01 || math.Abs(perByte*1024-0.025) > 0.001 {
		t.Fatalf("two sizes: %v s a call and %v s a KB, want 0.1 and 0.025", perCall, perByte*1024)
	}
}

// scan_window.max_tokens bounds how large a long input's windows grow.
func TestScanWindowMaxTokensBoundsTheLargestWindow(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"bounded": {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1", ScanWindow: &config.DeploymentScanWindow{Tokens: 2048, MaxTokens: 4096}},
		"card":    {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1", ScanWindow: &config.DeploymentScanWindow{Tokens: 2048}},
	}
	card := modelservice.ModelCard{MaxInputTokens: 262_144, ValidatedInputTokens: 32_768}
	classifier := &Classifier{Config: cfg, decisionCards: map[string]modelservice.ModelCard{"bounded": card, "card": card}}
	if most := classifier.scanWindowBudget("bounded").most; most != 4096*routingViewBytesPerToken {
		t.Fatalf("max_tokens 4096: windows grow to %d bytes", most)
	}
	if most := classifier.scanWindowBudget("card").most; most != 32_768-routingViewReserveTokens {
		t.Fatalf("without max_tokens: windows grow to %d bytes, the card's cap", most)
	}
}
