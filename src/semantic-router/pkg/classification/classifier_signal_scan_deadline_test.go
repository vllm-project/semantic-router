package classification

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// pacedGuardClassifier serves refusingGuardConfig, with the signals'
// deadline timeout, from a fake decision model validated at 65,536 tokens
// whose engine reads one state at a time, perWindow each.
func pacedGuardClassifier(t *testing.T, timeout, perWindow time.Duration) *Classifier {
	t.Helper()
	document := strings.NewReplacer("TIMEOUT", fmt.Sprint(timeout.Milliseconds()), "RULE\n", "").Replace(refusingGuardConfig)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"primary": {Family: presets.FamilyVela3D3, MaxInputTokens: 65_536, Answer: findsInjection},
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
	fake.SetStateTime(perWindow)
	return classifier
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

// A 950,000-token message is read whole by the signals' deadline once the
// deployment's pace is known, and an injection anywhere in it is found. In
// production's 2,048-token windows, grown so 256 cover it, the message takes
// 255 windows; at 10 ms a window, one at a time, they cannot all be read in
// 2 s, and the scan decided on the part it read. Larger windows, as large as
// the model takes, are all read in time.
func TestA950KMessageIsReadWholeByItsDeadlineAtTheMeasuredPace(t *testing.T) {
	const deadline = 2 * time.Second
	classifier := pacedGuardClassifier(t, deadline, 10*time.Millisecond)

	// The first long message shows the deployment's pace.
	evaluateTurn(classifier, distinctProse(950_000, 0)+"\n\nSummarize the minutes above.")

	started := time.Now()
	results := evaluateTurn(classifier, distinctProse(950_000, 1)+"\n\nSummarize the minutes above.")
	if elapsed := time.Since(started); results.ScanCoverage["jailbreak:prompt_attack"] != scanCoverageFull || elapsed > deadline {
		t.Fatalf("coverage %v after %v: the message should be read whole by its %v deadline", results.ScanCoverage, elapsed, deadline)
	}
	if guarded(results) {
		t.Fatalf("a benign message read whole passes: jailbreak %v, safety %v", results.MatchedJailbreakRules, results.MatchedSafetyRules)
	}

	for request, share := range []float64{0.05, 0.35, 0.6, 0.85, 1} {
		prose := distinctProse(950_000, request+2)
		at := min(len(prose), int(float64(len(prose))*share))
		at += strings.IndexByte(prose[at:]+" ", ' ') + 1
		text := prose[:min(at, len(prose))] + " " + scanInjection + " " + prose[min(at, len(prose)):] + "\n\nSummarize the minutes above."
		results := evaluateTurn(classifier, text)
		if !slices.Contains(results.MatchedJailbreakRules, "prompt_attack") || results.SignalErrorMatches["jailbreak:prompt_attack"] {
			t.Fatalf("the injection at %.0f%% was not read: coverage %v, errors %v", share*100, results.ScanCoverage, results.SignalErrors)
		}
	}
}

// Windows grow only as far as the time needs: an input whose windows of the
// configured size are all read in time keeps them; a longer one reads the
// smallest windows that are, and the largest when none are.
func TestWindowsGrowOnlyAsFarAsTheDeadlineNeeds(t *testing.T) {
	const size, most, overlap = 8192, 64_512, 1024
	budget := scanBudget{readTime: time.Second, perCall: 0.010}
	short := scanProse(100_000)
	if windows, chosen := sizedScanWindows(short, size, most, overlap, budget); chosen != size || len(windows)*10 > 1000 {
		t.Fatalf("%d windows of %d bytes for an input that fits at %d", len(windows), chosen, size)
	}
	long := scanProse(950_000)
	windows, chosen := sizedScanWindows(long, size, most, overlap, budget)
	if chosen <= size || chosen >= most || len(windows) > 100 {
		t.Fatalf("%d windows of %d bytes: want the smallest windows 100 of which cover the input", len(windows), chosen)
	}
	if smaller := windowCount(len(long), chosen-1, overlap); smaller <= 100 {
		t.Fatalf("windows of %d bytes would fit too (%d); %d is not the smallest", chosen-1, smaller, chosen)
	}
	budget.readTime = 100 * time.Millisecond
	if _, chosen := sizedScanWindows(long, size, most, overlap, budget); chosen != most {
		t.Fatalf("no size fits 100 ms; want the largest, %d, got %d", most, chosen)
	}
	if _, chosen := sizedScanWindows(long, size, most, overlap, scanBudget{}); chosen == most {
		t.Fatal("without a pace the windows only grow so 256 cover the input")
	}
}

// A deployment's line comes from its scans: windows of two sizes split a
// call's cost from its bytes; windows of one size share their time out as a
// call costs about 8 KB of text.
func TestADeploymentsPaceSplitsACallsCostFromItsBytes(t *testing.T) {
	var paces windowPaces
	if _, _, ok := paces.line("primary"); ok {
		t.Fatal("a deployment that served no scan has no pace")
	}
	paces.observe("primary", 16_384, 500*time.Millisecond)
	perCall, perByte, _ := paces.line("primary")
	if want := 0.5 / (16_384 + scanCallBytes); math.Abs(perByte-want) > 1e-12 || math.Abs(perCall-want*scanCallBytes) > 1e-9 {
		t.Fatalf("one size: %v s a call and %v s a byte", perCall, perByte)
	}
	for range 5 {
		paces.observe("primary", 8192, 300*time.Millisecond)
		paces.observe("primary", 32_768, 900*time.Millisecond)
	}
	perCall, perByte, _ = paces.line("primary")
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
