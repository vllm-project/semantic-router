package classification

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// guardScanConfig is production's guard (jailbreak at 0.995, safety at 0.9,
// both deciding the guard lane) and one routing question, on one decision
// model; TIMEOUT is the signals' deadline.
const guardScanConfig = `
version: v0.5
providers:
  defaults:
    model: general-model
  models:
    - name: general-model
      backend_refs:
        - name: backend
          endpoint: localhost:8000
          protocol: http
global:
  model_catalog:
    signal_timeout_ms: TIMEOUT
    deployments:
      primary:
        provider: model_runtime
        endpoint: http://primary.invalid:8100
    system:
      decision_model:
        deployment: primary
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_attack
          predicate: {gte: 0.995}
      safety:
        - name: unsafe_request
          predicate: {gte: 0.9}
      systemone:
        - name: task
          question:
            type: choice
            instructions: What kind of work does this request ask for?
            choices:
              - key: chat
              - key: writing
    decisions:
      - name: guard
        priority: 1000
        rules:
          operator: OR
          conditions:
            - {type: jailbreak, name: prompt_attack}
            - {type: safety, name: unsafe_request}
        modelRefs:
          - model: general-model
      - name: writing
        priority: 100
        rules:
          operator: AND
          conditions:
            - {type: systemone, name: task, label: writing}
        modelRefs:
          - model: general-model
`

// scanProse is benign prose of about tokens tokens of four and a half bytes,
// no two of its windows alike.
func scanProse(tokens int) string {
	var prose strings.Builder
	for item := 1; prose.Len() < tokens*9/2; item++ {
		fmt.Fprintf(&prose, "Item %d: the committee reviewed the quarterly report on regional water supplies and agreed to meet again. ", item)
	}
	return prose.String()
}

// guarded reports whether a guard rule matched, not one of its labels.
func guarded(results *SignalResults) bool {
	return slices.Contains(results.MatchedJailbreakRules, "prompt_attack") || slices.Contains(results.MatchedSafetyRules, "unsafe_request")
}

// guardScanClassifier serves guardScanConfig with signals' deadline timeout
// from a fake vela3-turbo validated at 32,768 tokens.
func guardScanClassifier(t *testing.T, timeout time.Duration) (*Classifier, *runtimetest.Runtime) {
	t.Helper()
	document := strings.Replace(guardScanConfig, "TIMEOUT", fmt.Sprint(timeout.Milliseconds()), 1)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"primary": {Family: presets.FamilyVela3D3, MaxInputTokens: 32_768},
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
	return classifier, fake
}

func evaluateTurn(classifier *Classifier, text string) *SignalResults {
	return classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: text, CurrentUserText: text, RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
}

// A 120K-token message passes the guard on the windows read by the signals'
// deadline, and reports how much of it the guard read. The fake reads a
// window in about 130 ms, the stage's other question too, one at a time:
// production's measured rate (0.5 s per thousand tokens, a 45 s deadline)
// scaled down 30 times. Sent in one call, no window would be read in time
// and the guard would refuse the message unread; read window by window, the
// tail and the head are read first, and the routing question does not wait
// for the scan.
func TestALongMessagePassesOnTheWindowsReadByItsDeadline(t *testing.T) {
	const deadline = 1500 * time.Millisecond
	classifier, fake := guardScanClassifier(t, deadline)
	fake.SetWordTime(25 * time.Microsecond)
	text := scanProse(120_000) + "\n\nSummarize the minutes above."
	started := time.Now()
	results := evaluateTurn(classifier, text)
	elapsed := time.Since(started)

	if guarded(results) {
		t.Fatalf("a benign message read in part passes best effort: jailbreak %v, safety %v, errors %v, coverage %v",
			results.MatchedJailbreakRules, results.MatchedSafetyRules, results.SignalErrors, results.ScanCoverage)
	}
	for _, rule := range []string{"jailbreak:prompt_attack", "safety:unsafe_request"} {
		var read, windows int
		if _, err := fmt.Sscanf(results.ScanCoverage[rule], "partial %d/%d", &read, &windows); err != nil || read < 2 || windows < 15 {
			t.Fatalf("%s coverage %q: the windows read by the deadline, the tail and the head among them", rule, results.ScanCoverage[rule])
		}
	}
	if _, failed := results.SignalErrors["systemone:task"]; failed {
		t.Fatalf("the routing question waited for the scan: %v", results.SignalErrors)
	}
	if elapsed > deadline+time.Second {
		t.Fatalf("the stage took %v; its signals' deadline is %v", elapsed, deadline)
	}
}

// pacedDecider answers after a pause and records how many of its calls ran at
// once and the order of the windows it read.
type pacedDecider struct {
	running, most atomic.Int32
	mu            sync.Mutex
	windows       []int
}

func (d *pacedDecider) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	now := d.running.Add(1)
	defer d.running.Add(-1)
	for old := d.most.Load(); now > old && !d.most.CompareAndSwap(old, now); old = d.most.Load() {
	}
	d.mu.Lock()
	d.windows = append(d.windows, request.Window)
	d.mu.Unlock()
	time.Sleep(20 * time.Millisecond)
	answers := map[string]modelservice.Answer{}
	for _, question := range request.Questions {
		answers[question.ID] = modelservice.Answer{Type: "choice", Choice: "benign", Probabilities: map[string]float64{"benign": 0.99, "jailbreak": 0.01}, InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: answers}, nil
}

// Outside a stage's bundle (the response stage, masking) a scan reads its
// windows a few at a time, the deployment's max_parallel, in reading order.
func TestWithoutABundleAScanReadsItsWindowsAFewAtATime(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Deployment: "kai"}}
	decider := &pacedDecider{}
	classifier := newApplicationTestClassifier(t, cfg, decider)
	results := newSignalResults()
	classifier.evaluateApplicationSignals(context.Background(), results, &sync.Mutex{}, config.SignalTypeJailbreak,
		systemOneReading{request: scanProse(80_000)}, usedApplicationRules(cfg))
	if most := int(decider.most.Load()); most != config.DefaultScanWindowParallel {
		t.Fatalf("%d windows read at once; the deployment reads %d", most, config.DefaultScanWindowParallel)
	}
	if len(decider.windows) < 8 || !slices.Contains(decider.windows[:config.DefaultScanWindowParallel], 1) {
		t.Fatalf("windows read in the order %v: the tail first", decider.windows)
	}
	if coverage := results.ScanCoverage["jailbreak:attack"]; coverage != scanCoverageFull {
		t.Fatalf("coverage %q", coverage)
	}
}
