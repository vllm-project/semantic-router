package classification

import (
	"context"
	"errors"
	"io"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type fakeJailbreakWindowModel struct {
	windows []tasks.LabelDistributionWindow
	usage   *tasks.InputUsage
	limits  binding.Limits
	inputs  []string
	options []tasks.TextWindowsRequest
	err     error
	closed  int
}

func (m *fakeJailbreakWindowModel) ClassifyWindows(text string, options tasks.TextWindowsRequest) (tasks.WindowedLabelDistribution, error) {
	m.inputs = append(m.inputs, text)
	m.options = append(m.options, options)
	return tasks.WindowedLabelDistribution{Windows: m.windows, Input: m.usage}, m.err
}

func (m *fakeJailbreakWindowModel) Close() error { m.closed++; return nil }

func windowedGuardFixture(t *testing.T) (*windowedJailbreakBackend, *fakeJailbreakWindowModel, *Classifier) {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.PromptGuard = config.PromptGuardConfig{
		Enabled: true, ModelID: "guard-fixture", JailbreakMappingPath: "fixture-mapping",
		Variant: config.PromptGuardVariantMmBERT32K, UseCPU: true,
		MaxSequenceLength: 32768, Threshold: .5,
		PositiveLabels: []string{"jailbreak", "injection"},
		Window:         &config.SequenceHeadWindowConfig{Size: 128, Overlap: 63},
	}
	mapping := &JailbreakMapping{
		LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1, "injection": 2},
		IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak", "2": "injection"},
	}
	backend, err := newWindowedJailbreakBackend(cfg.PromptGuard, mapping)
	if err != nil {
		t.Fatal(err)
	}
	model := &fakeJailbreakWindowModel{windows: []tasks.LabelDistributionWindow{
		{Probabilities: []float32{.2, .7, .1}}, {Probabilities: []float32{.1, .3, .6}},
	}, limits: binding.Limits{ModelTokens: 32768, TaskTokens: 128, DocumentTokens: 32768, DeploymentTokens: 32768, Overflow: "window"}}
	backend.prepare = func(ctx context.Context) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution], error) {
		if backend.spec.Deployment.Device != "cpu" {
			t.Fatalf("incorrect typed native contract: %+v", backend.spec)
		}
		resource, err := binding.NewPool().Acquire(ctx, binding.ResourceIdentity{Artifact: "fixture", Provider: "candle", Device: "cpu", Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return model, nil })
		if err != nil {
			return nil, err
		}
		task, err := binding.Register(binding.NewRegistry(), config.RemoteClassifierContractLabelDistribution, func(tasks.TextWindowsRequest) error { return nil }, func(tasks.TextWindowsRequest, tasks.WindowedLabelDistribution) error { return nil })
		if err != nil {
			return nil, err
		}
		return task.Resolve(binding.Identity{Recipe: string(backend.spec.Recipe), Name: "prompt_guard", Deployment: "fixture", Contract: config.RemoteClassifierContractLabelDistribution, Adapter: "modernbert"}, binding.Capability{Contract: config.RemoteClassifierContractLabelDistribution, Provider: "candle", Device: "cpu", Precision: "fp32", Labels: append([]string(nil), backend.labels...), Limits: model.limits, Window: &binding.WindowCapability{Size: backend.window.Size, Overlap: backend.window.Overlap}}, resource, func(_ context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelDistribution, error) {
			return model.ClassifyWindows(input.Text, input)
		})
	}
	if err := backend.Init(cfg.PromptGuard.ModelID, true, 3); err != nil {
		t.Fatal(err)
	}
	return backend, model, &Classifier{Config: cfg, JailbreakMapping: mapping, jailbreakInference: backend}
}

func TestWindowedJailbreakPreservesOneRealDistribution(t *testing.T) {
	backend, model, classifier := windowedGuardFixture(t)
	result, err := backend.Classify(context.Background(), "request")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Probabilities, model.windows[1].Probabilities) {
		t.Fatalf("mixed windows or chose the largest individual class: %+v", result)
	}
	risk := jailbreakRiskScore(classifier.JailbreakMapping, classifier.Config.PromptGuard.PositiveLabels, result)
	if math.Abs(float64(risk-.9)) > 1e-6 {
		t.Fatalf("incorrect combined positive risk: %v", risk)
	}
	result.Probabilities[0] = 1
	if model.windows[1].Probabilities[0] != .1 {
		t.Fatal("result aliases native scores")
	}
}

func TestWindowedJailbreakAPIReceivesFullInputAndNativeFailure(t *testing.T) {
	_, model, classifier := windowedGuardFixture(t)
	input := strings.Repeat("context 中文 <s> ", 1000) + "end"
	scan, err := classifier.ScanJailbreakRisk(context.Background(), input)
	if err != nil || scan.RiskScore < .89 {
		t.Fatalf("scan: %+v, %v", scan, err)
	}
	if !reflect.DeepEqual(model.inputs, []string{input}) ||
		!reflect.DeepEqual(model.options, []tasks.TextWindowsRequest{{Text: input, Size: 128, Overlap: 63}}) {
		t.Fatal("input was split or retokenized before native inference")
	}
	if chunks := classifier.jailbreakModelInputs(""); len(chunks) != 0 {
		t.Fatal("empty input was scored")
	}
	model.err = errors.New("total token budget exceeded")
	if _, err := classifier.ScanJailbreakRisk(context.Background(), input); err == nil {
		t.Fatal("native overflow became a clean verdict")
	}
}

func TestWindowedJailbreakRejectsIncompleteScoresAndHonorsLifecycle(t *testing.T) {
	backend, model, _ := windowedGuardFixture(t)
	for _, scores := range [][]float32{nil, {.5, .5}, {.2, .2, .2}, {.1, .2, float32(math.NaN())}} {
		model.windows = []tasks.LabelDistributionWindow{{Probabilities: []float32{.1, .3, .6}}, {Probabilities: scores}}
		if _, err := backend.Classify(context.Background(), "request"); err == nil {
			t.Fatal("accepted malformed window")
		}
	}
	model.windows = nil
	if _, err := backend.Classify(context.Background(), "request"); err == nil {
		t.Fatal("accepted empty scan")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	before := len(model.inputs)
	if _, err := backend.Classify(ctx, "request"); !errors.Is(err, context.Canceled) || len(model.inputs) != before {
		t.Fatal("canceled request entered native inference")
	}
	if err := backend.Close(); err != nil {
		t.Fatal(err)
	}
	if err := backend.Close(); err != nil || model.closed != 1 {
		t.Fatal("model closed more than once")
	}
	if _, err := backend.Classify(context.Background(), "request"); err == nil {
		t.Fatal("used closed model")
	}
	if err := backend.Init("guard-fixture", true, 3); err == nil {
		t.Fatal("reopened closed model")
	}
}

func TestWindowedJailbreakBinaryRiskBelowArgmaxAndRuleCache(t *testing.T) {
	backend, model, classifier := windowedGuardFixture(t)
	backend.labels, backend.positive = []string{"benign", "jailbreak"}, []int{1}
	classifier.JailbreakMapping = &JailbreakMapping{
		LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1},
		IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"},
	}
	classifier.Config.PromptGuard.PositiveLabels = []string{"jailbreak"}
	classifier.Config.PromptGuard.Threshold = .44471272826194763
	model.windows = []tasks.LabelDistributionWindow{{Probabilities: []float32{.99, .01}}, {Probabilities: []float32{.55, .45}}}
	detected, label, confidence, risk, err := classifier.CheckForJailbreakWithRisk(context.Background(), "request")
	if err != nil || !detected || label != "benign" || confidence != .55 || risk != .45 {
		t.Fatalf("positive risk was replaced with argmax confidence: %v %s %v %v %v", detected, label, confidence, risk, err)
	}
	classifier.Config.JailbreakRules = []config.JailbreakRule{
		{Name: "low", Threshold: .4}, {Name: "high", Threshold: .5},
	}
	results := &SignalResults{SignalConfidences: make(map[string]float64), Metrics: &SignalMetricsCollection{}}
	before := len(model.inputs)
	classifier.evaluateJailbreakSignal(context.Background(), results, &sync.Mutex{}, "request", nil)
	if len(model.inputs) != before+1 || !reflect.DeepEqual(results.MatchedJailbreakRules, []string{"low"}) {
		t.Fatalf("rules did not share a scan or threshold independently: calls=%d results=%+v", len(model.inputs)-before, results)
	}
}

func TestWindowedJailbreakDependencyUsesOneOwnedHandle(t *testing.T) {
	_, _, classifier := windowedGuardFixture(t)
	initializer, inference, err := buildJailbreakDependencies(classifier.Config, classifier.JailbreakMapping)
	if err != nil {
		t.Fatal(err)
	}
	backend, ok := initializer.(*windowedJailbreakBackend)
	if !ok || inference != backend {
		t.Fatal("initializer and inference do not share the owned handle")
	}
	if _, _, err := buildJailbreakDependencies(classifier.Config, nil); err != nil {
		t.Fatal("dormant model required weights")
	}
	classifier.Config.PromptGuard.PositiveLabels = []string{"missing"}
	if _, _, err := buildJailbreakDependencies(classifier.Config, classifier.JailbreakMapping); err == nil {
		t.Fatal("accepted an unknown positive label")
	}
}

func TestWindowedJailbreakAdmitsDocumentBeyondForwardCapacity(t *testing.T) {
	backend, model, _ := windowedGuardFixture(t)
	if err := backend.Close(); err != nil {
		t.Fatal(err)
	}
	model.closed = 0
	model.limits = binding.Limits{ModelTokens: 32768, TaskTokens: 32768, DocumentTokens: 262144, DeploymentTokens: 262144, Overflow: "window"}
	backend.closed, backend.handle = false, nil
	backend.spec.Deployment.Input.MaxTokens = 262144
	backend.window.Size, backend.window.Overlap = 32768, 16383
	if err := backend.Init("fixture", true, 3); err != nil {
		t.Fatal(err)
	}
	defer backend.Close()
	model.usage = &tasks.InputUsage{OriginalTokens: 42000, ProcessedTokens: 42000}
	if _, err := backend.Classify(context.Background(), "complete document"); err != nil {
		t.Fatal(err)
	}
	if len(model.options) != 1 || model.options[0].Size != 32768 {
		t.Fatal("changed prepared forward window")
	}
}
