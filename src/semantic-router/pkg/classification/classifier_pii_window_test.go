package classification

import (
	"context"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestPIIWindowInputsPreserveUnpreparedFallback(t *testing.T) {
	text := strings.Repeat("alpha ", 400)
	for _, classifier := range []*Classifier{nil, {}} {
		if got := classifier.piiInputs(text); !reflect.DeepEqual(got, piiSignalChunks(text)) {
			t.Fatal("unprepared classifier changed legacy inputs")
		}
		if got := classifier.piiInputSpans(text); !reflect.DeepEqual(got, piiSignalChunkSpans(text)) {
			t.Fatal("unprepared classifier changed legacy byte spans")
		}
	}
}

func TestDefaultPIIWindowUsesCompleteInputWithoutChangingSource(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	original := cfg.PIIModel
	models, err := newClassifierModelRuntime(&cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := models.cfg.PIIModel
	registered := config.GetModelByPath(config.DefaultSystemModels().PIIClassifier)
	if got.Window == nil || got.Window.Size != 512 || got.Window.Overlap != 255 || got.MaxSequenceLength != registered.MaxContextLength {
		t.Fatalf("incorrect default: %+v", got)
	}
	if !reflect.DeepEqual(cfg.PIIModel, original) {
		t.Fatal("runtime resolution changed source")
	}
	initializer, inference, err := buildPIIDependencies(models.cfg, nil, models)
	if err != nil {
		t.Fatal(err)
	}
	windowed, ok := inference.(*windowedPIIBackend)
	if !ok || initializer != windowed || windowed.spec.Deployment.Input.Overflow != "window" {
		t.Fatalf("wrong native task: %T", inference)
	}
	c := &Classifier{Config: models.cfg}
	text := strings.Repeat("𐐀", 400)
	if inputs := c.piiInputs(text); len(inputs) != 1 || inputs[0] != text {
		t.Fatal("estimated chunks survived before native token windows")
	}
	spans := c.piiInputSpans(text)
	if len(spans) != 1 || spans[0].Text != text || spans[0].StartByte != 0 {
		t.Fatal("details API lost global offset origin")
	}
}

func TestDefaultPIIWindowRespectsArtifactSelection(t *testing.T) {
	registered := config.GetModelByPath(config.DefaultSystemModels().PIIClassifier)
	absolute, err := filepath.Abs(registered.LocalPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name, artifact string
		windowed       bool
	}{
		{"custom", "models/custom-modernbert-pii-8k", false},
		{"legacy", "models/mmbert32k-pii-detector-merged", false},
		{"same basename elsewhere", filepath.Join(t.TempDir(), filepath.Base(registered.LocalPath)), false},
		{"registered alias", registered.Aliases[0], true},
		{"relative path", "./" + registered.LocalPath, true},
		{"absolute path", absolute, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := config.DefaultGlobalConfig()
			cfg.PIIModel.ModelID = test.artifact
			original := cfg.PIIModel
			models, err := newClassifierModelRuntime(&cfg, nil)
			if err != nil {
				t.Fatal(err)
			}
			resolved := models.cfg.PIIModel
			if !reflect.DeepEqual(cfg.PIIModel, original) {
				t.Fatal("runtime preparation changed the source policy")
			}
			if test.windowed {
				if resolved.MaxSequenceLength != registered.MaxContextLength || resolved.Window == nil {
					t.Fatalf("default artifact lost its window policy: %+v", resolved)
				}
			} else if !reflect.DeepEqual(resolved, original) {
				t.Fatalf("custom artifact inherited the default model's policy: %+v", resolved)
			}
			_, inference, err := buildPIIDependencies(models.cfg, nil, models)
			if err != nil {
				t.Fatal(err)
			}
			_, windowed := inference.(*windowedPIIBackend)
			if windowed != test.windowed {
				t.Fatalf("unexpected preparation backend: %T", inference)
			}
		})
	}
}

func TestDefaultPIIWindowPreservesExplicitPolicies(t *testing.T) {
	for name, change := range map[string]func(*config.RouterConfig){
		"512":  func(c *config.RouterConfig) { c.PIIModel.MaxSequenceLength = 512 },
		"8192": func(c *config.RouterConfig) { c.PIIModel.MaxSequenceLength = 8192 },
		"window": func(c *config.RouterConfig) {
			c.PIIModel.Window = &config.SequenceHeadWindowConfig{Size: 128, Overlap: 63}
		},
		"remote":   func(c *config.RouterConfig) { c.PIIModel.Backend = &config.RemoteClassifierBackend{} },
		"adapter":  func(c *config.RouterConfig) { c.PIIModel.UseMmBERT32K = false },
		"disabled": func(c *config.RouterConfig) { v := false; c.PIIModel.Enabled = &v },
	} {
		t.Run(name, func(t *testing.T) {
			cfg := config.DefaultGlobalConfig()
			change(&cfg)
			original := cfg.PIIModel
			m := &classifierModelRuntime{cfg: &cfg}
			if err := m.resolveDefaultPIIWindow(); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(original, cfg.PIIModel) {
				t.Fatal("explicit policy changed")
			}
		})
	}
	cfg := config.DefaultGlobalConfig()
	cfg.ModelDeployments = map[string]config.ModelDeployment{"pii-amd": {Artifact: config.DefaultSystemModels().PIIClassifier, Provider: "ort", Device: "migraphx:0", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 8192, Overflow: "reject"}}}
	cfg.ModelBindings = map[string]config.ModelBinding{"pii_classifier": {Deployment: "pii-amd", Adapter: "mmbert32k", Contract: config.RemoteClassifierContractTokenSpans}}
	models, err := newClassifierModelRuntime(&cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, inference, err := buildPIIDependencies(models.cfg, nil, models)
	if err != nil {
		t.Fatal(err)
	}
	owned, ok := inference.(*ownedTokenBackend)
	if !ok || owned.spec.Deployment.Input.MaxTokens != 8192 || owned.spec.Deployment.Input.Overflow != "reject" || models.cfg.PIIModel.Window != nil {
		t.Fatalf("explicit deployment changed: %T", inference)
	}
}

type capturedWindowPII struct {
	inputs []string
	text   string
	err    error
}

func (b *capturedWindowPII) ClassifyTokens(_ context.Context, text string) (tasks.TokenClassificationResult, error) {
	b.inputs = append(b.inputs, text)
	if text != b.text {
		return tasks.TokenClassificationResult{}, fmt.Errorf("received an estimated fragment")
	}
	if b.err != nil {
		return tasks.TokenClassificationResult{}, b.err
	}
	available := true
	return tasks.TokenClassificationResult{ScoresAvailable: &available, Input: &tasks.InputUsage{OriginalTokens: 1604, ProcessedTokens: 1604}, Entities: []tasks.TokenEntity{{EntityType: "PERSON", Text: "猫", Start: len(text) - 3, End: len(text), Confidence: .8}}}, nil
}

func TestPIIWindowFullInputReachesSignalsAndDetailOffsets(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	models, err := newClassifierModelRuntime(&cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("𐐀", 400) + "猫"
	backend := &capturedWindowPII{text: text}
	c := &Classifier{Config: models.cfg, PIIMapping: &PIIMapping{IdxToLabel: map[string]string{"0": "NONE", "1": "PERSON"}, LabelToIdx: map[string]int{"NONE": 0, "PERSON": 1}}, piiInference: backend}
	details, err := c.ClassifyPIIWithDetailsAndThreshold(context.Background(), text, .7)
	if err != nil || len(details) != 1 || details[0].Start != len(text)-3 || details[0].End != len(text) || len(backend.inputs) != 1 {
		t.Fatalf("lost full input/global details: %+v %v", details, err)
	}
	c.Config.PIIRules = []config.PIIRule{{Name: "low", Threshold: .7}, {Name: "high", Threshold: .9}}
	results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalConfidences: map[string]float64{}}
	c.evaluatePIISignal(context.Background(), results, &sync.Mutex{}, text, nil)
	if len(backend.inputs) != 2 || !reflect.DeepEqual(results.MatchedPIIRules, []string{"low"}) {
		t.Fatalf("rules did not share the exact scan: calls=%d rules=%v", len(backend.inputs), results.MatchedPIIRules)
	}
	backend.err = fmt.Errorf("native window failed")
	failed := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalConfidences: map[string]float64{}}
	c.evaluatePIISignal(context.Background(), failed, &sync.Mutex{}, text, nil)
	if len(failed.MatchedPIIRules) != 0 || len(failed.SignalErrors) != 2 {
		t.Fatalf("failed window looked clean: %+v", failed.SignalErrors)
	}
}
