package classification

import (
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestDefaultJailbreakWindowUsesRegistryBudgetWithoutChangingSource(t *testing.T) {
	cfg := vela1GuardConfig()
	original := cfg.PromptGuard
	models, err := newClassifierModelRuntime(&cfg, RecipeRuntimeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	resolved := models.cfg.PromptGuard
	registered := config.GetModelByPath(config.Vela1SystemModels().PromptGuard)
	if resolved.MaxSequenceLength != registered.MaxContextLength || resolved.Window == nil || resolved.Window.Size != 512 || resolved.Window.Overlap != 255 {
		t.Fatalf("incorrect bounded default: %+v", resolved)
	}
	if !reflect.DeepEqual(cfg.PromptGuard, original) {
		t.Fatal("resolving a runtime default mutated the source configuration")
	}
	mapping := newRiskTestClassifier(nil).JailbreakMapping
	_, inference, err := buildJailbreakDependencies(models.cfg, mapping, models)
	if err != nil {
		t.Fatal(err)
	}
	windowed, ok := inference.(*windowedJailbreakBackend)
	if !ok || windowed.spec.Deployment.Input.MaxTokens != registered.MaxContextLength {
		t.Fatalf("default did not construct the typed native window backend: %T", inference)
	}
	c := &Classifier{Config: models.cfg}
	text := strings.Repeat("sample ", 500)
	if inputs := c.jailbreakModelInputs(text); len(inputs) != 1 || inputs[0] != text {
		t.Fatal("the exact-window backend received estimated text chunks")
	}
}

func TestDefaultJailbreakWindowPreservesExplicitPolicies(t *testing.T) {
	for name, change := range map[string]func(*config.RouterConfig){
		"explicit short budget": func(c *config.RouterConfig) { c.PromptGuard.MaxSequenceLength = 512 },
		"explicit long budget":  func(c *config.RouterConfig) { c.PromptGuard.MaxSequenceLength = 8192 },
		"explicit window": func(c *config.RouterConfig) {
			c.PromptGuard.MaxSequenceLength = 8192
			c.PromptGuard.Window = &config.SequenceHeadWindowConfig{Size: 128, Overlap: 63}
		},
		"disabled": func(c *config.RouterConfig) { c.PromptGuard.Enabled = false },
		"remote backend": func(c *config.RouterConfig) {
			c.PromptGuard.Backend = &config.RemoteClassifierBackend{Model: "remote", Protocol: config.RemoteClassifierProtocolHTTPClassify}
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := vela1GuardConfig()
			change(&cfg)
			original := cfg.PromptGuard
			// Resolve the preparation policy without preparing a remote or native model.
			models := &classifierModelRuntime{cfg: &cfg}
			if err := models.resolveDefaultJailbreakWindow(); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(cfg.PromptGuard, original) {
				t.Fatalf("explicit policy changed: got=%+v want=%+v", cfg.PromptGuard, original)
			}
		})
	}
}

func TestDefaultJailbreakWindowRespectsArtifactSelection(t *testing.T) {
	registered := config.GetModelByPath(config.Vela1SystemModels().PromptGuard)
	absolute, err := filepath.Abs(registered.LocalPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name, artifact string
		windowed       bool
	}{
		{"custom", "models/custom-modernbert-guard-8k", false},
		{"legacy", "models/mmbert32k-jailbreak-detector-merged", false},
		{"same basename elsewhere", filepath.Join(t.TempDir(), filepath.Base(registered.LocalPath)), false},
		{"registered alias", registered.Aliases[0], true},
		{"relative path", "./" + registered.LocalPath, true},
		{"absolute path", absolute, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := vela1GuardConfig()
			cfg.PromptGuard.ModelID = test.artifact
			original := cfg.PromptGuard
			models, err := newClassifierModelRuntime(&cfg, RecipeRuntimeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			resolved := models.cfg.PromptGuard
			if !reflect.DeepEqual(cfg.PromptGuard, original) {
				t.Fatal("runtime preparation changed the source policy")
			}
			if test.windowed {
				if resolved.MaxSequenceLength != registered.MaxContextLength || resolved.Window == nil {
					t.Fatalf("default artifact lost its window policy: %+v", resolved)
				}
			} else if !reflect.DeepEqual(resolved, original) {
				t.Fatalf("custom artifact inherited the default model's policy: %+v", resolved)
			}
			_, inference, err := buildJailbreakDependencies(models.cfg, newRiskTestClassifier(nil).JailbreakMapping, models)
			if err != nil {
				t.Fatal(err)
			}
			_, windowed := inference.(*windowedJailbreakBackend)
			if windowed != test.windowed {
				t.Fatalf("unexpected preparation backend: %T", inference)
			}
		})
	}
}

func TestDefaultJailbreakWindowPreservesROCmDeployment(t *testing.T) {
	cfg := vela1GuardConfig()
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"guard-amd": {Artifact: config.Vela1SystemModels().PromptGuard, Provider: config.ModelRuntimeProvider, Device: "rocm:0", Input: config.ModelInputBudget{MaxTokens: 8192, Overflow: "reject"}},
	}
	cfg.ModelBindings = map[string]config.ModelBinding{
		"prompt_guard": {Deployment: "guard-amd", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelDistribution},
	}
	models, err := newClassifierModelRuntime(&cfg, RecipeRuntimeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if models.cfg.PromptGuard.Window != nil || models.cfg.PromptGuard.MaxSequenceLength != 8192 {
		t.Fatalf("explicit ROCm execution changed: %+v", models.cfg.PromptGuard)
	}
	_, inference, err := buildJailbreakDependencies(models.cfg, newRiskTestClassifier(nil).JailbreakMapping, models)
	if err != nil {
		t.Fatal(err)
	}
	owned, ok := inference.(*ownedSequenceBackend)
	if !ok || owned.spec.Deployment.Input.Overflow != "reject" || owned.spec.Deployment.Device != "rocm:0" {
		t.Fatalf("explicit owned backend was replaced: %T", inference)
	}
}

func TestWindowedJailbreakValidatesPreparedDocumentCapacity(t *testing.T) {
	for _, limits := range []binding.Limits{
		{},
		{ModelTokens: 512, TaskTokens: 512},
		{ModelTokens: 32768, TaskTokens: 512},
	} {
		backend, model, classifier := windowedGuardFixture(t)
		if err := backend.Close(); err != nil {
			t.Fatal(err)
		}
		model.closed = 0
		model.limits = limits
		candidate, err := newWindowedJailbreakBackend(classifier.Config.PromptGuard, classifier.JailbreakMapping)
		if err != nil {
			t.Fatal(err)
		}
		candidate.prepare = backend.prepare
		if err := candidate.Init("fixture", true, 3); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("unsupported prepared capacity was accepted: limits=%+v err=%v", limits, err)
		}
		if model.closed != 1 {
			t.Fatalf("rejected prepared handle leaked: close count=%d", model.closed)
		}
	}
}

func TestDefaultJailbreakWindowPreservesContrastiveInputs(t *testing.T) {
	text := strings.Repeat("sample ", 500)
	for name, explicit := range map[string]int{"implicit default": 0, "explicit long budget": 8192, "explicit same window": 32768} {
		t.Run(name, func(t *testing.T) {
			cfg := vela1GuardConfig()
			cfg.PromptGuard.MaxSequenceLength = explicit
			if explicit == 32768 {
				cfg.PromptGuard.Window = &config.SequenceHeadWindowConfig{Size: 512, Overlap: 255}
			}
			original := (&Classifier{Config: &cfg}).jailbreakInputs(text)
			if explicit == 0 && len(original) < 2 {
				t.Fatal("neutral fixture must exercise the original contrastive chunk boundary")
			}
			models, err := newClassifierModelRuntime(&cfg, RecipeRuntimeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			classifier := &Classifier{Config: models.cfg, models: models}
			if got := classifier.jailbreakInputs(text); !reflect.DeepEqual(got, original) {
				t.Fatalf("native default changed contrastive inputs: got %d pieces, want %d", len(got), len(original))
			}
			if got := classifier.jailbreakModelInputs(text); len(got) != 1 || got[0] != text {
				t.Fatal("native input no longer preserves the complete text piece")
			}
		})
	}
}

// vela1GuardConfig is the default configuration with the prompt guard pinned
// to Vela 1.0 Guard, whose 32K document scan the Router resolves itself.
func vela1GuardConfig() config.RouterConfig {
	cfg := config.DefaultGlobalConfig()
	cfg.PromptGuard.ModelID = config.Vela1SystemModels().PromptGuard
	return cfg
}
