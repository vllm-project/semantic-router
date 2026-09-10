//go:build !windows && cgo

package classification

import (
	"context"
	"strings"
	"testing"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func polarityTestConfig(enabled bool, mode, nliModel string) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.SemanticCache.Enabled = enabled
	if mode != "" {
		cfg.SemanticCache.PolarityGuard = &config.PolarityGuardConfig{Mode: mode}
	}
	cfg.HallucinationMitigation.NLIModel.ModelID = nliModel
	return cfg
}

func TestNeedsSemanticCacheNLIForRuntime(t *testing.T) {
	var nilClassifier *Classifier
	if nilClassifier.needsSemanticCacheNLIForRuntime() {
		t.Fatal("nil classifier must not require the NLI model")
	}
	if (&Classifier{}).needsSemanticCacheNLIForRuntime() {
		t.Fatal("classifier without config must not require the NLI model")
	}

	cases := []struct {
		name string
		cfg  *config.RouterConfig
		want bool
	}{
		{"nli_mode", polarityTestConfig(true, "nli", "models/mom-halugate-explainer"), true},
		{"lexical_nli_mode", polarityTestConfig(true, "lexical+nli", "models/mom-halugate-explainer"), true},
		{"lexical_mode", polarityTestConfig(true, "lexical", "models/mom-halugate-explainer"), false},
		{"cache_disabled", polarityTestConfig(false, "nli", "models/mom-halugate-explainer"), false},
		{"no_model", polarityTestConfig(true, "nli", ""), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := &Classifier{Config: tc.cfg}
			if got := c.needsSemanticCacheNLIForRuntime(); got != tc.want {
				t.Fatalf("needsSemanticCacheNLIForRuntime = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestSemanticCacheNLIRuntimeTaskRegistration(t *testing.T) {
	c := &Classifier{Config: polarityTestConfig(true, "nli", "models/mom-halugate-explainer")}
	var task *string
	for _, candidate := range c.runtimeTasks() {
		if candidate.Name == "classifier.semantic_cache_nli" {
			name := candidate.Name
			task = &name
			if candidate.BestEffort {
				t.Fatal("semantic cache NLI init must not be best-effort: an unloadable model has to fail startup")
			}
		}
	}
	if task == nil {
		t.Fatal("semantic cache NLI task not registered for an NLI polarity mode")
	}

	off := &Classifier{Config: polarityTestConfig(true, "lexical", "models/mom-halugate-explainer")}
	for _, candidate := range off.runtimeTasks() {
		if candidate.Name == "classifier.semantic_cache_nli" {
			t.Fatal("semantic cache NLI task must not be registered for the lexical mode")
		}
	}
}

func TestInitializeSemanticCacheNLIFailsOnUnloadableModel(t *testing.T) {
	if candle.IsNLIModelInitialized() {
		t.Skip("an NLI model is already loaded in this process; the failure path cannot be observed")
	}
	c := &Classifier{Config: polarityTestConfig(true, "nli", "/nonexistent/semantic-cache-nli-model")}
	err := c.initializeSemanticCacheNLI()
	if err == nil {
		t.Fatal("expected initialization to fail for an unloadable model path")
	}
	if !strings.Contains(err.Error(), "semantic cache polarity guard") && !strings.Contains(err.Error(), "does not support local NLI") {
		t.Fatalf("error should name the polarity guard or the backend gap, got: %v", err)
	}
}

func TestSemanticCachePolarityVerifierSurfacesBackendErrors(t *testing.T) {
	if candle.IsNLIModelInitialized() {
		t.Skip("an NLI model is loaded; the uninitialized error path cannot be observed")
	}
	if _, err := semanticCachePolarityVerifier(context.Background(), "premise", "hypothesis"); err == nil {
		t.Fatal("verifier must return the binding error when no NLI model is loaded (the cache then degrades the candidate to a miss)")
	}
}
