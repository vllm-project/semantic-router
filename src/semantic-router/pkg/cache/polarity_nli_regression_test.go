//go:build !windows && cgo

package cache

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// Model-backed regression for #2751 through the production NLI binding and the
// in-memory lookup finisher. It skips without the hallucination explainer
// (models/mom-halugate-explainer, tasksource/ModernBERT-base-nli);
// polarity_nli_test.go provides model-free coverage.

// Each pair is {cached, incoming}; the incoming query means the opposite.
var polarityNLIContradictionPairs = [][2]string{
	{"How do I enable dark mode?", "How do I disable dark mode?"},
	{"How do I reset my password?", "How do I not reset my password?"},
	{"How do I open the file?", "How do I close the file?"},
	{"How do I start the server?", "How do I stop the server?"},
	{"How do I add a user?", "How do I remove a user?"},
}

// Paraphrases and synonyms must keep hitting; the NLI head treats
// delete/remove as compatible, which the lexical antonym table cannot.
var polarityNLIParaphrasePairs = [][2]string{
	{"How do I reset my password?", "How can I reset my password?"},
	{"How do I delete a user?", "How do I remove a user?"},
}

var nliExplainerInitOnce sync.Once

func findNLIExplainerModel() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for i := 0; i < 8; i++ {
		p := filepath.Join(dir, "models", "mom-halugate-explainer")
		if st, statErr := os.Stat(p); statErr == nil && st.IsDir() {
			return p
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

func ensureNLIExplainer(t *testing.T) {
	t.Helper()
	modelPath := findNLIExplainerModel()
	if modelPath == "" {
		t.Skip("NLI explainer model not found (models/mom-halugate-explainer); skipping model-backed regression")
	}
	var initErr error
	nliExplainerInitOnce.Do(func() {
		initErr = candle_binding.InitNLIModel(modelPath, true)
	})
	if initErr != nil {
		t.Fatalf("failed to initialize NLI model %s: %v", modelPath, initErr)
	}
}

// candleContradiction is the production verifier shape (see
// classification.semanticCachePolarityVerifier) built here directly so the
// cache package test does not import pkg/classification.
func candleContradiction(_ context.Context, cached, incoming string) (float32, error) {
	r, err := candle_binding.ClassifyNLI(cached, incoming)
	if err != nil {
		return 0, err
	}
	return r.ContradictProb, nil
}

func TestPolarityNLIRegression(t *testing.T) {
	ensureNLIExplainer(t)
	installVerifier(t, candleContradiction)

	c := NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: polarityTestThreshold,
		MaxEntries:          32,
		Enabled:             true,
		EvictionPolicy:      FIFOEvictionPolicyType,
		PolarityGuard:       PolarityGuardOptions{UseNLI: true, ContradictionThreshold: 0.5},
	})
	t.Cleanup(func() { _ = c.Close() })

	lookup := func(cached, incoming string) (LookupResult, float32) {
		entry := CacheEntry{
			RequestID: "r", Model: "m", Query: cached, ResponseBody: []byte("ANSWER"),
			Embedding: []float32{1, 0, 0}, Timestamp: time.Now(),
		}
		contradiction, err := candleContradiction(context.Background(), cached, incoming)
		if err != nil {
			t.Fatalf("NLI classification failed for %q / %q: %v", cached, incoming, err)
		}
		result, err := c.finishFindSimilarSearch(context.Background(), time.Now(), entry.Model, incoming,
			polarityTestThreshold, 0, entry, 1.0, 1, 0)
		if err != nil {
			t.Fatalf("lookup failed: %v", err)
		}
		return result, contradiction
	}

	for _, pair := range polarityNLIContradictionPairs {
		result, contradiction := lookup(pair[0], pair[1])
		if result.Found {
			t.Errorf("%q / %q: served despite contradiction=%.3f", pair[0], pair[1], contradiction)
		}
		t.Logf("reject  %-38q -> %-38q contradiction=%.3f", pair[0], pair[1], contradiction)
	}
	for _, pair := range polarityNLIParaphrasePairs {
		result, contradiction := lookup(pair[0], pair[1])
		if !result.Found {
			t.Errorf("%q / %q: rejected paraphrase, contradiction=%.3f", pair[0], pair[1], contradiction)
		}
		t.Logf("serve   %-38q -> %-38q contradiction=%.3f", pair[0], pair[1], contradiction)
	}
}
