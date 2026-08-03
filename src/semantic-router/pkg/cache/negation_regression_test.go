//go:build !windows && cgo

package cache

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// End-to-end regression for #2691: negated / antonym variants of a cached query
// must not be served that entry's cached answer, while genuine paraphrases must
// still hit. This drives the real production cache path
// (NewInMemoryCache{EmbeddingModel:"mmbert"} → AddEntry → FindSimilarWithThreshold),
// so it needs the mmbert embedding model and cgo; it is skipped when the model
// is not present (e.g. a CI job without models fetched). The pure-function guard
// logic is covered separately, without a model, in polarity_test.go.
//
// PAWS-style hard negatives: high lexical overlap, opposite meaning. The full
// PAWS/QQP calibration corpus is the offline threshold-calibration tool's input
// (a separate PR), not duplicated here.

const negationRegressionThreshold = 0.80 // config/plugin/semantic-cache/memory.yaml ships 0.80

// Each pair is {cached, incoming}: the cache is seeded with an answer for
// `cached`, then queried with the opposite-meaning `incoming`.
var negationRegressionPairs = [][2]string{
	{"How to turn on dark mode?", "How to turn off dark mode?"},
	{"Is port 6379 open by default?", "Is port 6379 closed by default?"},
	{"Should I commit this change?", "Should I not commit this change?"},
	{"Does this feature require a license?", "Does this feature not require a license?"},
	{"Is caching enabled in production?", "Is caching disabled in production?"},
	{"Is the API rate limit currently active?", "Is the API rate limit currently inactive?"},
	{"Can I increase my storage quota?", "Can I decrease my storage quota?"},
	{"How do I enable two-factor authentication?", "How do I disable two-factor authentication?"},
}

// Genuine paraphrases (equivalent meaning, high surface overlap) that clear the
// threshold on the mmbert path — the guard must keep hitting them, proving it
// does not eat legitimate recall. These differ only by a non-polarity token
// swap (do/can, dropped article), so polarityMismatch must not fire on them.
var paraphraseControlPairs = [][2]string{
	{"How do I reset my password?", "How can I reset my password?"},
	{"Where are the logs stored?", "Where are logs stored?"},
	{"How do I update my email address?", "How can I update my email address?"},
}

var mmbertInitOnce sync.Once

// findMmbertModel walks up from the test's working directory looking for the
// locally-fetched mmbert embedding model, returning "" if it is not present.
func findMmbertModel() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for i := 0; i < 8; i++ {
		p := filepath.Join(dir, "models", "mmbert-embed-32k-2d-matryoshka")
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

func ensureMmbert(t *testing.T) {
	t.Helper()
	modelPath := findMmbertModel()
	if modelPath == "" {
		t.Skip("mmbert embedding model not found (models/mmbert-embed-32k-2d-matryoshka); skipping cgo regression")
	}
	var initErr error
	mmbertInitOnce.Do(func() {
		initErr = candle_binding.InitMmBertEmbeddingModel(modelPath, true)
	})
	if initErr != nil {
		t.Skipf("mmbert init failed (%v); skipping cgo regression", initErr)
	}
}

func newMmbertCache(t *testing.T) *InMemoryCache {
	t.Helper()
	return NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: negationRegressionThreshold,
		MaxEntries:          16,
		TTLSeconds:          3600, // 0 means "not cached"; use a long TTL so entries stay live
		Enabled:             true,
		EvictionPolicy:      FIFOEvictionPolicyType,
		EmbeddingModel:      "mmbert",
	})
}

func TestNegationFalseHitRegressionInMemory(t *testing.T) {
	ensureMmbert(t)

	guardExercised := runNegationRegressionPairs(t)
	if guardExercised == 0 {
		t.Fatalf("no negation pair cleared threshold %.2f, so the guard was never exercised — embedder/threshold mismatch, the regression is not actually testing anything",
			negationRegressionThreshold)
	}
	t.Logf("polarity guard exercised on %d/%d above-threshold negation pairs", guardExercised, len(negationRegressionPairs))

	paraphraseExercised := runParaphraseControlPairs(t)
	if paraphraseExercised == 0 {
		t.Fatalf("no paraphrase control pair cleared threshold %.2f, so the guard-does-not-eat-legit-hits property was never verified",
			negationRegressionThreshold)
	}
	t.Logf("paraphrase control exercised on %d/%d above-threshold pairs", paraphraseExercised, len(paraphraseControlPairs))
}

func runNegationRegressionPairs(t *testing.T) int {
	t.Helper()
	guardExercised := 0
	for _, pair := range negationRegressionPairs {
		cached, incoming := pair[0], pair[1]
		c := newMmbertCache(t)
		cachedAnswer := []byte("CACHED-ANSWER-FOR::" + cached)
		if err := c.AddEntry("neg", "model-x", cached, []byte("req"), cachedAnswer, 3600); err != nil {
			t.Fatalf("AddEntry(%q): %v", cached, err)
		}

		body, hit, err := c.FindSimilarWithThreshold("model-x", incoming, negationRegressionThreshold)
		if err != nil {
			t.Fatalf("FindSimilarWithThreshold(%q): %v", incoming, err)
		}
		sim := float64(c.LastSimilarity())

		if sim >= negationRegressionThreshold {
			// The embedding cleared the threshold: absent the guard this WOULD
			// be a false hit. Assert the guard rejected it.
			guardExercised++
			if hit {
				t.Errorf("negation false-hit: incoming %q matched cached %q at sim=%.4f >= %.2f and returned %q; polarity guard should have rejected it",
					incoming, cached, sim, negationRegressionThreshold, string(body))
			}
		} else {
			t.Logf("below-threshold genuine miss (guard N/A): %q vs %q sim=%.4f", incoming, cached, sim)
		}
		_ = c.Close()
	}
	return guardExercised
}

// runParaphraseControlPairs verifies that the guard preserves genuine cache hits.
func runParaphraseControlPairs(t *testing.T) int {
	t.Helper()
	paraphraseExercised := 0
	for _, pair := range paraphraseControlPairs {
		cached, incoming := pair[0], pair[1]
		c := newMmbertCache(t)
		if err := c.AddEntry("para", "model-x", cached, []byte("req"), []byte("PARAPHRASE-ANSWER"), 3600); err != nil {
			t.Fatalf("AddEntry(%q): %v", cached, err)
		}
		body, hit, err := c.FindSimilarWithThreshold("model-x", incoming, negationRegressionThreshold)
		if err != nil {
			t.Fatalf("FindSimilarWithThreshold(%q): %v", incoming, err)
		}
		sim := float64(c.LastSimilarity())
		if sim >= negationRegressionThreshold {
			paraphraseExercised++
			if !hit {
				t.Errorf("paraphrase regression: %q vs %q sim=%.4f >= %.2f but was rejected; genuine recall lost (body=%q)",
					incoming, cached, sim, negationRegressionThreshold, string(body))
			} else if !strings.Contains(string(body), "PARAPHRASE-ANSWER") {
				t.Errorf("paraphrase %q returned unexpected body %q", incoming, string(body))
			}
		} else {
			t.Logf("paraphrase below threshold (control N/A): %q vs %q sim=%.4f", incoming, cached, sim)
		}
		_ = c.Close()
	}
	return paraphraseExercised
}
