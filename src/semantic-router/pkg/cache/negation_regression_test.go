//go:build !windows && cgo

package cache

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// Model-backed regression for #2691 through the production in-memory cache
// path. Set VLLM_SR_MMBERT_TEST_MODEL explicitly; polarity_test.go provides
// model-free coverage when this opt-in model test is not selected.
// The original pairs are a focused smoke set. The separate PAWS-derived
// fixture has pinned provenance and explicitly labelled local negatives;
// neither is a PAWS/QQP calibration corpus.

const negationRegressionThreshold = 0.80 // config/plugin/semantic-cache/memory.yaml ships 0.80

// Each pair is {cached, incoming}.
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

// High-overlap paraphrases ensure the guard preserves legitimate recall.
var paraphraseControlPairs = [][2]string{
	{"How do I reset my password?", "How can I reset my password?"},
	{"Where are the logs stored?", "Where are logs stored?"},
	{"How do I update my email address?", "How can I update my email address?"},
}

func prepareMmbert(t *testing.T) embedding.Provider {
	t.Helper()
	modelPath := os.Getenv("VLLM_SR_MMBERT_TEST_MODEL")
	if modelPath == "" {
		t.Skip("set VLLM_SR_MMBERT_TEST_MODEL to run the model-backed polarity regression")
	}
	provider, err := native.New(nil).Embedding(context.Background(), config.ResolvedModelBinding{
		Recipe: "cache-regression", Name: "embedding",
		Binding: config.ModelBinding{Deployment: "mmbert-test", Contract: "embedding.v1", Adapter: "mmbert"},
		Deployment: config.ModelDeployment{
			Artifact: modelPath, Provider: "candle", Device: "cpu", Precision: "float32",
			Input: config.ModelInputBudget{MaxTokens: 512, Overflow: "reject"},
		},
	}, mmbertMemoryCacheDimension, mmbertMemoryCacheLayer)
	if err != nil {
		t.Fatalf("mmbert provider preparation failed for explicit model %s: %v", modelPath, err)
	}
	t.Cleanup(func() {
		if err := provider.Close(); err != nil {
			t.Errorf("close mmbert provider: %v", err)
		}
	})
	return provider
}

func newMmbertCache(t *testing.T, provider embedding.Provider) *InMemoryCache {
	t.Helper()
	return NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: negationRegressionThreshold,
		MaxEntries:          16,
		TTLSeconds:          3600, // 0 means "not cached"; use a long TTL so entries stay live
		Enabled:             true,
		EvictionPolicy:      FIFOEvictionPolicyType,
		EmbeddingModel:      "mmbert",
		EmbeddingProvider:   provider,
	})
}

func TestNegationFalseHitRegressionInMemory(t *testing.T) {
	provider := prepareMmbert(t)

	guardExercised := runNegationRegressionPairs(t, provider)
	if guardExercised == 0 {
		t.Fatalf("no negation pair cleared threshold %.2f, so the guard was never exercised — embedder/threshold mismatch, the regression is not actually testing anything",
			negationRegressionThreshold)
	}
	t.Logf("polarity guard exercised on %d/%d above-threshold negation pairs", guardExercised, len(negationRegressionPairs))

	paraphraseExercised := runParaphraseControlPairs(t, provider)
	if paraphraseExercised == 0 {
		t.Fatalf("no paraphrase control pair cleared threshold %.2f, so the guard-does-not-eat-legit-hits property was never verified",
			negationRegressionThreshold)
	}
	t.Logf("paraphrase control exercised on %d/%d above-threshold pairs", paraphraseExercised, len(paraphraseControlPairs))
	runPAWSPolarityModelRegression(t, provider)
}

func runNegationRegressionPairs(t *testing.T, provider embedding.Provider) int {
	t.Helper()
	guardExercised := 0
	for _, pair := range negationRegressionPairs {
		cached, incoming := pair[0], pair[1]
		c := newMmbertCache(t, provider)
		cachedAnswer := []byte("CACHED-ANSWER-FOR::" + cached)
		if err := c.AddEntry(context.Background(), "neg", "model-x", cached, []byte("req"), cachedAnswer, 3600); err != nil {
			t.Fatalf("AddEntry(%q): %v", cached, err)
		}

		result, err := c.LookupSimilarWithThreshold(
			context.Background(), "model-x", incoming, negationRegressionThreshold,
		)
		if err != nil {
			t.Fatalf("LookupSimilarWithThreshold(%q): %v", incoming, err)
		}
		sim := float64(result.Similarity)

		if sim >= negationRegressionThreshold {
			guardExercised++
			if result.Found {
				t.Errorf("negation false-hit: incoming %q matched cached %q at sim=%.4f >= %.2f and returned %q; polarity guard should have rejected it",
					incoming, cached, sim, negationRegressionThreshold, string(result.ResponseBody))
			}
		} else {
			t.Logf("below-threshold genuine miss (guard N/A): %q vs %q sim=%.4f", incoming, cached, sim)
		}
		_ = c.Close()
	}
	return guardExercised
}

func runParaphraseControlPairs(t *testing.T, provider embedding.Provider) int {
	t.Helper()
	paraphraseExercised := 0
	for _, pair := range paraphraseControlPairs {
		cached, incoming := pair[0], pair[1]
		c := newMmbertCache(t, provider)
		if err := c.AddEntry(context.Background(), "para", "model-x", cached, []byte("req"), []byte("PARAPHRASE-ANSWER"), 3600); err != nil {
			t.Fatalf("AddEntry(%q): %v", cached, err)
		}
		result, err := c.LookupSimilarWithThreshold(
			context.Background(), "model-x", incoming, negationRegressionThreshold,
		)
		if err != nil {
			t.Fatalf("LookupSimilarWithThreshold(%q): %v", incoming, err)
		}
		sim := float64(result.Similarity)
		if sim >= negationRegressionThreshold {
			paraphraseExercised++
			if !result.Found {
				t.Errorf("paraphrase regression: %q vs %q sim=%.4f >= %.2f but was rejected; genuine recall lost (body=%q)",
					incoming, cached, sim, negationRegressionThreshold, string(result.ResponseBody))
			} else if !strings.Contains(string(result.ResponseBody), "PARAPHRASE-ANSWER") {
				t.Errorf("paraphrase %q returned unexpected body %q", incoming, string(result.ResponseBody))
			}
		} else {
			t.Logf("paraphrase below threshold (control N/A): %q vs %q sim=%.4f", incoming, cached, sim)
		}
		_ = c.Close()
	}
	return paraphraseExercised
}
