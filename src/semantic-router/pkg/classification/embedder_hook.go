package classification

import (
	"sync"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

var (
	embedderMu             sync.Mutex
	embedderOverrideActive bool
)

// SetEmbeddingFuncForTests overrides the embedding generator for tests/benchmarks.
// It returns a restore function that must be called to revert to the original implementation.
func SetEmbeddingFuncForTests(fn func(string, string, int) (*candle_binding.EmbeddingOutput, error)) func() {
	embedderMu.Lock()
	orig := getEmbeddingWithModelType
	origOverride := embedderOverrideActive
	getEmbeddingWithModelType = fn
	embedderOverrideActive = true
	embedderMu.Unlock()

	return func() {
		embedderMu.Lock()
		getEmbeddingWithModelType = orig
		embedderOverrideActive = origOverride
		embedderMu.Unlock()
	}
}
