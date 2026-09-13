package classification

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

var (
	embedderMu             sync.Mutex
	embedderOverrideActive bool
)

// SetEmbeddingFuncForTests overrides the embedding generator for tests/benchmarks.
// It returns a restore function that must be called to revert to the original implementation.
func SetEmbeddingFuncForTests(fn func(string, string, int) (*tasks.EmbeddingResult, error)) func() {
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
