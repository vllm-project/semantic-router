package classification

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

const (
	kbPreloadSecretOne = "confidential acquisition plan alpha"
	kbPreloadSecretTwo = "private customer identifier beta"
)

func TestKnowledgeBasePreloadAllFailuresAreAtomicAndRetryableRemote(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")

	classifier := newKBPreloadTestClassifier(config.EmbeddingBackendOpenAICompatible, nil)
	exemplarCount := kbPreloadExemplarCount(classifier)
	var calls atomic.Int64
	provider, err := embedding.NewFuncProvider(
		config.EmbeddingBackendOpenAICompatible,
		2,
		func(_ context.Context, text string) ([]float32, error) {
			if calls.Add(1) <= int64(exemplarCount) {
				return nil, fmt.Errorf("remote failure included exemplar %q", text)
			}
			return []float32{1, 0}, nil
		},
	)
	if err != nil {
		t.Fatalf("NewFuncProvider: %v", err)
	}
	classifier.provider = provider

	preloadErr := classifier.ensureEmbeddingsPreloaded()
	assertSafeKBPreloadError(t, preloadErr)
	assertKBPreloadNotPublished(t, classifier)
	if got := calls.Load(); got != int64(exemplarCount) {
		t.Fatalf("remote calls after failed preload = %d, want %d", got, exemplarCount)
	}

	if err := classifier.ensureEmbeddingsPreloaded(); err != nil {
		t.Fatalf("retry preload: %v", err)
	}
	assertKBPreloadPublished(t, classifier)
	if got := calls.Load(); got != int64(2*exemplarCount) {
		t.Fatalf("remote calls after retry = %d, want %d", got, 2*exemplarCount)
	}
}

func TestKnowledgeBasePreloadPartialFailureIsAtomicAndRetryableCandle(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendCandle)

	originalCandle := getEmbeddingWithModelType
	var calls atomic.Int64
	var failedOnce atomic.Bool
	getEmbeddingWithModelType = func(text, _ string, _ int) (*candle_binding.EmbeddingOutput, error) {
		calls.Add(1)
		if text == kbPreloadSecretOne && !failedOnce.Swap(true) {
			return nil, fmt.Errorf("local failure included exemplar %q", text)
		}
		return &candle_binding.EmbeddingOutput{Embedding: []float32{1, 0}}, nil
	}
	t.Cleanup(func() {
		getEmbeddingWithModelType = originalCandle
	})

	classifier := newKBPreloadTestClassifier(config.EmbeddingBackendCandle, nil)
	exemplarCount := kbPreloadExemplarCount(classifier)
	preloadErr := classifier.ensureEmbeddingsPreloaded()
	assertSafeKBPreloadError(t, preloadErr)
	assertKBPreloadNotPublished(t, classifier)
	if got := calls.Load(); got != int64(exemplarCount) {
		t.Fatalf("Candle calls after partial failure = %d, want %d", got, exemplarCount)
	}

	if err := classifier.ensureEmbeddingsPreloaded(); err != nil {
		t.Fatalf("retry preload: %v", err)
	}
	assertKBPreloadPublished(t, classifier)
	if got := calls.Load(); got != int64(2*exemplarCount) {
		t.Fatalf("Candle calls after retry = %d, want %d", got, 2*exemplarCount)
	}
}

func newKBPreloadTestClassifier(backend string, provider embedding.Provider) *KnowledgeBaseClassifier {
	return &KnowledgeBaseClassifier{
		rule:      (config.KnowledgeBaseConfig{Name: "preload-test"}).WithDefaults(),
		modelType: config.EmbeddingModelTypeQwen3,
		backend:   backend,
		provider:  provider,
		labels: map[string]*kbLabelData{
			"private": {
				Exemplars: []string{kbPreloadSecretOne, kbPreloadSecretTwo},
			},
			"public": {
				Exemplars: []string{"public documentation example"},
			},
		},
	}
}

func kbPreloadExemplarCount(classifier *KnowledgeBaseClassifier) int {
	total := 0
	for _, data := range classifier.labels {
		total += len(data.Exemplars)
	}
	return total
}

func assertSafeKBPreloadError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("preload succeeded, want failure")
	}
	errorText := err.Error()
	for _, sensitive := range []string{kbPreloadSecretOne, kbPreloadSecretTwo, "remote failure", "local failure"} {
		if strings.Contains(errorText, sensitive) {
			t.Fatalf("preload error exposed sensitive upstream detail %q: %v", sensitive, err)
		}
	}
	if !strings.Contains(errorText, "of 3 exemplars") {
		t.Fatalf("preload error = %q, want aggregate failure count", errorText)
	}
}

func assertKBPreloadNotPublished(t *testing.T, classifier *KnowledgeBaseClassifier) {
	t.Helper()
	if classifier.preloaded {
		t.Fatal("failed preload marked classifier as preloaded")
	}
	for label, data := range classifier.labels {
		if len(data.Embeddings) != 0 {
			t.Fatalf("failed preload published %d embeddings for label %q", len(data.Embeddings), label)
		}
		if data.Prototype != nil {
			t.Fatalf("failed preload published prototype for label %q", label)
		}
	}
}

func assertKBPreloadPublished(t *testing.T, classifier *KnowledgeBaseClassifier) {
	t.Helper()
	if !classifier.preloaded {
		t.Fatal("successful retry did not mark classifier as preloaded")
	}
	for label, data := range classifier.labels {
		if len(data.Embeddings) != len(data.Exemplars) {
			t.Fatalf("published embeddings for label %q = %d, want %d", label, len(data.Embeddings), len(data.Exemplars))
		}
		for i, value := range data.Embeddings {
			if len(value) == 0 {
				t.Fatalf("published empty embedding for label %q exemplar %d", label, i)
			}
		}
		if data.Prototype == nil {
			t.Fatalf("successful retry did not publish prototype for label %q", label)
		}
	}
}
