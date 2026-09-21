package classification

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestRequestMediaCacheDeduplicatesConcurrentRequests(t *testing.T) {
	cache := newRequestMediaEmbeddingCache()
	var calls atomic.Int32
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := cache.resolveFor(nil, config.QueryModalityImage, "image", 0, func() ([]float32, error) { calls.Add(1); return []float32{1, 0}, nil })
			if err != nil || !equalFloat32Slices(result, []float32{1, 0}) {
				t.Errorf("resolve = %v, %v", result, err)
			}
		}()
	}
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("computed %d times", calls.Load())
	}
}

func TestRequestMediaCacheSeparatesRepresentationContracts(t *testing.T) {
	cache := newRequestMediaEmbeddingCache()
	first, _ := embedding.NewFuncProvider("first", 4, func(context.Context, string) ([]float32, error) { return nil, nil })
	second, _ := embedding.NewFuncProvider("second", 4, func(context.Context, string) ([]float32, error) { return nil, nil })
	calls := 0
	compute := func() ([]float32, error) { calls++; return []float32{float32(calls)}, nil }
	for _, request := range []struct {
		provider  embedding.Provider
		modality  config.QueryModality
		payload   string
		dimension int
	}{
		{first, config.QueryModalityImage, "same", 0},
		{first, config.QueryModalityImage, "same", 4}, // explicit full dimension is the same view
		{first, config.QueryModalityImage, "same", 2}, // provider must validate this view
		{first, config.QueryModalityAudio, "same", 0},
		{first, config.QueryModalityImage, "different", 0},
		{second, config.QueryModalityImage, "same", 0},
	} {
		_, err := cache.resolveFor(request.provider, request.modality, request.payload, request.dimension, compute)
		if err != nil {
			t.Fatal(err)
		}
	}
	if calls != 5 {
		t.Fatalf("different contracts shared results: computed %d times", calls)
	}
}

func TestRequestMediaCachePreservesProviderErrorsAndVectors(t *testing.T) {
	cache := newRequestMediaEmbeddingCache()
	failure := errors.New("dimension unsupported")
	calls := 0
	for range 2 {
		_, err := cache.resolveFor(nil, config.QueryModalityImage, "a", 2, func() ([]float32, error) { calls++; return nil, failure })
		if !errors.Is(err, failure) {
			t.Fatalf("lost provider error: %v", err)
		}
	}
	if calls != 1 {
		t.Fatalf("recomputed failure %d times", calls)
	}
	var disabled *requestMediaEmbeddingCache
	want := []float32{3, 4, 5}
	got, err := disabled.resolveFor(nil, config.QueryModalityImage, "a", 2, func() ([]float32, error) { return want, nil })
	if err != nil || !equalFloat32Slices(got, want) {
		t.Fatalf("cache synthesized a vector: %v, %v", got, err)
	}
}

func equalFloat32Slices(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
