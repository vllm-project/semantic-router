package classification

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

// TestRequestImageEmbeddingCache_DedupsConcurrentResolves verifies the cache's
// core guarantee: when multiple goroutines resolve the same (imageRef, targetDim)
// concurrently, the underlying compute func runs exactly once and every caller
// observes the same embedding. This is the property that turns the two
// independent signal evaluators (complexity + embedding) into a single FFI
// encode per request.
func TestRequestImageEmbeddingCache_DedupsConcurrentResolves(t *testing.T) {
	cache := newRequestImageEmbeddingCache()
	var computeCount int32
	want := []float32{0.1, 0.2, 0.3}

	const goroutines = 8
	var wg sync.WaitGroup
	wg.Add(goroutines)

	results := make([][]float32, goroutines)
	errs := make([]error, goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			emb, err := cache.resolve("img-A", 0, func() ([]float32, error) {
				atomic.AddInt32(&computeCount, 1)
				return want, nil
			})
			results[i] = emb
			errs[i] = err
		}()
	}
	wg.Wait()

	if got := atomic.LoadInt32(&computeCount); got != 1 {
		t.Errorf("expected compute func to run exactly once, ran %d times", got)
	}
	for i := 0; i < goroutines; i++ {
		if errs[i] != nil {
			t.Errorf("goroutine %d got error %v", i, errs[i])
		}
		if !equalFloat32Slices(results[i], want) {
			t.Errorf("goroutine %d got embedding %v, want %v", i, results[i], want)
		}
	}
}

// TestRequestImageEmbeddingCache_DistinctImagesComputeIndependently confirms
// the cache keys on imageRef so two genuinely different images each pay
// their own FFI cost. (The cache deliberately does not key on targetDim;
// see TestRequestImageEmbeddingCache_DifferentTargetDimsShareFullEmbedding
// for the rationale.)
func TestRequestImageEmbeddingCache_DistinctImagesComputeIndependently(t *testing.T) {
	cache := newRequestImageEmbeddingCache()
	var computeCount int32

	embA, _ := cache.resolve("img-A", 0, func() ([]float32, error) {
		atomic.AddInt32(&computeCount, 1)
		return []float32{1.0}, nil
	})
	embB, _ := cache.resolve("img-B", 0, func() ([]float32, error) {
		atomic.AddInt32(&computeCount, 1)
		return []float32{2.0}, nil
	})

	if got := atomic.LoadInt32(&computeCount); got != 2 {
		t.Errorf("expected 2 compute calls (distinct images), got %d", got)
	}
	if embA[0] != 1.0 || embB[0] != 2.0 {
		t.Errorf("distinct images returned wrong embeddings: A=%v B=%v", embA, embB)
	}
}

// TestRequestImageEmbeddingCache_DifferentTargetDimsShareFullEmbedding is the
// load-bearing property test for the production deduplication case. The
// canonical config has complexity hardcoding targetDim=0 and embedding using
// optimizationConfig.TargetDimension (768 in canonical_defaults, often
// overridden to 384 for multimodal). A targetDim-keyed cache would silently
// fail to deduplicate across these signals; this cache uses MRL to serve any
// sub-dim caller from a single full-dim cache entry.
func TestRequestImageEmbeddingCache_DifferentTargetDimsShareFullEmbedding(t *testing.T) {
	cache := newRequestImageEmbeddingCache()
	var computeCount int32
	full := []float32{0.6, 0.8, 0.0, 0.0} // 0.6^2 + 0.8^2 = 1.0, already L2-normalized

	compute := func() ([]float32, error) {
		atomic.AddInt32(&computeCount, 1)
		return full, nil
	}

	fullView, err := cache.resolve("img-A", 0, compute)
	if err != nil {
		t.Fatalf("first resolve failed: %v", err)
	}
	truncatedView, err := cache.resolve("img-A", 2, compute)
	if err != nil {
		t.Fatalf("second resolve failed: %v", err)
	}

	if got := atomic.LoadInt32(&computeCount); got != 1 {
		t.Errorf("compute should run exactly once across different targetDim asks, ran %d times", got)
	}
	if !equalFloat32Slices(fullView, full) {
		t.Errorf("full-dim view should equal cached full embedding, got %v want %v", fullView, full)
	}
	// 2-dim prefix of [0.6, 0.8, 0.0, 0.0] is [0.6, 0.8]; already unit-norm so renormalize is a no-op.
	if len(truncatedView) != 2 || !approxEqual(float64(truncatedView[0]), 0.6) || !approxEqual(float64(truncatedView[1]), 0.8) {
		t.Errorf("truncated view should be [0.6, 0.8], got %v", truncatedView)
	}
}

// TestTruncateAndRenormalize confirms the post-hoc Matryoshka truncation
// produces a unit-norm prefix. The cache's lossless dedupe across targetDims
// hinges on this being a faithful mirror of candle-binding's .narrow + L2
// renormalize step.
func TestTruncateAndRenormalize(t *testing.T) {
	in := []float32{3.0, 4.0, 0.0, 0.0} // pre-renorm; magnitude of first 2 = 5
	got := truncateAndRenormalize(in, 2)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	// 3/5 = 0.6, 4/5 = 0.8
	if !approxEqual(float64(got[0]), 0.6) || !approxEqual(float64(got[1]), 0.8) {
		t.Errorf("got %v, want [0.6, 0.8]", got)
	}

	// targetDim=0 returns input unchanged
	pass := truncateAndRenormalize(in, 0)
	if !equalFloat32Slices(pass, in) {
		t.Errorf("targetDim=0 should pass through, got %v", pass)
	}

	// targetDim >= len returns input unchanged
	pass = truncateAndRenormalize(in, 10)
	if !equalFloat32Slices(pass, in) {
		t.Errorf("targetDim>=len should pass through, got %v", pass)
	}
}

// TestRequestImageEmbeddingCache_PropagatesErrorToAllCallers confirms that
// when the underlying FFI errors, every concurrent caller for the same key
// observes the same error rather than some racing through. Without this,
// a transient FFI failure could flap between "embedding present" and
// "embedding nil + log error" depending on goroutine ordering.
func TestRequestImageEmbeddingCache_PropagatesErrorToAllCallers(t *testing.T) {
	cache := newRequestImageEmbeddingCache()
	wantErr := errors.New("synthetic FFI failure")

	const goroutines = 4
	var wg sync.WaitGroup
	wg.Add(goroutines)
	errs := make([]error, goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			_, err := cache.resolve("img-A", 0, func() ([]float32, error) {
				return nil, wantErr
			})
			errs[i] = err
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if !errors.Is(err, wantErr) {
			t.Errorf("goroutine %d got error %v, want %v", i, err, wantErr)
		}
	}
}

// TestRequestImageEmbeddingCache_NilReceiverFallsThrough confirms a nil
// cache pointer is treated as cache-disabled and the compute func runs
// directly. This is the codepath callers outside EvaluateAllSignalsWithContext
// take (e.g. one-shot ClassifyDetailedMultimodal calls in tests).
func TestRequestImageEmbeddingCache_NilReceiverFallsThrough(t *testing.T) {
	var cache *requestImageEmbeddingCache
	var computeCount int32

	emb, err := cache.resolve("img-A", 0, func() ([]float32, error) {
		atomic.AddInt32(&computeCount, 1)
		return []float32{0.5}, nil
	})
	if err != nil {
		t.Fatalf("nil cache resolve returned error: %v", err)
	}
	if computeCount != 1 || emb[0] != 0.5 {
		t.Errorf("nil cache should run compute once and return its result, got count=%d emb=%v", computeCount, emb)
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
