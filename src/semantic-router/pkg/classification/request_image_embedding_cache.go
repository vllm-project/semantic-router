package classification

import (
	"math"
	"sync"
)

// requestImageEmbeddingCache memoizes a single image embedding computation
// across the lifetime of one EvaluateAllSignalsWithContext call. Both the
// embedding signal and the complexity signal can be configured against the
// same image; without this cache they would each pay a full SigLIP forward
// pass FFI call for the same input, since their computations live in two
// independently-launched goroutines and neither sees the other's work.
//
// The cache always computes at the model's full embedding dimension and
// keys on imageRef alone. Callers wanting a Matryoshka-truncated view pass
// their target dim to resolve(); the cache returns the prefix of the cached
// full-dim embedding, L2-renormalized. The model is MRL-trained so a
// truncated-and-renormalized prefix of the full embedding is mathematically
// equivalent to a full-dim FFI call asking for the truncated dim directly
// (candle-binding does the same .narrow + renormalize internally). This
// matters because complexity hardcodes targetDim=0 while embedding can be
// configured with any TargetDimension (default 768 in canonical_defaults,
// often overridden to 384 for the multimodal model); a targetDim-keyed cache
// would silently fail to deduplicate any non-default configuration.
//
// The cache is request-scoped: the orchestrator (EvaluateAllSignalsWithContext)
// allocates a fresh one on entry and lets it go out of scope on exit. This is
// intentional - cross-request caching of attached-image embeddings would need
// a real eviction policy and content-addressed keying, which is out of scope
// for the deduplication this type exists to solve.
type requestImageEmbeddingCache struct {
	mu      sync.Mutex
	entries map[string]*requestImageEmbeddingCacheEntry
}

type requestImageEmbeddingCacheEntry struct {
	once      sync.Once
	embedding []float32 // always full-dim
	err       error
}

func newRequestImageEmbeddingCache() *requestImageEmbeddingCache {
	return &requestImageEmbeddingCache{
		entries: make(map[string]*requestImageEmbeddingCacheEntry),
	}
}

// resolve returns the embedding for imageRef truncated to targetDim, computing
// the full-dim embedding via compute on the first call for this imageRef.
// Concurrent callers for the same imageRef block on the same sync.Once and
// observe the same full-dim result, then independently produce their own
// truncated view (truncation does not mutate the cached slice).
//
// compute MUST return the full-dim embedding (callers should pass targetDim=0
// to the underlying FFI). targetDim=0 or targetDim >= len(full) returns the
// full embedding unchanged.
//
// A nil receiver is treated as cache-disabled: compute runs unconditionally
// and the result is not memoized, then truncated on return. This lets callers
// outside the orchestrator (tests, single-shot calls) share the same call
// site as cached callers.
func (c *requestImageEmbeddingCache) resolve(imageRef string, targetDim int, compute func() ([]float32, error)) ([]float32, error) {
	if c == nil {
		full, err := compute()
		if err != nil {
			return nil, err
		}
		return truncateAndRenormalize(full, targetDim), nil
	}
	c.mu.Lock()
	entry, ok := c.entries[imageRef]
	if !ok {
		entry = &requestImageEmbeddingCacheEntry{}
		c.entries[imageRef] = entry
	}
	c.mu.Unlock()
	entry.once.Do(func() {
		entry.embedding, entry.err = compute()
	})
	if entry.err != nil {
		return nil, entry.err
	}
	return truncateAndRenormalize(entry.embedding, targetDim), nil
}

// truncateAndRenormalize returns the first targetDim entries of v, L2-renormalized
// in place (on a fresh slice; the input is not mutated). targetDim <= 0 or
// targetDim >= len(v) returns v unchanged. Mirrors the .narrow + renormalize
// step in candle-binding's MRL truncation path so a cached full-dim embedding
// can serve any sub-dim caller losslessly under MRL.
func truncateAndRenormalize(v []float32, targetDim int) []float32 {
	if targetDim <= 0 || targetDim >= len(v) {
		return v
	}
	out := make([]float32, targetDim)
	var norm float64
	for i := 0; i < targetDim; i++ {
		out[i] = v[i]
		norm += float64(v[i]) * float64(v[i])
	}
	if norm > 0 {
		scale := float32(1.0 / math.Sqrt(norm))
		for i := range out {
			out[i] *= scale
		}
	}
	return out
}
