package cache

import "sync"

// defaultEmbeddingMemoSize bounds the embedding memo. Each entry costs roughly
// embedding_dim * 4 bytes (e.g. ~1KB at 256 dims), so 512 entries is well under
// a few MB even for 1024-dim models.
const defaultEmbeddingMemoSize = 512

// embeddingMemo is a small, bounded, concurrency-safe cache of query text ->
// embedding.
//
// An embedding is a deterministic pure function of (query text, configured
// model), so memoizing it is always correct: the worst case under a race is a
// redundant recompute, never a wrong vector. The memo removes the duplicate
// embedding inference that otherwise runs on every cache-miss request, where
// the lookup (FindSimilarWithThreshold) and the pending-entry write
// (AddPendingRequest) each embed the same query. It also serves repeated
// identical queries across requests.
type embeddingMemo struct {
	mu      sync.Mutex
	entries map[string][]float32
	order   []string // insertion order, for FIFO eviction
	maxSize int
}

func newEmbeddingMemo(maxSize int) *embeddingMemo {
	if maxSize <= 0 {
		maxSize = defaultEmbeddingMemoSize
	}
	return &embeddingMemo{
		entries: make(map[string][]float32, maxSize),
		order:   make([]string, 0, maxSize),
		maxSize: maxSize,
	}
}

func (m *embeddingMemo) get(key string) ([]float32, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	v, ok := m.entries[key]
	return v, ok
}

func (m *embeddingMemo) put(key string, value []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.entries[key]; exists {
		return
	}
	if len(m.order) >= m.maxSize {
		oldest := m.order[0]
		m.order = m.order[1:]
		delete(m.entries, oldest)
	}
	m.entries[key] = value
	m.order = append(m.order, key)
}

// getOrCompute returns the memoized embedding for key, computing and storing it
// via compute on a miss.
//
// compute runs WITHOUT the memo lock held, so concurrent embeddings are never
// serialized behind one another (holding the lock across model inference would
// be far worse than the rare duplicate compute a racing miss can cause).
// Compute errors are propagated and not memoized, so a transient failure is
// retried on the next call.
func (m *embeddingMemo) getOrCompute(key string, compute func(string) ([]float32, error)) ([]float32, error) {
	if v, ok := m.get(key); ok {
		return v, nil
	}
	v, err := compute(key)
	if err != nil {
		return nil, err
	}
	m.put(key, v)
	return v, nil
}
