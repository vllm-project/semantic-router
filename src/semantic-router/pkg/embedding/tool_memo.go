package embedding

import "sync"

// DefaultToolEmbeddingMemoSize bounds the tool embedding memo. Each entry costs
// roughly embedding_dim * 4 bytes (~3KB at 768 dims), so the default holds the
// memo under ~6MB even for 768-dim models (entries * dim * 4 bytes).
const DefaultToolEmbeddingMemoSize = 2048

// ToolEmbeddingMemo is a small, bounded, concurrency-safe cache of
// embedding-key -> embedding vector, used to reuse tool-definition embeddings
// across requests instead of re-embedding every request-supplied tool on every
// request.
//
// There is deliberately no TTL. Invalidation is structural: the key embeds both
// the text being embedded and the identity of the model that embedded it, so a
// renamed tool, an edited description, or a reconfigured embedding model
// produces a different key and therefore a different entry. An embedding is a
// deterministic pure function of (text, model), so a memoized vector can never
// go stale - it can only become unreachable, at which point FIFO eviction
// retires it.
//
// Concurrent misses on the same key may each compute the embedding and both
// call Put; that is intentional. Computation happens outside this type (and so
// outside its lock), because holding a lock across model inference or a remote
// embedding call would serialize every request behind the slowest one. The
// worst case under a race is redundant work, never a wrong vector.
//
// Stored slices are shared with callers, which only read them (dot products).
// Neither the memo nor its callers mutate a stored vector.
type ToolEmbeddingMemo struct {
	mu      sync.Mutex
	entries map[string][]float32
	order   []string // insertion order, for FIFO eviction
	maxSize int
}

// NewToolEmbeddingMemo returns a memo bounded to maxSize entries.
// A non-positive maxSize selects DefaultToolEmbeddingMemoSize.
func NewToolEmbeddingMemo(maxSize int) *ToolEmbeddingMemo {
	if maxSize <= 0 {
		maxSize = DefaultToolEmbeddingMemoSize
	}
	return &ToolEmbeddingMemo{
		entries: make(map[string][]float32, maxSize),
		order:   make([]string, 0, maxSize),
		maxSize: maxSize,
	}
}

// Get returns the memoized embedding for key.
func (m *ToolEmbeddingMemo) Get(key string) ([]float32, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	v, ok := m.entries[key]
	return v, ok
}

// Put stores value under key, evicting the oldest entry when the memo is full.
// Storing a key that is already present is a no-op, so a racing duplicate
// compute never disturbs the FIFO order or replaces an equivalent vector.
func (m *ToolEmbeddingMemo) Put(key string, value []float32) {
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
