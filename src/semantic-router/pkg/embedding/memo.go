package embedding

import "sync"

// Memo is the repo's shared bounded, concurrency-safe cache of
// embedding-key -> embedding vector: a plain data structure with FIFO eviction
// and nothing else. Callers use it to reuse embeddings across requests instead
// of re-embedding the same text every time.
//
// Only the data structure is shared. Every semantic choice belongs to the
// caller: how a key is derived (raw text, a model-namespaced prefix, a sha256
// digest), whether vectors are copied on store or read, which inputs are worth
// memoizing at all, and how misses are batched. Memo takes no position on any
// of it.
//
// There is deliberately no TTL, because an embedding is a deterministic pure
// function of (text, model): a memoized vector can never go stale - it can
// only become unreachable, at which point FIFO eviction retires it. That
// argument holds only while the key uniquely identifies (text, model) for the
// memo's lifetime, which callers establish one of two ways: fold the model
// identity into the key (the tool embedder, the compression scorer), or bind
// the memo to a single fixed model and key by raw text (the in-memory
// semantic cache, whose model never changes after construction).
//
// Concurrent misses on the same key may each compute the embedding and both
// call Put; that is intentional. Computation happens outside the memo lock,
// because holding a lock across model inference or a remote embedding call
// would serialize every request behind the slowest one. The worst case under a
// race is redundant work, never a wrong vector.
//
// Stored slices are shared with callers by default; a caller that mutates
// vectors must copy on store and on read itself. Callers that only read stored
// vectors (dot products) can share them freely.
type Memo struct {
	mu      sync.Mutex
	entries map[string][]float32
	order   []string // insertion order, for FIFO eviction
	maxSize int
}

// NewMemo returns a memo bounded to maxSize entries. maxSize must be positive;
// there is no default because the right bound is a caller workload decision
// (in-repo callers use 512, 1024, and 2048). A non-positive value is clamped
// to 1 so a misconfigured caller degrades to thrashing, not a hidden budget.
func NewMemo(maxSize int) *Memo {
	if maxSize <= 0 {
		maxSize = 1
	}
	return &Memo{
		entries: make(map[string][]float32, maxSize),
		order:   make([]string, 0, maxSize),
		maxSize: maxSize,
	}
}

// Get returns the memoized embedding for key.
func (m *Memo) Get(key string) ([]float32, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	v, ok := m.entries[key]
	return v, ok
}

// Put stores value under key, evicting the oldest entry when the memo is full.
// Storing a key that is already present is a no-op, so a racing duplicate
// compute never disturbs the FIFO order or replaces an equivalent vector.
func (m *Memo) Put(key string, value []float32) {
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

// GetOrCompute returns the memoized vector for key, computing and storing it on
// a miss. compute receives no arguments: the caller closes over whatever text
// it embeds, so a derived key (hash, prefix) is never mistaken for the text.
//
// compute runs WITHOUT the memo lock held (holding it across model inference
// would serialize every caller); a racing miss may compute redundantly but
// never stores a wrong vector. Errors are not memoized, so a transient failure
// is retried on the next call.
func (m *Memo) GetOrCompute(key string, compute func() ([]float32, error)) ([]float32, error) {
	if v, ok := m.Get(key); ok {
		return v, nil
	}
	v, err := compute()
	if err != nil {
		return nil, err
	}
	m.Put(key, v)
	return v, nil
}
