package memory

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// InMemoryStore is an in-memory implementation of Store for testing and POC.
// It stores memories in memory with embedding-based similarity search.
type InMemoryStore struct {
	mu              sync.RWMutex
	memories        map[string]*Memory // ID -> Memory
	enabled         bool
	embeddingConfig EmbeddingConfig
}

// NewInMemoryStore creates a new in-memory memory store with default embedding config.
func NewInMemoryStore() *InMemoryStore {
	return NewInMemoryStoreWithConfig(EmbeddingConfig{Model: EmbeddingModelBERT})
}

// NewInMemoryStoreWithConfig creates a new in-memory memory store with custom embedding config.
func NewInMemoryStoreWithConfig(embeddingConfig EmbeddingConfig) *InMemoryStore {
	return &InMemoryStore{
		memories:        make(map[string]*Memory),
		enabled:         true,
		embeddingConfig: embeddingConfig,
	}
}

// IsEnabled returns whether the store is enabled.
func (s *InMemoryStore) IsEnabled() bool {
	return s.enabled
}

// Store saves a new memory.
func (s *InMemoryStore) Store(ctx context.Context, memory *Memory) error {
	if !s.enabled {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Generate embedding if not already set
	if len(memory.Embedding) == 0 {
		embedding, err := GenerateEmbedding(memory.Content, s.embeddingConfig)
		if err != nil {
			return err
		}
		memory.Embedding = embedding
	}

	// Check if ID already exists
	if _, exists := s.memories[memory.ID]; exists {
		return fmt.Errorf("memory with ID %s already exists", memory.ID)
	}

	// Set CreatedAt if not set
	if memory.CreatedAt.IsZero() {
		memory.CreatedAt = time.Now()
	}

	s.memories[memory.ID] = memory
	logging.Debugf("InMemoryStore: stored memory id=%s, type=%s, user=%s", memory.ID, memory.Type, memory.UserID)
	return nil
}

// Retrieve performs semantic search for relevant memories.
func (s *InMemoryStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	if !s.enabled {
		return nil, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Generate embedding for query using unified embedding configuration
	queryEmbedding, err := GenerateEmbedding(opts.Query, s.embeddingConfig)
	if err != nil {
		return nil, err
	}

	var results []*RetrieveResult

	// Search through all memories
	for _, mem := range s.memories {
		// Filter by user ID
		if opts.UserID != "" && mem.UserID != opts.UserID {
			continue
		}

		// Filter by project ID if specified
		if opts.ProjectID != "" && mem.ProjectID != opts.ProjectID {
			continue
		}

		// Filter by types if specified
		if len(opts.Types) > 0 {
			found := false
			for _, t := range opts.Types {
				if mem.Type == t {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		// Calculate similarity
		similarity := cosineSimilarity(queryEmbedding, mem.Embedding)
		if similarity < opts.Threshold {
			continue
		}

		results = append(results, &RetrieveResult{
			Memory: mem,
			Score:  similarity,
		})
	}

	// Sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Apply limit
	if opts.Limit > 0 && len(results) > opts.Limit {
		results = results[:opts.Limit]
	}

	return results, nil
}

// Get retrieves a memory by ID.
func (s *InMemoryStore) Get(ctx context.Context, id string) (*Memory, error) {
	if !s.enabled {
		return nil, fmt.Errorf("store not enabled")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	mem, exists := s.memories[id]
	if !exists {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	return mem, nil
}

// Update modifies an existing memory.
func (s *InMemoryStore) Update(ctx context.Context, id string, memory *Memory) error {
	if !s.enabled {
		return fmt.Errorf("store not enabled")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if memory exists
	existing, exists := s.memories[id]
	if !exists {
		return fmt.Errorf("memory not found: %s", id)
	}

	// Update fields
	existing.Content = memory.Content
	existing.Type = memory.Type
	existing.UpdatedAt = time.Now()
	if memory.ProjectID != "" {
		existing.ProjectID = memory.ProjectID
	}
	if memory.Source != "" {
		existing.Source = memory.Source
	}

	// Regenerate embedding if content changed
	if existing.Content != memory.Content {
		embedding, err := GenerateEmbedding(memory.Content, s.embeddingConfig)
		if err != nil {
			return err
		}
		existing.Embedding = embedding
	}

	logging.Debugf("InMemoryStore: updated memory id=%s", id)
	return nil
}

// Forget deletes a memory by ID.
func (s *InMemoryStore) Forget(ctx context.Context, id string) error {
	if !s.enabled {
		return fmt.Errorf("store not enabled")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.memories[id]; !exists {
		return fmt.Errorf("memory not found: %s", id)
	}

	delete(s.memories, id)
	logging.Debugf("InMemoryStore: deleted memory id=%s", id)
	return nil
}

// ForgetByScope deletes all memories matching the scope.
func (s *InMemoryStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	if !s.enabled {
		return fmt.Errorf("store not enabled")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var toDelete []string
	for id, mem := range s.memories {
		// Must match user ID
		if mem.UserID != scope.UserID {
			continue
		}

		// Optionally match project ID
		if scope.ProjectID != "" && mem.ProjectID != scope.ProjectID {
			continue
		}

		// Optionally match types
		if len(scope.Types) > 0 {
			found := false
			for _, t := range scope.Types {
				if mem.Type == t {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		toDelete = append(toDelete, id)
	}

	for _, id := range toDelete {
		delete(s.memories, id)
	}

	logging.Debugf("InMemoryStore: deleted %d memories by scope", len(toDelete))
	return nil
}

// CheckConnection verifies the store connection is healthy.
// For in-memory store, this is always healthy (no external connection).
func (s *InMemoryStore) CheckConnection(ctx context.Context) error {
	if !s.enabled {
		return fmt.Errorf("store not enabled")
	}
	// In-memory store has no external connection to check
	return nil
}

// Close releases resources.
func (s *InMemoryStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.memories = nil
	s.enabled = false
	return nil
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct float32
	var normA, normB float32

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
