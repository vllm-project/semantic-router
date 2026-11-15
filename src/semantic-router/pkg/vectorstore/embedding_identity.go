package vectorstore

import (
	"context"
	"errors"
	"fmt"
)

// EmbeddingIdentityMetadataKey is owned by the router, not client metadata.
const EmbeddingIdentityMetadataKey = "_router_embedding_identity"

var ErrEmbeddingIncompatible = errors.New("vector store embeddings are incompatible with the active model; create a new vector store and reattach the original files")

// ErrReservedMetadataKey reports client metadata that sets a key the router owns.
var ErrReservedMetadataKey = errors.New("metadata key is reserved for the router")

// ManagerOption configures immutable properties of a manager at construction.
type ManagerOption func(*Manager)

// WithEmbeddingIdentity binds newly created stores to the actual loaded
// representation. An empty identity preserves untagged legacy stores only.
func WithEmbeddingIdentity(identity string) ManagerOption {
	return func(m *Manager) { m.embeddingIdentity = identity }
}

// EmbeddingIdentity identifies the active representation for dependent caches.
func (m *Manager) EmbeddingIdentity() string { return m.embeddingIdentity }

func validateClientMetadata(metadata map[string]interface{}) error {
	if _, exists := metadata[EmbeddingIdentityMetadataKey]; exists {
		return fmt.Errorf("%w: %q", ErrReservedMetadataKey, EmbeddingIdentityMetadataKey)
	}
	return nil
}

func (m *Manager) validateEmbeddingBackend() error {
	if _, external := m.backend.(*LlamaStackBackend); external && m.embeddingIdentity != "" {
		return fmt.Errorf("%w: Llama Stack embeds queries remotely and cannot verify compatibility with local document embeddings", ErrEmbeddingIncompatible)
	}
	return nil
}

// CheckEmbeddingCompatibility leaves historical stores visible for inventory
// and file recovery while refusing to compare or append incompatible vectors.
func (m *Manager) CheckEmbeddingCompatibility(id string) error {
	vs, err := m.GetStore(id)
	if err != nil {
		return err
	}
	if err := m.validateEmbeddingBackend(); err != nil {
		return err
	}
	value, exists := vs.Metadata[EmbeddingIdentityMetadataKey]
	if !exists && m.embeddingIdentity == "" {
		return nil
	}
	stored, ok := value.(string)
	if !ok || stored == "" || stored != m.embeddingIdentity {
		return ErrEmbeddingIncompatible
	}
	return nil
}

// Search and InsertChunks enforce the same representation contract for API,
// RAG and background ingestion callers.
func (m *Manager) Search(ctx context.Context, id string, query []float32, topK int, threshold float32, filter map[string]interface{}) ([]SearchResult, error) {
	if err := m.CheckEmbeddingCompatibility(id); err != nil {
		return nil, err
	}
	return m.backend.Search(ctx, id, query, topK, threshold, filter)
}

func (m *Manager) InsertChunks(ctx context.Context, id string, chunks []EmbeddedChunk) error {
	if err := m.CheckEmbeddingCompatibility(id); err != nil {
		return err
	}
	return m.backend.InsertChunks(ctx, id, chunks)
}

func (m *Manager) HybridSearch(ctx context.Context, id, query string, vector []float32, topK int, threshold float32, filter map[string]interface{}, cfg *HybridSearchConfig) ([]SearchResult, error) {
	if err := m.CheckEmbeddingCompatibility(id); err != nil {
		return nil, err
	}
	if searcher, ok := m.backend.(HybridSearcher); ok {
		return searcher.HybridSearch(ctx, id, query, vector, topK, threshold, filter, cfg)
	}
	return GenericHybridRerank(ctx, m.backend, id, query, vector, topK, threshold, filter, cfg)
}

// LexicalSearch searches a store's chunks by BM25 alone. It compares no
// vectors, so it reads a store of any embedding representation.
func (m *Manager) LexicalSearch(ctx context.Context, id, query string, topK int, threshold float32, filter map[string]interface{}) ([]SearchResult, error) {
	if _, err := m.GetStore(id); err != nil {
		return nil, err
	}
	searcher, ok := m.backend.(LexicalSearcher)
	if !ok {
		return nil, fmt.Errorf("the %T vector store backend keeps no lexical index", m.backend)
	}
	return searcher.LexicalSearch(ctx, id, query, topK, threshold, filter)
}

// ListChunks lists a store's chunks, the most recently inserted first, and
// how many it holds; like LexicalSearch it reads any representation.
func (m *Manager) ListChunks(ctx context.Context, id string, limit int, filter map[string]interface{}) ([]SearchResult, int, error) {
	if _, err := m.GetStore(id); err != nil {
		return nil, 0, err
	}
	lister, ok := m.backend.(ChunkLister)
	if !ok {
		return nil, 0, fmt.Errorf("the %T vector store backend does not list its chunks", m.backend)
	}
	return lister.ListChunks(ctx, id, limit, filter)
}
