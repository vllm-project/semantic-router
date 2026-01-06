package memory

// MemoryType represents the type of memory entry
type MemoryType string

// MemoryTypeConversation represents conversation memory
// MemoryTypeFact represents fact memory
// MemoryTypeContext represents context memory
// MemoryTypeUser represents user-specific memory
const (
	MemoryTypeConversation MemoryType = "conversation"
	MemoryTypeFact         MemoryType = "fact"
	MemoryTypeContext      MemoryType = "context"
	MemoryTypeUser         MemoryType = "user"
)

// RetrieveOptions contains options for memory retrieval
//
//	Query is the search query text
//	UserID is the user identifier for filtering
//	Types is an optional filter for memory types
//	Limit is the maximum number of results to return (default: 5)
//	Threshold is the minimum similarity score (default: 0.6)
type RetrieveOptions struct {
	Query     string
	UserID    string
	Types     []MemoryType
	Limit     int
	Threshold float32
}

// MemoryConfig contains configuration for memory operations
//
//	Embedding contains embedding model configuration
//	DefaultRetrievalLimit is the default limit for retrieval (default: 5)
//	DefaultSimilarityThreshold is the default similarity threshold (default: 0.6)
type MemoryConfig struct {
	Embedding                  EmbeddingConfig `yaml:"embedding"`
	DefaultRetrievalLimit      int             `yaml:"default_retrieval_limit"`
	DefaultSimilarityThreshold float32         `yaml:"default_similarity_threshold"`
}

// EmbeddingConfig contains configuration for embedding generation
//
//	Model is the embedding model name (default: "all-MiniLM-L6-v2")
//	Dimension is the embedding dimension (default: 384 for all-MiniLM-L6-v2)
type EmbeddingConfig struct {
	Model     string `yaml:"model"`
	Dimension int    `yaml:"dimension"`
}

// DefaultMemoryConfig returns a default memory configuration
func DefaultMemoryConfig() MemoryConfig {
	return MemoryConfig{
		Embedding: EmbeddingConfig{
			Model:     "all-MiniLM-L6-v2",
			Dimension: 384,
		},
		DefaultRetrievalLimit:      5,
		DefaultSimilarityThreshold: 0.6,
	}
}

// RetrieveResult represents a single memory retrieval result
//
//	ID is the unique identifier of the memory entry
//	Content is the content of the memory entry
//	Type is the type of memory
//	Similarity is the similarity score (0.0 to 1.0)
//	Metadata contains additional metadata
type RetrieveResult struct {
	ID         string
	Content    string
	Type       MemoryType
	Similarity float32
	Metadata   map[string]interface{}
}
