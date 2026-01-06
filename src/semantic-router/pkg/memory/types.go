package memory

import "time"

// MemoryType represents the category of a memory
type MemoryType string

const (
	// MemoryTypeSemantic represents facts, preferences, knowledge
	// Example: "User's budget for Hawaii is $10,000"
	MemoryTypeSemantic MemoryType = "semantic"

	// MemoryTypeProcedural represents instructions, how-to, steps
	// Example: "To deploy payment-service: run npm build, then docker push"
	MemoryTypeProcedural MemoryType = "procedural"

	// MemoryTypeEpisodic represents session summaries, past events
	// Example: "On Dec 29 2024, user planned Hawaii vacation with $10K budget"
	MemoryTypeEpisodic MemoryType = "episodic"

	// MemoryTypeConversation represents conversation memory
	MemoryTypeConversation MemoryType = "conversation"

	// MemoryTypeFact represents fact memory
	MemoryTypeFact MemoryType = "fact"

	// MemoryTypeContext represents context memory
	MemoryTypeContext MemoryType = "context"

	// MemoryTypeUser represents user-specific memory
	MemoryTypeUser MemoryType = "user"
)

// Memory represents a stored memory unit in the agentic memory system
type Memory struct {
	// ID is the unique identifier for this memory
	ID string `json:"id"`

	// Type is the category of this memory (semantic, procedural, episodic)
	Type MemoryType `json:"type"`

	// Content is the actual memory text
	// Should be self-contained with context (e.g., "budget for Hawaii is $10K" not just "$10K")
	Content string `json:"content"`

	// Embedding is the vector representation (not serialized to JSON)
	Embedding []float32 `json:"-"`

	// UserID is the owner of this memory (for user isolation)
	UserID string `json:"user_id"`

	// ProjectID is an optional project scope
	ProjectID string `json:"project_id,omitempty"`

	// Source indicates where this memory came from
	Source string `json:"source,omitempty"`

	// CreatedAt is when the memory was first stored
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt is when the memory was last modified
	UpdatedAt time.Time `json:"updated_at,omitempty"`

	// AccessCount tracks how often this memory is retrieved
	AccessCount int `json:"access_count"`

	// Importance is a score for prioritizing memories (0.0 to 1.0)
	Importance float32 `json:"importance"`
}

// RetrieveOptions configures memory retrieval
//
//	Query is the search query (will be embedded for vector search)
//	UserID filters memories to this user only
//	ProjectID optionally filters to a specific project
//	Types optionally filters to specific memory types
//	Limit is the maximum number of results to return (default: 5)
//	Threshold is the minimum similarity score (default: 0.6)
type RetrieveOptions struct {
	// Query is the search query text
	Query string

	// UserID is the user identifier for filtering
	UserID string

	// ProjectID optionally filters to a specific project
	ProjectID string

	// Types is an optional filter for memory types
	Types []MemoryType

	// Limit is the maximum number of results to return (default: 5)
	Limit int

	// Threshold is the minimum similarity score (default: 0.6)
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

// MemoryScope defines the scope for bulk operations (e.g., ForgetByScope)
type MemoryScope struct {
	// UserID is required - all operations are user-scoped
	UserID string

	// ProjectID optionally narrows scope to a project
	ProjectID string

	// Types optionally narrows scope to specific memory types
	Types []MemoryType
}
