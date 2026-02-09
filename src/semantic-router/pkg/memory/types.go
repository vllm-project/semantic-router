package memory

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// MemoryType represents the category of a memory in the agentic memory system.
type MemoryType string

const (
	// MemoryTypeSemantic represents facts, preferences, knowledge.
	// Example: "User's budget for Hawaii is $10,000"
	MemoryTypeSemantic MemoryType = "semantic"

	// MemoryTypeProcedural represents instructions, how-to, steps.
	// Example: "To deploy payment-service: run npm build, then docker push"
	MemoryTypeProcedural MemoryType = "procedural"

	// MemoryTypeEpisodic represents session summaries, past events.
	// Example: "On Dec 29 2024, user planned Hawaii vacation with $10K budget"
	MemoryTypeEpisodic MemoryType = "episodic"
)

// ExtractedFact represents a fact extracted by the LLM from conversation.
// This is the output of ExtractFacts().
type ExtractedFact struct {
	// Type is the category (semantic, procedural, episodic)
	Type MemoryType `json:"type"`

	// Content is the extracted fact with context.
	// Should be self-contained: "budget for Hawaii is $10K" not just "$10K"
	Content string `json:"content"`
}

// Message represents a conversation message used for fact extraction.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

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

// RetrieveResult represents a memory retrieved from search with its relevance score
type RetrieveResult struct {
	// Memory is the retrieved memory
	Memory *Memory `json:"memory"`

	// Score is the similarity score (0.0 to 1.0, higher = more relevant)
	Score float32 `json:"score"`
}

// MemoryHit represents a single memory retrieval result in flat structure
//
// ID is the unique identifier of the memory entry
// Content is the content of the memory entry
// Type is the type of memory
// Similarity is the similarity score (0.0 to 1.0)
// Metadata contains additional metadata
type MemoryHit struct {
	ID         string
	Content    string
	Type       MemoryType
	Similarity float32
	Metadata   map[string]interface{}
}

// RetrieveOptions configures memory retrieval
type RetrieveOptions struct {
	// Query is the search query (will be embedded for vector search)
	Query string

	// UserID filters memories to this user only
	UserID string

	// ProjectID optionally filters to a specific project
	ProjectID string

	// Types optionally filters to specific memory types
	Types []MemoryType

	// Limit is the maximum number of results to return (default: 5)
	Limit int

	// Threshold is the minimum similarity score (range 0.0 to 1.0, default: 0.70)
	Threshold float32
}

// DefaultMemoryConfig returns a default memory configuration.
// EmbeddingModel is intentionally omitted - let router auto-detect from embedding_models config.
func DefaultMemoryConfig() config.MemoryConfig {
	return config.MemoryConfig{
		Milvus: config.MemoryMilvusConfig{
			Dimension: 384, // Safe default, will be overridden by router
		},
		DefaultRetrievalLimit:      5,
		DefaultSimilarityThreshold: 0.70,
	}
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
