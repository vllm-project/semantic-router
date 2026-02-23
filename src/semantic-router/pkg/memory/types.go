package memory

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// MemoryType represents the category of a memory in the agentic memory system.
type MemoryType string

const (
	// MemoryTypeSemantic represents facts, preferences, knowledge.
	MemoryTypeSemantic MemoryType = "semantic"

	// MemoryTypeProcedural represents instructions, how-to, steps.
	MemoryTypeProcedural MemoryType = "procedural"

	// MemoryTypeEpisodic represents session summaries, past events.
	MemoryTypeEpisodic MemoryType = "episodic"
)

// MemoryVisibility controls who can access a memory within a group.
type MemoryVisibility string

const (
	// VisibilityUser restricts access to the owning user only (default).
	VisibilityUser MemoryVisibility = "user"

	// VisibilityGroup allows all members of the same group to retrieve this memory.
	VisibilityGroup MemoryVisibility = "group"

	// VisibilityPublic allows any user to retrieve this memory.
	VisibilityPublic MemoryVisibility = "public"
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

	// Content is the actual memory text (L2 full detail)
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

	// AccessCount tracks how often this memory is retrieved (used for retention score S = S0 + AccessCount)
	AccessCount int `json:"access_count"`

	// LastAccessed is when the memory was last retrieved (used for retention score t and reinforcement)
	LastAccessed time.Time `json:"last_accessed,omitempty"`

	// Importance is a score for prioritizing memories (0.0 to 1.0)
	Importance float32 `json:"importance"`

	// --- Hierarchical organization ---

	// GroupID associates this memory with a group for shared access.
	GroupID string `json:"group_id,omitempty"`

	// ParentID links this memory to a parent category node, forming a tree.
	ParentID string `json:"parent_id,omitempty"`

	// IsCategory marks this memory as a category (non-leaf) node.
	IsCategory bool `json:"is_category,omitempty"`

	// Abstract is a short summary used for fast candidate scoring (L0).
	// Length is controlled by CategorizerConfig.AbstractMaxLen.
	Abstract string `json:"abstract,omitempty"`

	// AbstractEmbedding is the vector for the Abstract text (not serialized to JSON).
	AbstractEmbedding []float32 `json:"-"`

	// Overview is a longer summary for reranking and navigation (L1).
	// Length is controlled by CategorizerConfig.OverviewMaxLen.
	// Content serves as the full detail (L2).
	Overview string `json:"overview,omitempty"`

	// Visibility controls access scope: "user" (default), "group", or "public".
	Visibility MemoryVisibility `json:"visibility,omitempty"`

	// RelatedIDs holds IDs of related memories for cross-memory linking.
	RelatedIDs []string `json:"related_ids,omitempty"`
}

// RetrieveResult represents a memory retrieved from search with its relevance score
type RetrieveResult struct {
	// Memory is the retrieved memory
	Memory *Memory `json:"memory"`

	// Score is the similarity score (0.0 to 1.0, higher = more relevant)
	Score float32 `json:"score"`

	// Related holds memories linked via cross-memory relations (populated when relations are enabled).
	Related []*RelatedMemory `json:"related,omitempty"`
}

// RelatedMemory is a lightweight reference to a linked memory surfaced during retrieval.
type RelatedMemory struct {
	ID       string  `json:"id"`
	Abstract string  `json:"abstract,omitempty"`
	Reason   string  `json:"reason,omitempty"`
	Score    float32 `json:"score,omitempty"`
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

// MemoryHybridConfig configures hybrid scoring (BM25 + n-gram + vector fusion)
// for memory retrieval. When non-nil on HierarchicalRetrieveOptions, every scoring
// step in the hierarchical pipeline uses fused scores instead of pure cosine similarity.
type MemoryHybridConfig struct {
	// Mode selects the score fusion method: "weighted" (default) or "rrf".
	Mode string

	// VectorWeight is the relative weight for cosine similarity (default: 0.7).
	VectorWeight float64
	// BM25Weight is the relative weight for BM25 term matching (default: 0.2).
	BM25Weight float64
	// NgramWeight is the relative weight for character n-gram Jaccard (default: 0.1).
	NgramWeight float64

	// RRFConstant for RRF mode (default: 60).
	RRFConstant int

	// BM25K1 controls term frequency saturation (default: 1.2).
	BM25K1 float64
	// BM25B controls document length normalization (default: 0.75).
	BM25B float64

	// NgramSize is the character n-gram size for Jaccard similarity (default: 3).
	NgramSize int
}

// ApplyDefaults fills zero-valued fields with sensible defaults.
func (c *MemoryHybridConfig) ApplyDefaults() {
	if c.Mode == "" {
		c.Mode = "weighted"
	}
	if c.VectorWeight == 0 && c.BM25Weight == 0 && c.NgramWeight == 0 {
		c.VectorWeight = 0.7
		c.BM25Weight = 0.2
		c.NgramWeight = 0.1
	}
	if c.RRFConstant <= 0 {
		c.RRFConstant = 60
	}
	if c.BM25K1 == 0 {
		c.BM25K1 = 1.2
	}
	if c.BM25B == 0 {
		c.BM25B = 0.75
	}
	if c.NgramSize <= 0 {
		c.NgramSize = 3
	}
}

// NormalizedWeights returns vector, bm25, ngram weights that sum to 1.
func (c *MemoryHybridConfig) NormalizedWeights() (float64, float64, float64) {
	total := c.VectorWeight + c.BM25Weight + c.NgramWeight
	if total == 0 {
		return 1.0 / 3, 1.0 / 3, 1.0 / 3
	}
	return c.VectorWeight / total, c.BM25Weight / total, c.NgramWeight / total
}

// HierarchicalRetrieveOptions extends RetrieveOptions with group-aware, hierarchical search parameters.
type HierarchicalRetrieveOptions struct {
	RetrieveOptions

	// GroupIDs includes group-shared memories from these groups alongside the user's own.
	GroupIDs []string

	// IncludeGroupLevel enables searching group-visible memories (requires GroupIDs).
	IncludeGroupLevel bool

	// MaxDepth limits the tree traversal depth (default: 3).
	MaxDepth int

	// ScorePropAlpha controls score propagation: final = alpha*child + (1-alpha)*parent (default: 0.5).
	ScorePropAlpha float32

	// EnableRelations causes retrieval to populate Related on each RetrieveResult.
	EnableRelations bool

	// MaxRelationsPerHit caps the number of related memories per result (default: 5).
	MaxRelationsPerHit int

	// Hybrid enables fused BM25 + n-gram + vector scoring at every level of the
	// hierarchical search. When nil, pure cosine similarity is used (existing behavior).
	Hybrid *MemoryHybridConfig

	// FollowLinks enables graph expansion: after collecting initial results,
	// follow RelatedIDs on each hit to discover cross-category memories that
	// the tree drill-down alone would miss. Inspired by GraphRAG research
	// showing +15-23% recall improvement from cross-document traversal.
	FollowLinks bool

	// MaxLinkDepth controls how many hops to follow (1 = direct links only,
	// 2 = links-of-links, etc.). Default: 1.
	MaxLinkDepth int

	// LinkEmbeddingConfig specifies which embedding model to use when scoring
	// linked memories during graph expansion. Required when FollowLinks is true.
	// Falls back to EmbeddingModelBERT if not set.
	LinkEmbeddingConfig *EmbeddingConfig
}

const (
	DefaultHierarchicalMaxDepth         = 3
	DefaultScorePropAlpha       float32 = 0.5
	DefaultMaxRelationsPerHit           = 5
	DefaultHierarchicalLimit            = 5
	DefaultMaxLinkDepth                 = 1
)

// ApplyDefaults fills in zero-valued fields.
func (h *HierarchicalRetrieveOptions) ApplyDefaults() {
	if h.MaxDepth <= 0 {
		h.MaxDepth = DefaultHierarchicalMaxDepth
	}
	if h.ScorePropAlpha <= 0 {
		h.ScorePropAlpha = DefaultScorePropAlpha
	}
	if h.MaxRelationsPerHit <= 0 {
		h.MaxRelationsPerHit = DefaultMaxRelationsPerHit
	}
	if h.FollowLinks && h.MaxLinkDepth <= 0 {
		h.MaxLinkDepth = DefaultMaxLinkDepth
	}
}

// MemoryRelation represents a directional link between two memories.
type MemoryRelation struct {
	FromID    string    `json:"from_id"`
	ToID      string    `json:"to_id"`
	Reason    string    `json:"reason"`
	Strength  float32   `json:"strength"`
	CreatedAt time.Time `json:"created_at"`
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

// ListOptions configures memory listing (non-semantic, filter-based retrieval)
type ListOptions struct {
	// UserID filters memories to this user only (required)
	UserID string

	// Types optionally filters to specific memory types
	Types []MemoryType

	// Limit is the maximum number of results to return (default: 20, max: 100)
	Limit int
}

// ListResult contains the memories returned by a List operation
type ListResult struct {
	// Memories is the list of memories returned
	Memories []*Memory `json:"memories"`

	// Total is the total number of matching memories
	Total int `json:"total"`

	// Limit is the limit that was applied
	Limit int `json:"limit"`
}

// MemoryScope defines the scope for bulk operations (e.g., ForgetByScope)
type MemoryScope struct {
	// UserID is required - all operations are user-scoped
	UserID string

	// ProjectID optionally narrows scope to a project
	ProjectID string

	// GroupID optionally narrows scope to a group
	GroupID string

	// Types optionally narrows scope to specific memory types
	Types []MemoryType
}
