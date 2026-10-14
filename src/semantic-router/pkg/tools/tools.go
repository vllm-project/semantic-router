package tools

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding/vecmath"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ToolEntry represents a tool stored in the tools database
type ToolEntry struct {
	Tool        openai.ChatCompletionToolParam `json:"tool"`
	Description string                         `json:"description"` // Used for similarity matching
	Embedding   []float32                      `json:"-"`           // Generated from description
	Tags        []string                       `json:"tags,omitempty"`
	Category    string                         `json:"category,omitempty"`
}

// ToolSimilarity represents a tool candidate with its similarity score.
type ToolSimilarity struct {
	Entry      ToolEntry
	Similarity float32
}

// ToolsDatabase manages a collection of tools with semantic search capabilities
type ToolsDatabase struct {
	entries             []ToolEntry
	mu                  sync.RWMutex
	similarityThreshold float32
	enabled             bool
	modelType           string // Model type to use for embeddings (e.g., "mmbert", "qwen3")
	targetDim           int    // Target dimension for embeddings
	provider            embedding.Provider
}

// ToolsDatabaseOptions holds options for creating a new tools database
type ToolsDatabaseOptions struct {
	SimilarityThreshold float32
	Enabled             bool
	ModelType           string // Model type to use for embeddings
	TargetDimension     int    // Target dimension for embeddings
	Provider            embedding.Provider
}

// NewToolsDatabase creates a new tools database with the given options
func NewToolsDatabase(options ToolsDatabaseOptions) *ToolsDatabase {
	return &ToolsDatabase{
		entries:             []ToolEntry{},
		similarityThreshold: options.SimilarityThreshold,
		enabled:             options.Enabled,
		modelType:           options.ModelType,
		targetDim:           options.TargetDimension,
		provider:            options.Provider,
	}
}

// IsEnabled returns whether the tools database is enabled
func (db *ToolsDatabase) IsEnabled() bool {
	return db.enabled
}

// AddTool admits an isolated definition under its normalized unique name.
// Validation and embedding failures leave the catalog unchanged.
func (db *ToolsDatabase) AddTool(tool openai.ChatCompletionToolParam, description string, category string, tags []string) error {
	if !db.enabled {
		return nil
	}
	entry, err := prepareToolEntry(ToolEntry{Tool: tool, Description: description, Category: category, Tags: tags})
	if err != nil {
		return err
	}
	entry.Embedding, err = db.embedText(description)
	if err != nil {
		return fmt.Errorf("failed to generate tool embedding: %w", err)
	}
	if err := validateEmbedding(entry.Embedding); err != nil {
		return err
	}
	if err := db.appendEntries([]ToolEntry{entry}); err != nil {
		return err
	}
	logging.ComponentEvent("tools", "tool_added", map[string]interface{}{
		"tool_name": entry.Tool.Function.Name, "category": category,
		"has_tags": len(tags) > 0, "description_len": len(description), "model_type": db.modelType,
	})
	return nil
}

func (db *ToolsDatabase) similarityFloor(minOverride *float32) float32 {
	if minOverride != nil {
		return *minOverride
	}
	return db.similarityThreshold
}

// FindSimilarTools finds the most similar tools based on the query
func (db *ToolsDatabase) FindSimilarTools(query string, topK int) ([]openai.ChatCompletionToolParam, error) {
	results, err := db.FindSimilarToolsWithScores(query, topK)
	if err != nil {
		return nil, err
	}

	selectedTools := make([]openai.ChatCompletionToolParam, len(results))
	for i, result := range results {
		selectedTools[i] = result.Entry.Tool
	}

	return selectedTools, nil
}

// FindSimilarToolsWithScores finds the most similar tools based on the query and returns scores.
func (db *ToolsDatabase) FindSimilarToolsWithScores(query string, topK int) ([]ToolSimilarity, error) {
	return db.FindSimilarToolsWithScoresMinSimilarity(query, topK, nil)
}

// FindSimilarToolsWithScoresMinSimilarity is like FindSimilarToolsWithScores but allows overriding
// the minimum similarity cutoff for this query (embedding dot-product scores).
func (db *ToolsDatabase) FindSimilarToolsWithScoresMinSimilarity(query string, topK int, minSimilarity *float32) ([]ToolSimilarity, error) {
	if !db.enabled {
		return []ToolSimilarity{}, nil
	}

	queryEmbedding, err := db.embedText(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}

	if err := validateEmbedding(queryEmbedding); err != nil {
		return nil, err
	}
	floor := db.similarityFloor(minSimilarity)
	if !finiteScore(floor) {
		return nil, fmt.Errorf("tools: similarity threshold must be finite")
	}
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Calculate similarities
	results := make([]ToolSimilarity, 0, len(db.entries))
	for _, entry := range db.entries {
		n := min(len(queryEmbedding), len(entry.Embedding))
		dotProduct := vecmath.Dot(queryEmbedding[:n], entry.Embedding[:n])

		if !finiteScore(dotProduct) {
			return nil, fmt.Errorf("tools: similarity score must be finite")
		}
		logging.Debugf("Tool '%s' similarity score: %.4f (threshold: %.4f)",
			entry.Tool.Function.Name, dotProduct, floor)

		if dotProduct >= floor {
			results = append(results, ToolSimilarity{
				Entry:      entry,
				Similarity: dotProduct,
			})
		}
	}

	// No results found
	if len(results) == 0 {
		return []ToolSimilarity{}, nil
	}

	// Name is the final tie-breaker before truncation, independent of load order.
	sort.Slice(results, func(i, j int) bool {
		if results[i].Similarity == results[j].Similarity {
			return results[i].Entry.Tool.Function.Name < results[j].Entry.Tool.Function.Name
		}
		return results[i].Similarity > results[j].Similarity
	})

	limit := topK
	if limit <= 0 || limit > len(results) {
		limit = len(results)
	}

	selected := make([]ToolSimilarity, limit)
	for i, result := range results[:limit] {
		selected[i] = ToolSimilarity{Entry: cloneToolEntry(result.Entry), Similarity: result.Similarity}
		logging.Infof("Selected tool: %s (similarity=%.4f)",
			result.Entry.Tool.Function.Name, result.Similarity)
	}

	logging.Infof("Found %d similar tools for query: %s", len(selected), logging.ContentDescriptor(query))
	return selected, nil
}

func (db *ToolsDatabase) embedText(text string) ([]float32, error) {
	if db.provider != nil {
		return db.provider.Embed(context.Background(), text)
	}
	return nil, fmt.Errorf("tools embedding provider was not prepared")
}

// GetAllTools returns isolated definitions in normalized-name order.
func (db *ToolsDatabase) GetAllTools() []openai.ChatCompletionToolParam {
	if !db.enabled {
		return []openai.ChatCompletionToolParam{}
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	tools := make([]openai.ChatCompletionToolParam, len(db.entries))
	for i, entry := range db.entries {
		tools[i] = cloneToolDefinition(entry.Tool)
	}

	return tools
}

// GetToolCount returns the number of tools in the database
func (db *ToolsDatabase) GetToolCount() int {
	if !db.enabled {
		return 0
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	return len(db.entries)
}
