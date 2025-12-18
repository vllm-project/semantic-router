package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"

	"github.com/openai/openai-go"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
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

// ToolsDatabase manages a collection of tools with semantic search capabilities
type ToolsDatabase struct {
	entries             []ToolEntry
	mu                  sync.RWMutex
	similarityThreshold float32
	enabled             bool
}

// ToolsDatabaseOptions holds options for creating a new tools database
type ToolsDatabaseOptions struct {
	SimilarityThreshold float32
	Enabled             bool
}

// NewToolsDatabase creates a new tools database with the given options
func NewToolsDatabase(options ToolsDatabaseOptions) *ToolsDatabase {
	return &ToolsDatabase{
		entries:             []ToolEntry{},
		similarityThreshold: options.SimilarityThreshold,
		enabled:             options.Enabled,
	}
}

// IsEnabled returns whether the tools database is enabled
func (db *ToolsDatabase) IsEnabled() bool {
	return db.enabled
}

// LoadToolsFromFile loads tools from a JSON file
func (db *ToolsDatabase) LoadToolsFromFile(filePath string) error {
	if !db.enabled {
		return nil
	}

	// Read the JSON file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read tools file: %w", err)
	}

	// Parse the JSON data into ToolEntry slice
	var toolEntries []ToolEntry
	if err := json.Unmarshal(data, &toolEntries); err != nil {
		return fmt.Errorf("failed to parse tools JSON: %w", err)
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// Generate embeddings for each tool and add to database
	for _, entry := range toolEntries {
		// Generate embedding for the description using Qwen3/Gemma with automatic routing
		// qualityPriority=0.5, latencyPriority=0.5 for balanced performance
		output, err := candle_binding.GetEmbeddingWithMetadata(entry.Description, 0.5, 0.5, 0)
		if err != nil {
			logging.Warnf("Failed to generate embedding for tool %s: %v", entry.Tool.Function.Name, err)
			continue
		}

		// Set the embedding
		entry.Embedding = output.Embedding

		// Add to the database
		db.entries = append(db.entries, entry)
		logging.Debugf("Loaded tool: %s - %s (model: %s)", entry.Tool.Function.Name, entry.Description, output.ModelType)
	}

	logging.Infof("Loaded %d tools from file: %s", len(toolEntries), filePath)
	return nil
}

// AddTool adds a tool to the database with automatic embedding generation
func (db *ToolsDatabase) AddTool(tool openai.ChatCompletionToolParam, description string, category string, tags []string) error {
	if !db.enabled {
		return nil
	}

	// Generate embedding for the description using Qwen3/Gemma with automatic routing
	// qualityPriority=0.5, latencyPriority=0.5 for balanced performance
	output, err := candle_binding.GetEmbeddingWithMetadata(description, 0.5, 0.5, 0)
	if err != nil {
		return fmt.Errorf("failed to generate embedding for tool %s: %w", tool.Function.Name, err)
	}

	entry := ToolEntry{Tool: tool, Description: description, Embedding: output.Embedding, Category: category, Tags: tags}

	db.mu.Lock()
	defer db.mu.Unlock()

	db.entries = append(db.entries, entry)
	logging.Infof("Added tool: %s (%s) using model: %s", tool.Function.Name, description, output.ModelType)

	return nil
}

// FindSimilarTools finds the most similar tools based on the query
func (db *ToolsDatabase) FindSimilarTools(query string, topK int) ([]openai.ChatCompletionToolParam, error) {
	if !db.enabled {
		return []openai.ChatCompletionToolParam{}, nil
	}

	// Generate embedding for the query using Qwen3/Gemma with automatic routing
	// qualityPriority=0.5, latencyPriority=0.5 for balanced performance
	output, err := candle_binding.GetEmbeddingWithMetadata(query, 0.5, 0.5, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}
	queryEmbedding := output.Embedding

	db.mu.RLock()
	defer db.mu.RUnlock()

	type SimilarityResult struct {
		Entry      ToolEntry
		Similarity float32
	}

	// Calculate similarities
	results := make([]SimilarityResult, 0, len(db.entries))
	for _, entry := range db.entries {
		// Calculate similarity
		var dotProduct float32
		for i := 0; i < len(queryEmbedding) && i < len(entry.Embedding); i++ {
			dotProduct += queryEmbedding[i] * entry.Embedding[i]
		}

		// Debug logging to see similarity scores
		logging.Debugf("Tool '%s' similarity score: %.4f (threshold: %.4f)",
			entry.Tool.Function.Name, dotProduct, db.similarityThreshold)

		// Only consider if above threshold
		if dotProduct >= db.similarityThreshold {
			results = append(results, SimilarityResult{
				Entry:      entry,
				Similarity: dotProduct,
			})
		}
	}

	// No results found
	if len(results) == 0 {
		return []openai.ChatCompletionToolParam{}, nil
	}

	// Sort by similarity (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Select top-k tools that meet the threshold
	limit := min(topK, len(results))
	selectedTools := make([]openai.ChatCompletionToolParam, 0, limit)
	for i := range limit {
		selectedTools = append(selectedTools, results[i].Entry.Tool)
		logging.Infof("Selected tool: %s (similarity=%.4f)",
			results[i].Entry.Tool.Function.Name, results[i].Similarity)
	}

	logging.Infof("Found %d similar tools for query: %s", len(selectedTools), query)
	return selectedTools, nil
}

// GetAllTools returns all tools in the database
func (db *ToolsDatabase) GetAllTools() []openai.ChatCompletionToolParam {
	if !db.enabled {
		return []openai.ChatCompletionToolParam{}
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	tools := make([]openai.ChatCompletionToolParam, len(db.entries))
	for i, entry := range db.entries {
		tools[i] = entry.Tool
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
