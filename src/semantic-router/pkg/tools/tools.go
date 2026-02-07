package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
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
	modelType           string // Model type to use for embeddings (e.g., "mmbert", "qwen3", "gemma")
	targetDim           int    // Target dimension for embeddings
}

// ToolsDatabaseOptions holds options for creating a new tools database
type ToolsDatabaseOptions struct {
	SimilarityThreshold float32
	Enabled             bool
	ModelType           string // Model type to use for embeddings
	TargetDimension     int    // Target dimension for embeddings
}

// NewToolsDatabase creates a new tools database with the given options
func NewToolsDatabase(options ToolsDatabaseOptions) *ToolsDatabase {
	return &ToolsDatabase{
		entries:             []ToolEntry{},
		similarityThreshold: options.SimilarityThreshold,
		enabled:             options.Enabled,
		modelType:           options.ModelType,
		targetDim:           options.TargetDimension,
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

	logging.Infof("[Tool Selection] Loading tools and generating embeddings with concurrent processing (model: %s, dimension: %d)...",
		db.modelType, db.targetDim)

	// Use worker pool for concurrent embedding generation
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(toolEntries) {
		numWorkers = len(toolEntries)
	}

	type result struct {
		entry ToolEntry
		err   error
	}

	resultChan := make(chan result, len(toolEntries))
	entryChan := make(chan ToolEntry, len(toolEntries))

	// Send all entries to channel
	for _, entry := range toolEntries {
		entryChan <- entry
	}
	close(entryChan)

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for entry := range entryChan {
				// Generate embedding using GetEmbeddingWithModelType (aligned with Embedding Signal)
				output, err := candle_binding.GetEmbeddingWithModelType(entry.Description, db.modelType, db.targetDim)
				if err != nil {
					resultChan <- result{entry: entry, err: err}
				} else {
					entry.Embedding = output.Embedding
					resultChan <- result{entry: entry, err: nil}
				}
			}
		}(i)
	}

	// Close result channel when all workers are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	successCount := 0
	failedCount := 0

	db.mu.Lock()
	defer db.mu.Unlock()

	for res := range resultChan {
		if res.err != nil {
			logging.Warnf("Failed to generate embedding for tool %s: %v", res.entry.Tool.Function.Name, res.err)
			failedCount++
		} else {
			// Add to the database
			db.entries = append(db.entries, res.entry)
			logging.Debugf("[Tool Selection] Loaded tool: %s - %s",
				res.entry.Tool.Function.Name, res.entry.Description)
			successCount++
		}
	}

	logging.Infof("[Tool Selection] Loaded %d/%d tools from file: %s using model: %s (workers: %d)",
		successCount, len(toolEntries), filePath, db.modelType, numWorkers)
	return nil
}

// AddTool adds a tool to the database with automatic embedding generation
func (db *ToolsDatabase) AddTool(tool openai.ChatCompletionToolParam, description string, category string, tags []string) error {
	if !db.enabled {
		return nil
	}

	// Generate embedding using GetEmbeddingWithModelType (aligned with Embedding Signal)
	output, err := candle_binding.GetEmbeddingWithModelType(description, db.modelType, db.targetDim)
	if err != nil {
		return fmt.Errorf("failed to generate embedding for tool %s: %w", tool.Function.Name, err)
	}

	entry := ToolEntry{Tool: tool, Description: description, Embedding: output.Embedding, Category: category, Tags: tags}

	db.mu.Lock()
	defer db.mu.Unlock()

	db.entries = append(db.entries, entry)
	logging.Infof("Added tool: %s (%s) using model: %s", tool.Function.Name, description, db.modelType)

	return nil
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
	if !db.enabled {
		return []ToolSimilarity{}, nil
	}

	// Generate embedding using GetEmbeddingWithModelType (aligned with Embedding Signal)
	output, err := candle_binding.GetEmbeddingWithModelType(query, db.modelType, db.targetDim)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}
	queryEmbedding := output.Embedding

	db.mu.RLock()
	defer db.mu.RUnlock()

	// Calculate similarities
	results := make([]ToolSimilarity, 0, len(db.entries))
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

	// Sort by similarity (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	limit := topK
	if limit <= 0 || limit > len(results) {
		limit = len(results)
	}

	selected := results[:limit]
	for _, result := range selected {
		logging.Infof("Selected tool: %s (similarity=%.4f)",
			result.Entry.Tool.Function.Name, result.Similarity)
	}

	logging.Infof("Found %d similar tools for query: %s", len(selected), query)
	return selected, nil
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
