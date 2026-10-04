package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// LoadToolsFromFile appends a validated batch in normalized-name order.
// Embeddings run concurrently but worker completion order never controls
// catalog or error ordering. Any invalid definition, duplicate, or embedding
// failure rejects the whole batch and leaves the existing catalog unchanged.
func (db *ToolsDatabase) LoadToolsFromFile(filePath string) error {
	if !db.enabled {
		return nil
	}
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read tools file: %w", err)
	}
	entries, err := decodeCatalog(data)
	if err != nil {
		return fmt.Errorf("failed to parse tools JSON: %w", err)
	}
	logging.ComponentEvent("tools", "tool_database_load_started", map[string]interface{}{
		"file_path": filePath, "model_type": db.modelType, "target_dimension": db.targetDim,
		"tool_count": len(entries), "worker_count": min(runtime.NumCPU()*2, len(entries)),
	})
	if err := db.embedCatalog(entries); err != nil {
		return err
	}
	if err := db.appendEntries(entries); err != nil {
		return err
	}
	logging.ComponentEvent("tools", "tool_database_loaded", map[string]interface{}{
		"file_path": filePath, "model_type": db.modelType, "target_dimension": db.targetDim,
		"tool_count": len(entries), "loaded_count": len(entries), "failed_count": 0,
		"worker_count": min(runtime.NumCPU()*2, len(entries)),
	})
	return nil
}

func decodeCatalog(data []byte) ([]ToolEntry, error) {
	var raw []struct {
		Tool        json.RawMessage `json:"tool"`
		Description string          `json:"description"`
		Tags        []string        `json:"tags"`
		Category    string          `json:"category"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	entries := make([]ToolEntry, 0, len(raw))
	names := make(map[string]bool, len(raw))
	for _, item := range raw {
		tool, err := decodeTool(item.Tool)
		if err != nil {
			return nil, err
		}
		if names[tool.Function.Name] {
			return nil, fmt.Errorf("tools: duplicate normalized function name")
		}
		names[tool.Function.Name] = true
		entries = append(entries, ToolEntry{
			Tool: tool, Description: item.Description, Tags: item.Tags, Category: item.Category,
		})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Tool.Function.Name < entries[j].Tool.Function.Name })
	return entries, nil
}

func (db *ToolsDatabase) embedCatalog(entries []ToolEntry) error {
	jobs := make(chan int, len(entries))
	for i := range entries {
		jobs <- i
	}
	close(jobs)
	errors := make([]error, len(entries))
	var workers sync.WaitGroup
	for i := 0; i < min(runtime.NumCPU()*2, len(entries)); i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for index := range jobs {
				vector, err := db.embedText(entries[index].Description)
				if err == nil {
					err = validateEmbedding(vector)
				}
				entries[index].Embedding, errors[index] = vector, err
			}
		}()
	}
	workers.Wait()
	for _, err := range errors {
		if err != nil {
			return fmt.Errorf("failed to generate catalog embedding: %w", err)
		}
	}
	return nil
}
