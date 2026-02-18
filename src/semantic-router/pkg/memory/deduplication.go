package memory

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DeduplicationConfig contains configuration for deduplication behavior
type DeduplicationConfig struct {
	// UpdateThreshold is the similarity threshold above which we UPDATE existing memory
	// Default: 0.9
	UpdateThreshold float32

	// SearchThreshold is the minimum similarity to consider when searching for similar memories
	// Default: 0.7
	SearchThreshold float32
}

// DefaultDeduplicationConfig returns default deduplication configuration
func DefaultDeduplicationConfig() DeduplicationConfig {
	return DeduplicationConfig{
		UpdateThreshold: 0.9,
		SearchThreshold: 0.7,
	}
}

// DeduplicationResult represents the result of deduplication check
type DeduplicationResult struct {
	// Action indicates what should be done: "update", "create", or "skip"
	Action string

	// ExistingMemory is the existing memory if similarity > UpdateThreshold
	ExistingMemory *Memory

	// Similarity is the similarity score (0.0 to 1.0)
	Similarity float32
}

// FindSimilar searches for existing similar memories.
// It searches for memories with the same user and type, and returns the most similar one.
//
// Parameters:
//   - ctx: context
//   - store: memory store to search in
//   - userID: user ID to filter by
//   - content: content to search for similarity
//   - memType: memory type to filter by
//   - config: deduplication configuration
//
// Returns:
//   - *Memory: the most similar existing memory (nil if none found)
//   - float32: similarity score (0.0 if none found)
func FindSimilar(
	ctx context.Context,
	store Store,
	userID string,
	content string,
	memType MemoryType,
	config DeduplicationConfig,
) (*Memory, float32) {
	if store == nil || !store.IsEnabled() {
		return nil, 0
	}

	// Search for similar memories using Retrieve
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:     content,
		UserID:    userID,
		Types:     []MemoryType{memType},
		Limit:     1, // Only need the most similar one
		Threshold: config.SearchThreshold,
	})
	if err != nil {
		logging.Debugf("Memory deduplication: search failed: %v", err)
		return nil, 0
	}

	if len(results) == 0 {
		logging.Debugf("Memory deduplication: no similar memories found for user=%s, type=%s", userID, memType)
		return nil, 0
	}

	// Return the most similar memory (first result, already sorted by similarity)
	result := results[0]
	logging.Debugf("Memory deduplication: found similar memory id=%s, similarity=%.3f", result.Memory.ID, result.Score)
	return result.Memory, result.Score
}

// CheckDeduplication checks if a new memory should be created, updated, or skipped.
//
// Logic:
//   - If similarity > UpdateThreshold (default 0.9): UPDATE existing memory
//   - If similarity 0.7-0.9: CREATE new (gray zone, conservative approach)
//   - If similarity < 0.7: CREATE new
//
// Parameters:
//   - ctx: context
//   - store: memory store to search in
//   - userID: user ID
//   - content: new memory content
//   - memType: memory type
//   - config: deduplication configuration
//
// Returns:
//   - DeduplicationResult with action, existing memory (if found), and similarity score
func CheckDeduplication(
	ctx context.Context,
	store Store,
	userID string,
	content string,
	memType MemoryType,
	config DeduplicationConfig,
) DeduplicationResult {
	existing, similarity := FindSimilar(ctx, store, userID, content, memType, config)

	if existing == nil || similarity < config.SearchThreshold {
		// No similar memory found or below search threshold
		return DeduplicationResult{
			Action:         "create",
			ExistingMemory: nil,
			Similarity:     0,
		}
	}

	if similarity > config.UpdateThreshold {
		// Very similar → UPDATE existing memory
		logging.Debugf("Memory deduplication: UPDATE existing memory id=%s (similarity=%.3f > %.3f)",
			existing.ID, similarity, config.UpdateThreshold)
		return DeduplicationResult{
			Action:         "update",
			ExistingMemory: existing,
			Similarity:     similarity,
		}
	}

	// Similarity in gray zone (0.7-0.9): CREATE new (conservative approach)
	logging.Debugf("Memory deduplication: CREATE new memory (similarity=%.3f in gray zone %.3f-%.3f)",
		similarity, config.SearchThreshold, config.UpdateThreshold)
	return DeduplicationResult{
		Action:         "create",
		ExistingMemory: existing,
		Similarity:     similarity,
	}
}
