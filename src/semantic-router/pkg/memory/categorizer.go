package memory

import (
	"context"
	"crypto/sha256"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// CategorizerConfig holds tunable parameters for auto-categorization and summary generation.
type CategorizerConfig struct {
	// AbstractMaxLen is the max character length for L0 abstracts.
	AbstractMaxLen int

	// OverviewMaxLen is the max character length for L1 overviews.
	OverviewMaxLen int

	// TopicKeywords maps topic labels to keyword lists for extraction.
	// If nil, DefaultTopicKeywords is used.
	TopicKeywords map[string][]string
}

// ApplyDefaults fills zero-valued fields.
func (c *CategorizerConfig) ApplyDefaults() {
	if c.AbstractMaxLen <= 0 {
		c.AbstractMaxLen = 200
	}
	if c.OverviewMaxLen <= 0 {
		c.OverviewMaxLen = 600
	}
	if c.TopicKeywords == nil {
		c.TopicKeywords = DefaultTopicKeywords()
	}
}

// DefaultTopicKeywords returns the built-in topic-to-keyword mapping.
func DefaultTopicKeywords() map[string][]string {
	return map[string][]string{
		"deployment":     {"deploy", "rollback", "release", "ci/cd", "pipeline"},
		"preferences":    {"prefer", "like", "favorite", "style", "convention"},
		"travel":         {"travel", "flight", "hotel", "vacation", "trip"},
		"coding":         {"code", "programming", "debug", "refactor", "review"},
		"communication":  {"email", "meeting", "slack", "chat", "discuss"},
		"finance":        {"budget", "cost", "price", "payment", "invoice"},
		"project":        {"project", "milestone", "sprint", "deadline", "task"},
		"infrastructure": {"server", "cluster", "database", "container", "kubernetes"},
		"security":       {"security", "auth", "permission", "access", "credential"},
		"testing":        {"test", "qa", "coverage", "benchmark", "regression"},
	}
}

// CategoryTree manages the hierarchical category structure for memories.
// Categories are organized as: Type (semantic/procedural/episodic) -> Topic -> Sub-topic.
// Each category is itself a Memory with IsCategory=true.
type CategoryTree struct {
	store           Store
	embeddingConfig EmbeddingConfig
	config          CategorizerConfig
}

// NewCategoryTree creates a CategoryTree that operates on the given store.
func NewCategoryTree(store Store, embeddingCfg EmbeddingConfig, cfg CategorizerConfig) *CategoryTree {
	cfg.ApplyDefaults()
	return &CategoryTree{
		store:           store,
		embeddingConfig: embeddingCfg,
		config:          cfg,
	}
}

// categorizeResult holds the output of auto-categorization.
type categorizeResult struct {
	ParentID string
	Abstract string
	Overview string
}

// AutoCategorize assigns a memory to an existing or new category node
// and generates L0 (Abstract) and L1 (Overview) summaries.
//
// Returns an error if the category node could not be created or persisted.
func (ct *CategoryTree) AutoCategorize(ctx context.Context, mem *Memory) (*categorizeResult, error) {
	if mem.IsCategory {
		return nil, nil
	}

	topic := extractTopic(mem.Content, mem.Type, ct.config.TopicKeywords)
	categoryID := buildCategoryID(mem.UserID, mem.GroupID, mem.Type, topic)

	existing, err := ct.store.Get(ctx, categoryID)
	if err == nil && existing != nil {
		return &categorizeResult{
			ParentID: categoryID,
			Abstract: generateAbstract(mem.Content, ct.config.AbstractMaxLen),
			Overview: generateOverview(mem.Content, ct.config.OverviewMaxLen),
		}, nil
	}

	categoryContent := fmt.Sprintf("Category: %s/%s", mem.Type, topic)
	categoryAbstract := fmt.Sprintf("%s memories about %s", mem.Type, topic)

	categoryMem := &Memory{
		ID:         categoryID,
		Type:       mem.Type,
		Content:    categoryContent,
		UserID:     mem.UserID,
		GroupID:    mem.GroupID,
		ProjectID:  mem.ProjectID,
		IsCategory: true,
		Abstract:   categoryAbstract,
		Visibility: mem.Visibility,
		CreatedAt:  time.Now(),
		Source:     "auto-categorization",
	}

	embedding, embErr := GenerateEmbedding(categoryContent, ct.embeddingConfig)
	if embErr != nil {
		return nil, fmt.Errorf("failed to generate embedding for category %s: %w", categoryID, embErr)
	}
	categoryMem.Embedding = embedding

	if storeErr := ct.store.Store(ctx, categoryMem); storeErr != nil {
		return nil, fmt.Errorf("failed to store category %s: %w", categoryID, storeErr)
	}

	logging.Debugf("CategoryTree: created category %s for memory %s", categoryID, mem.ID)

	return &categorizeResult{
		ParentID: categoryID,
		Abstract: generateAbstract(mem.Content, ct.config.AbstractMaxLen),
		Overview: generateOverview(mem.Content, ct.config.OverviewMaxLen),
	}, nil
}

// extractTopic derives a topic label from the memory content using keyword matching.
func extractTopic(content string, memType MemoryType, topicKeywords map[string][]string) string {
	lower := strings.ToLower(content)

	bestTopic := "general"
	bestScore := 0

	for topic, keywords := range topicKeywords {
		score := 0
		for _, kw := range keywords {
			if strings.Contains(lower, kw) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			bestTopic = topic
		}
	}

	return bestTopic
}

// buildCategoryID creates a deterministic, unique ID for a category node.
func buildCategoryID(userID, groupID string, memType MemoryType, topic string) string {
	scope := userID
	if groupID != "" {
		scope = "group:" + groupID
	}
	raw := fmt.Sprintf("cat:%s:%s:%s", scope, memType, topic)
	hash := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("cat-%x", hash[:8])
}

// generateAbstract produces a concise L0 summary by extracting the first sentence
// or truncating at maxLen characters.
func generateAbstract(content string, maxLen int) string {
	for _, sep := range []string{". ", ".\n", "! ", "? "} {
		if idx := strings.Index(content, sep); idx > 0 && idx < maxLen {
			return content[:idx+1]
		}
	}

	if len(content) <= maxLen {
		return content
	}
	truncated := content[:maxLen]
	if lastSpace := strings.LastIndex(truncated, " "); lastSpace > maxLen/2 {
		return truncated[:lastSpace] + "..."
	}
	return truncated + "..."
}

// generateOverview produces an L1 summary by returning the content up to maxLen characters.
func generateOverview(content string, maxLen int) string {
	if len(content) <= maxLen {
		return content
	}
	truncated := content[:maxLen]
	if lastSpace := strings.LastIndex(truncated, " "); lastSpace > maxLen*2/3 {
		return truncated[:lastSpace] + "..."
	}
	return truncated + "..."
}

// EnrichMemoryBeforeStore populates hierarchical fields on a memory before it is persisted.
// Returns an error if enrichment fails so the caller can decide how to handle it.
func EnrichMemoryBeforeStore(ctx context.Context, store Store, mem *Memory, embeddingCfg EmbeddingConfig, catCfg CategorizerConfig) error {
	if mem.Visibility == "" {
		mem.Visibility = VisibilityUser
	}

	ct := NewCategoryTree(store, embeddingCfg, catCfg)
	result, err := ct.AutoCategorize(ctx, mem)
	if err != nil {
		return fmt.Errorf("auto-categorization failed for %s: %w", mem.ID, err)
	}
	if result == nil {
		return nil
	}

	if mem.ParentID == "" {
		mem.ParentID = result.ParentID
	}
	if mem.Abstract == "" {
		mem.Abstract = result.Abstract
	}
	if mem.Overview == "" {
		mem.Overview = result.Overview
	}
	return nil
}
