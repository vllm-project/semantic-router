package memory

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// =============================================================================
// Memory Chunk Store
// =============================================================================

// MemoryExtractor stores conversation turns directly in the vector store.
// No LLM extraction is performed -- the original user question and assistant
// response (with think tags stripped) are embedded and stored as-is, preserving
// full conversation fidelity with zero additional inference cost.
//
// Usage:
//
//	store := NewMemoryChunkStore(milvusStore)
//	err := store.ProcessResponse(ctx, sessionID, userID, userMsg, assistantMsg)
type MemoryExtractor struct {
	store Store
}

// NewMemoryChunkStore creates a MemoryExtractor backed by the given Store.
// Unlike the previous LLM-based extractor, this requires no external model
// endpoint -- conversation chunks are embedded and stored directly.
func NewMemoryChunkStore(store Store) *MemoryExtractor {
	if store == nil {
		return nil
	}
	return &MemoryExtractor{store: store}
}

// =============================================================================
// Think-Tag Stripping
// =============================================================================

var thinkClosedPattern = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)
var thinkUnclosedPattern = regexp.MustCompile(`(?s)<think>.*`)

// StripThinkTags removes <think>...</think> blocks (and unclosed <think> tails)
// from LLM output. This is a safety net for backends that embed reasoning in
// the content field instead of the separate reasoning_content field.
func StripThinkTags(s string) string {
	s = thinkClosedPattern.ReplaceAllString(s, "")
	s = thinkUnclosedPattern.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

// =============================================================================
// Direct Chunk Storage
// =============================================================================

// ProcessResponse stores the current conversation turn directly in the vector
// store. The user message and assistant response are combined into a single
// chunk (Q: ... / A: ...) so that semantic search can match both question-style
// and answer-style queries. Think tags in the assistant response are stripped.
//
// This replaces the previous LLM-based extraction pipeline, eliminating extra
// inference tokens, JSON parsing fragility, and information loss.
func (e *MemoryExtractor) ProcessResponse(
	ctx context.Context,
	_ string, // sessionID (unused, kept for interface compatibility)
	userID string,
	userMessage string,
	assistantResponse string,
) error {
	if e == nil || e.store == nil || !e.store.IsEnabled() {
		logging.Infof("Memory chunk store: SKIPPED - store not enabled (store=%v)", e != nil && e.store != nil)
		return nil
	}

	assistantResponse = StripThinkTags(assistantResponse)

	if userMessage == "" && assistantResponse == "" {
		logging.Debugf("Memory chunk store: SKIPPED - empty turn")
		return nil
	}

	startTime := time.Now()
	status := "success"
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryExtraction(status, duration, 1, "episodic")
	}()

	chunk := formatTurnChunk(userMessage, assistantResponse)

	mem := &Memory{
		ID:         generateMemoryID(),
		Type:       MemoryTypeEpisodic,
		Content:    chunk,
		UserID:     userID,
		Source:     "conversation",
		CreatedAt:  time.Now(),
		Importance: 0.5,
	}

	if err := e.store.Store(ctx, mem); err != nil {
		status = "error"
		return fmt.Errorf("failed to store conversation chunk: %w", err)
	}

	logging.Infof("Memory chunk store: stored turn for user=%s (len=%d)", userID, len(chunk))
	return nil
}

// formatTurnChunk combines a user message and assistant response into a single
// searchable chunk. Both halves are included so that semantic search can match
// on either the question or the answer.
func formatTurnChunk(userMessage, assistantResponse string) string {
	var parts []string
	if userMessage != "" {
		parts = append(parts, "Q: "+userMessage)
	}
	if assistantResponse != "" {
		parts = append(parts, "A: "+assistantResponse)
	}
	return strings.Join(parts, "\n")
}

// =============================================================================
// Helpers
// =============================================================================

func generateMemoryID() string {
	return fmt.Sprintf("mem_%d", time.Now().UnixNano())
}
