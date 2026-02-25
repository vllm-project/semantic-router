package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// NewMemoryChunkStore Tests
// =============================================================================

func TestNewMemoryChunkStore_NilStore(t *testing.T) {
	extractor := NewMemoryChunkStore(nil)
	assert.Nil(t, extractor, "should return nil when store is nil")
}

func TestNewMemoryChunkStore_ValidStore(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)
	assert.NotNil(t, extractor, "should return non-nil for valid store")
}

// =============================================================================
// ProcessResponse Tests
// =============================================================================

func TestProcessResponse_StoresChunk(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)

	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"What is the capital of France?",
		"The capital of France is Paris.",
	)
	require.NoError(t, err)

	results, err := store.List(context.Background(), ListOptions{UserID: "user1", Limit: 10})
	require.NoError(t, err)
	require.Len(t, results.Memories, 1)

	mem := results.Memories[0]
	assert.Equal(t, MemoryTypeEpisodic, mem.Type)
	assert.Equal(t, "user1", mem.UserID)
	assert.Equal(t, "conversation", mem.Source)
	assert.Contains(t, mem.Content, "Q: What is the capital of France?")
	assert.Contains(t, mem.Content, "A: The capital of France is Paris.")
}

func TestProcessResponse_StripsThinkTags(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)

	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"What is 2+2?",
		"<think>Let me calculate... 2+2=4</think>The answer is 4.",
	)
	require.NoError(t, err)

	results, err := store.List(context.Background(), ListOptions{UserID: "user1", Limit: 10})
	require.NoError(t, err)
	require.Len(t, results.Memories, 1)

	assert.NotContains(t, results.Memories[0].Content, "<think>")
	assert.Contains(t, results.Memories[0].Content, "A: The answer is 4.")
}

func TestProcessResponse_StripsUnclosedThinkTags(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)

	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"Tell me a fact",
		"<think>Reasoning about this...",
	)
	require.NoError(t, err)

	results, err := store.List(context.Background(), ListOptions{UserID: "user1", Limit: 10})
	require.NoError(t, err)

	// Unclosed think tag strips everything after <think>, leaving empty assistant response.
	// Only user message should remain.
	require.Len(t, results.Memories, 1)
	assert.Contains(t, results.Memories[0].Content, "Q: Tell me a fact")
	assert.NotContains(t, results.Memories[0].Content, "<think>")
}

func TestProcessResponse_EmptyTurnSkipped(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)

	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"", "",
	)
	require.NoError(t, err)

	results, err := store.List(context.Background(), ListOptions{UserID: "user1", Limit: 10})
	require.NoError(t, err)
	assert.Len(t, results.Memories, 0, "empty turns should not be stored")
}

func TestProcessResponse_OnlyUserMessage(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)

	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"Hello world", "",
	)
	require.NoError(t, err)

	results, err := store.List(context.Background(), ListOptions{UserID: "user1", Limit: 10})
	require.NoError(t, err)
	require.Len(t, results.Memories, 1)
	assert.Equal(t, "Q: Hello world", results.Memories[0].Content)
}

func TestProcessResponse_OnlyAssistantResponse(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)

	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"", "Here is the answer.",
	)
	require.NoError(t, err)

	results, err := store.List(context.Background(), ListOptions{UserID: "user1", Limit: 10})
	require.NoError(t, err)
	require.Len(t, results.Memories, 1)
	assert.Equal(t, "A: Here is the answer.", results.Memories[0].Content)
}

func TestProcessResponse_StoreDisabled(t *testing.T) {
	disabledStore := &InMemoryStore{
		memories: make(map[string]*Memory),
		enabled:  false,
	}
	extractor := NewMemoryChunkStore(disabledStore)

	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"test", "response",
	)
	assert.NoError(t, err, "should return nil when store is disabled")
}

func TestProcessResponse_NilExtractor(t *testing.T) {
	var extractor *MemoryExtractor
	err := extractor.ProcessResponse(
		context.Background(), "session1", "user1",
		"test", "response",
	)
	assert.NoError(t, err, "nil extractor should not panic")
}

func TestProcessResponse_MultipleTurns(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1",
			"question", "answer",
		)
		require.NoError(t, err)
	}

	results, err := store.List(ctx, ListOptions{UserID: "user1", Limit: 100})
	require.NoError(t, err)
	assert.Len(t, results.Memories, 5, "each turn should be stored independently")
}

// =============================================================================
// StripThinkTags Tests
// =============================================================================

func TestStripThinkTags(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"no tags", "Hello world", "Hello world"},
		{"closed tag", "<think>reasoning</think>Answer", "Answer"},
		{"unclosed tag", "<think>reasoning...", ""},
		{"multiline closed", "<think>\nline1\nline2\n</think>Answer", "Answer"},
		{"multiple closed tags", "<think>a</think>middle<think>b</think>end", "middleend"},
		{"empty", "", ""},
		{"only think tag", "<think>all reasoning</think>", ""},
		{"whitespace after closed", "<think>r</think>   Answer", "Answer"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := StripThinkTags(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// =============================================================================
// formatTurnChunk Tests
// =============================================================================

func TestFormatTurnChunk(t *testing.T) {
	tests := []struct {
		name      string
		user      string
		assistant string
		expected  string
	}{
		{"both", "question", "answer", "Q: question\nA: answer"},
		{"user only", "question", "", "Q: question"},
		{"assistant only", "", "answer", "A: answer"},
		{"both empty", "", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatTurnChunk(tt.user, tt.assistant)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// =============================================================================
// generateMemoryID Tests
// =============================================================================

func TestGenerateMemoryID(t *testing.T) {
	id1 := generateMemoryID()
	id2 := generateMemoryID()
	assert.True(t, strings.HasPrefix(id1, "mem_"))
	assert.True(t, strings.HasPrefix(id2, "mem_"))
	assert.NotEqual(t, id1, id2, "IDs should be unique")
}
