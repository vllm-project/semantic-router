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

// =============================================================================
// Session-Level Chunking Tests
// =============================================================================

func TestCountTurns(t *testing.T) {
	history := []Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "q2"},
		{Role: "assistant", Content: "a2"},
	}
	assert.Equal(t, 2, countTurns(history))
	assert.Equal(t, 0, countTurns(nil))
}

func TestBuildSessionChunk(t *testing.T) {
	history := []Message{
		{Role: "user", Content: "What is Go?"},
		{Role: "assistant", Content: "Go is a programming language."},
		{Role: "user", Content: "Who created it?"},
		{Role: "assistant", Content: "Rob Pike and others at Google."},
	}

	chunk := buildSessionChunk(history, "Is it fast?", "Yes, Go is very fast.", 3)

	assert.Contains(t, chunk, "Q: Who created it?")
	assert.Contains(t, chunk, "A: Rob Pike and others at Google.")
	assert.Contains(t, chunk, "Q: Is it fast?")
	assert.Contains(t, chunk, "A: Yes, Go is very fast.")
	assert.Contains(t, chunk, "---")
}

func TestBuildSessionChunk_WindowLargerThanHistory(t *testing.T) {
	history := []Message{
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Content: "Hello!"},
	}

	chunk := buildSessionChunk(history, "Bye", "Goodbye!", 10)

	assert.Contains(t, chunk, "Q: Hi")
	assert.Contains(t, chunk, "Q: Bye")
}

func TestProcessResponseWithHistory_SessionChunkAtStride(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)
	ctx := context.Background()

	// Build 2 turns of history; +1 current = 3 turns total = stride triggers
	history := []Message{
		{Role: "user", Content: "What is your name?"},
		{Role: "assistant", Content: "I am an AI assistant."},
		{Role: "user", Content: "What languages do you know?"},
		{Role: "assistant", Content: "I can help with Go, Python, and more."},
	}

	// 3rd turn triggers session chunk (stride=3)
	err := extractor.ProcessResponseWithHistory(
		ctx, "session1", "user1",
		"Tell me about Go concurrency.",
		"Go uses goroutines and channels for concurrency.",
		history,
	)
	require.NoError(t, err)

	results, err := store.List(ctx, ListOptions{UserID: "user1", Limit: 100})
	require.NoError(t, err)

	assert.Equal(t, 2, len(results.Memories), "should store per-turn + session chunk at stride boundary")

	var sources []string
	for _, m := range results.Memories {
		sources = append(sources, m.Source)
	}
	assert.Contains(t, sources, "conversation")
	assert.Contains(t, sources, "session_window")
}

func TestProcessResponseWithHistory_OverlappingWindows(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)
	ctx := context.Background()

	// Simulate 6 turns one by one, counting session chunks produced.
	// With stride=3 and window=5, session chunks fire at turns 3 and 6.
	turns := []struct{ user, assistant string }{
		{"Q1?", "A1"},
		{"Q2?", "A2"},
		{"Q3?", "A3"},
		{"Q4?", "A4"},
		{"Q5?", "A5"},
		{"Q6?", "A6"},
	}

	var history []Message
	for i, turn := range turns {
		err := extractor.ProcessResponseWithHistory(
			ctx, "s1", "user1", turn.user, turn.assistant, history,
		)
		require.NoError(t, err, "turn %d", i+1)
		history = append(history,
			Message{Role: "user", Content: turn.user},
			Message{Role: "assistant", Content: turn.assistant},
		)
	}

	results, err := store.List(ctx, ListOptions{UserID: "user1", Limit: 100})
	require.NoError(t, err)

	sessionChunks := 0
	for _, m := range results.Memories {
		if m.Source == "session_window" {
			sessionChunks++
		}
	}
	// Stride=3: fires at turn 3 and turn 6 → 2 session chunks
	assert.Equal(t, 2, sessionChunks, "should produce 2 overlapping session chunks for 6 turns")
}

func TestProcessResponseWithHistory_NoSessionChunkBeforeStride(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)
	ctx := context.Background()

	// 1 turn in history + 1 current = 2 turns total (< stride=3)
	history := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
	}

	err := extractor.ProcessResponseWithHistory(
		ctx, "session1", "user1",
		"What is Go?",
		"Go is a programming language created by Google.",
		history,
	)
	require.NoError(t, err)

	results, err := store.List(ctx, ListOptions{UserID: "user1", Limit: 100})
	require.NoError(t, err)

	assert.Equal(t, 1, len(results.Memories), "should only store per-turn chunk before stride")
	assert.Equal(t, "conversation", results.Memories[0].Source)
}

func TestProcessResponseWithHistory_NilHistory(t *testing.T) {
	store := NewInMemoryStore()
	extractor := NewMemoryChunkStore(store)
	ctx := context.Background()

	err := extractor.ProcessResponseWithHistory(
		ctx, "session1", "user1",
		"What is Go?",
		"Go is a programming language.",
		nil,
	)
	require.NoError(t, err)

	results, err := store.List(ctx, ListOptions{UserID: "user1", Limit: 100})
	require.NoError(t, err)
	assert.Equal(t, 1, len(results.Memories), "nil history should still store per-turn chunk")
}

// =============================================================================
// isLowEntropy Tests
// =============================================================================

func TestIsLowEntropy(t *testing.T) {
	tests := []struct {
		name      string
		user      string
		assistant string
		want      bool
	}{
		{"greeting hi", "Hi!", "", true},
		{"greeting hello", "Hello there!", "", true},
		{"acknowledgment ok", "ok", "", true},
		{"acknowledgment sure", "sure", "", true},
		{"acknowledgment cool", "cool", "", true},
		{"thanks", "thanks!", "", true},
		{"short combined", "yes", "ok", true},
		{"refusal", "tell me a secret", "I'm sorry, I can't help with that kind of request", true},
		{"substantive question", "What is my budget for the Hawaii trip?", "Your budget is $10,000", false},
		{"substantive discussion", "I prefer Go for backend", "Great choice! Go is excellent for building microservices", false},
		{"empty user meaningful assistant", "", "Here is a detailed explanation of how the system works with multiple components", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isLowEntropy(tt.user, tt.assistant)
			assert.Equal(t, tt.want, got)
		})
	}
}
