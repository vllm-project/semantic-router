package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// =============================================================================
// Test Helpers
// =============================================================================

// createMockRouterConfig creates a RouterConfig with external_models pointing to the test server
func createMockRouterConfig(serverURL string) *config.RouterConfig {
	if serverURL == "" {
		return nil
	}

	// Parse URL to extract host and port (skip "http://")
	hostPort := strings.TrimPrefix(serverURL, "http://")
	parts := strings.Split(hostPort, ":")
	address := parts[0]
	port := 0
	if len(parts) > 1 {
		port, _ = strconv.Atoi(parts[1])
	}

	return &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{
			{
				Provider:  "vllm",
				ModelRole: config.ModelRoleMemoryExtraction,
				ModelEndpoint: config.ClassifierVLLMEndpoint{
					Address: address,
					Port:    port,
				},
				ModelName:      "test-model",
				TimeoutSeconds: 30,
			},
		},
	}
}

// =============================================================================
// ExtractFacts Tests
// =============================================================================

func TestExtractFacts_DisabledConfig(t *testing.T) {
	// Test with nil routerCfg (no external model configured)
	extractor := NewMemoryExtractorWithStore(nil, 10, nil)
	assert.Nil(t, extractor, "should return nil when no external model configured")

	// Test with empty routerCfg (no memory_extraction role)
	extractor = NewMemoryExtractorWithStore(&config.RouterConfig{}, 10, nil)
	assert.Nil(t, extractor, "should return nil when no memory_extraction role")
}

func TestExtractFacts_EmptyMessages(t *testing.T) {
	routerCfg := createMockRouterConfig("http://localhost:8080")
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	facts, err := extractor.ExtractFacts(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, facts, "should return nil for nil messages")

	facts, err = extractor.ExtractFacts(context.Background(), []Message{})
	require.NoError(t, err)
	assert.Nil(t, facts, "should return nil for empty messages")
}

func TestExtractFacts_SingleSemanticFact(t *testing.T) {
	// Create mock LLM server
	mockResponse := `[{"type": "semantic", "content": "User's budget for Hawaii vacation is $10,000"}]`
	server := createMockLLMServer(t, mockResponse)
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "My budget for the Hawaii trip is $10,000"},
		{Role: "assistant", Content: "That's a great budget for Hawaii!"},
	}

	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Equal(t, MemoryTypeSemantic, facts[0].Type)
	assert.Equal(t, "User's budget for Hawaii vacation is $10,000", facts[0].Content)
}

func TestExtractFacts_MultipleFacts(t *testing.T) {
	mockResponse := `[
		{"type": "semantic", "content": "User's budget for Hawaii vacation is $10,000"},
		{"type": "semantic", "content": "User prefers direct flights over connections"},
		{"type": "procedural", "content": "To book flights: check prices on Google Flights first"}
	]`
	server := createMockLLMServer(t, mockResponse)
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "My budget is $10,000. I prefer direct flights."},
		{Role: "assistant", Content: "I recommend checking Google Flights first for prices."},
	}

	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err)
	require.Len(t, facts, 3)

	assert.Equal(t, MemoryTypeSemantic, facts[0].Type)
	assert.Equal(t, MemoryTypeSemantic, facts[1].Type)
	assert.Equal(t, MemoryTypeProcedural, facts[2].Type)
}

func TestExtractFacts_EmptyArray(t *testing.T) {
	// LLM returns empty array when nothing to extract
	server := createMockLLMServer(t, "[]")
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "Hello!"},
		{Role: "assistant", Content: "Hi there! How can I help?"},
	}

	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err)
	assert.Nil(t, facts, "should return nil for empty extraction")
}

func TestExtractFacts_MarkdownCodeBlock(t *testing.T) {
	// LLM sometimes wraps JSON in markdown code blocks
	mockResponse := "```json\n[{\"type\": \"semantic\", \"content\": \"User likes coffee\"}]\n```"
	server := createMockLLMServer(t, mockResponse)
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "I love coffee"},
	}

	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err)
	require.Len(t, facts, 1)
	assert.Equal(t, "User likes coffee", facts[0].Content)
}

func TestExtractFacts_LLMError(t *testing.T) {
	// Create server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "My budget is $10,000"},
	}

	// Should return nil (graceful degradation), not error
	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err, "should not return error on LLM failure")
	assert.Nil(t, facts, "should return nil on LLM failure")
}

func TestExtractFacts_InvalidJSON(t *testing.T) {
	// LLM returns invalid JSON
	server := createMockLLMServer(t, "not valid json at all")
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "My budget is $10,000"},
	}

	// Should return nil (graceful degradation)
	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err, "should not return error on parse failure")
	assert.Nil(t, facts, "should return nil on parse failure")
}

func TestExtractFacts_InvalidType(t *testing.T) {
	// LLM returns invalid memory type - should be filtered
	mockResponse := `[
		{"type": "semantic", "content": "Valid fact"},
		{"type": "invalid_type", "content": "Should be skipped"},
		{"type": "procedural", "content": "Another valid fact"}
	]`
	server := createMockLLMServer(t, mockResponse)
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "Test message"},
	}

	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err)
	require.Len(t, facts, 2, "should filter out invalid type")

	assert.Equal(t, "Valid fact", facts[0].Content)
	assert.Equal(t, "Another valid fact", facts[1].Content)
}

func TestExtractFacts_EmptyContent(t *testing.T) {
	// Facts with empty content should be filtered
	mockResponse := `[
		{"type": "semantic", "content": "Valid fact"},
		{"type": "semantic", "content": ""},
		{"type": "semantic", "content": "   "}
	]`
	server := createMockLLMServer(t, mockResponse)
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "Test message"},
	}

	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err)
	require.Len(t, facts, 1, "should filter out empty content")
	assert.Equal(t, "Valid fact", facts[0].Content)
}

func TestExtractFacts_AllMemoryTypes(t *testing.T) {
	mockResponse := `[
		{"type": "semantic", "content": "User prefers window seats"},
		{"type": "procedural", "content": "To reset password: go to settings, click security"},
		{"type": "episodic", "content": "On Jan 5 2026, user booked flight to Hawaii"}
	]`
	server := createMockLLMServer(t, mockResponse)
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, nil)

	messages := []Message{
		{Role: "user", Content: "Various conversation"},
	}

	facts, err := extractor.ExtractFacts(context.Background(), messages)
	require.NoError(t, err)
	require.Len(t, facts, 3)

	assert.Equal(t, MemoryTypeSemantic, facts[0].Type)
	assert.Equal(t, MemoryTypeProcedural, facts[1].Type)
	assert.Equal(t, MemoryTypeEpisodic, facts[2].Type)
}

// =============================================================================
// parseExtractedFacts Tests
// =============================================================================

func TestParseExtractedFacts_ValidJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected int
	}{
		{
			name:     "single fact",
			input:    `[{"type": "semantic", "content": "Budget is $10K"}]`,
			expected: 1,
		},
		{
			name:     "multiple facts",
			input:    `[{"type": "semantic", "content": "A"}, {"type": "procedural", "content": "B"}]`,
			expected: 2,
		},
		{
			name:     "empty array",
			input:    `[]`,
			expected: 0,
		},
		{
			name:     "with whitespace",
			input:    `  [{"type": "semantic", "content": "Fact"}]  `,
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			facts, err := parseExtractedFacts(tt.input)
			require.NoError(t, err)
			assert.Len(t, facts, tt.expected)
		})
	}
}

func TestParseExtractedFacts_MarkdownCleanup(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "json code block",
			input: "```json\n[{\"type\": \"semantic\", \"content\": \"Fact\"}]\n```",
		},
		{
			name:  "plain code block",
			input: "```\n[{\"type\": \"semantic\", \"content\": \"Fact\"}]\n```",
		},
		{
			name:  "code block with extra whitespace",
			input: "```json\n  [{\"type\": \"semantic\", \"content\": \"Fact\"}]  \n```",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			facts, err := parseExtractedFacts(tt.input)
			require.NoError(t, err)
			require.Len(t, facts, 1)
			assert.Equal(t, "Fact", facts[0].Content)
		})
	}
}

func TestParseExtractedFacts_InvalidJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "not json",
			input: "this is not json",
		},
		{
			name:  "incomplete json",
			input: `[{"type": "semantic"`,
		},
		{
			name:  "wrong structure",
			input: `{"type": "semantic", "content": "fact"}`, // not an array
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			facts, err := parseExtractedFacts(tt.input)
			assert.Error(t, err, "should return error for: %s", tt.name)
			assert.Nil(t, facts)
		})
	}
}

// =============================================================================
// normalizeMemoryType Tests
// =============================================================================

func TestNormalizeMemoryType(t *testing.T) {
	tests := []struct {
		input    string
		expected MemoryType
	}{
		{"semantic", MemoryTypeSemantic},
		{"SEMANTIC", MemoryTypeSemantic},
		{"Semantic", MemoryTypeSemantic},
		{"  semantic  ", MemoryTypeSemantic},
		{"procedural", MemoryTypeProcedural},
		{"PROCEDURAL", MemoryTypeProcedural},
		{"episodic", MemoryTypeEpisodic},
		{"EPISODIC", MemoryTypeEpisodic},
		{"invalid", ""},
		{"", ""},
		{"unknown_type", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := normalizeMemoryType(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// =============================================================================
// cleanJSONResponse Tests
// =============================================================================

func TestCleanJSONResponse(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "plain json",
			input:    `[{"type": "semantic"}]`,
			expected: `[{"type": "semantic"}]`,
		},
		{
			name:     "json code block",
			input:    "```json\n[{\"type\": \"semantic\"}]\n```",
			expected: `[{"type": "semantic"}]`,
		},
		{
			name:     "plain code block",
			input:    "```\n[{\"type\": \"semantic\"}]\n```",
			expected: `[{"type": "semantic"}]`,
		},
		{
			name:     "with surrounding whitespace",
			input:    "  \n[{\"type\": \"semantic\"}]\n  ",
			expected: `[{"type": "semantic"}]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cleanJSONResponse(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// =============================================================================
// formatMessagesForExtraction Tests
// =============================================================================

func TestFormatMessagesForExtraction(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
		{Role: "user", Content: "My budget is $10,000"},
	}

	result := formatMessagesForExtraction(messages)

	assert.Contains(t, result, "[user]: Hello")
	assert.Contains(t, result, "[assistant]: Hi there!")
	assert.Contains(t, result, "[user]: My budget is $10,000")
}

func TestFormatMessagesForExtraction_Empty(t *testing.T) {
	result := formatMessagesForExtraction(nil)
	assert.Empty(t, result)

	result = formatMessagesForExtraction([]Message{})
	assert.Empty(t, result)
}

// =============================================================================
// truncateForLog Tests
// =============================================================================

func TestTruncateForLog(t *testing.T) {
	tests := []struct {
		input    string
		maxLen   int
		expected string
	}{
		{"short", 10, "short"},
		{"exactly10!", 10, "exactly10!"},
		{"this is longer than ten", 10, "this is lo..."},
		{"", 10, ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := truncateForLog(tt.input, tt.maxLen)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// =============================================================================
// ProcessResponse Batch Size Threshold Tests
// =============================================================================

// TestProcessResponse_DefaultBatchSize tests default batch size of 10 turns.
// Verifies extraction only happens when turnCount % 10 == 0.
func TestProcessResponse_DefaultBatchSize(t *testing.T) {
	server, callCount := createTrackingMockLLMServer(t, "[]")
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, store)
	require.NotNil(t, extractor)

	ctx := context.Background()
	history := []Message{{Role: "user", Content: "test"}, {Role: "assistant", Content: "response"}}

	// Turns 1-9 should NOT trigger extraction
	for i := 1; i <= 9; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(0), atomic.LoadInt32(callCount), "no extraction should happen before turn 10")

	// Turn 10 SHOULD trigger extraction
	err := extractor.ProcessResponse(ctx, "session1", "user1", history)
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "extraction should happen at turn 10")
}

// TestProcessResponse_DefaultBatchSizeBoundary tests boundary conditions at exactly turn 10.
// Verifies turn 9 does NOT extract, turn 10 DOES extract, turn 11 does NOT extract.
func TestProcessResponse_DefaultBatchSizeBoundary(t *testing.T) {
	server, callCount := createTrackingMockLLMServer(t, "[]")
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, store)
	require.NotNil(t, extractor)

	ctx := context.Background()
	history := []Message{{Role: "user", Content: "test"}, {Role: "assistant", Content: "response"}}

	// Advance to turn 9 (should not extract)
	for i := 1; i <= 9; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(0), atomic.LoadInt32(callCount), "turn 9: no extraction (9 % 10 != 0)")

	// Turn 10 (should extract)
	err := extractor.ProcessResponse(ctx, "session1", "user1", history)
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "turn 10: extraction (10 % 10 == 0)")

	// Turn 11 (should NOT extract)
	err = extractor.ProcessResponse(ctx, "session1", "user1", history)
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "turn 11: no extraction (11 % 10 != 0)")
}

// TestProcessResponse_CustomBatchSize tests custom batch size of 5 turns.
// Verifies extraction happens at turn 5.
func TestProcessResponse_CustomBatchSize(t *testing.T) {
	server, callCount := createTrackingMockLLMServer(t, "[]")
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 5, store)
	require.NotNil(t, extractor)

	ctx := context.Background()
	history := []Message{{Role: "user", Content: "test"}, {Role: "assistant", Content: "response"}}

	// Turns 1-4 should NOT trigger extraction
	for i := 1; i <= 4; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(0), atomic.LoadInt32(callCount), "no extraction before turn 5")

	// Turn 5 SHOULD trigger extraction
	err := extractor.ProcessResponse(ctx, "session1", "user1", history)
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "extraction at turn 5 (5 % 5 == 0)")
}

// TestProcessResponse_BatchSizeZeroUsesDefault tests that BatchSize = 0 defaults to 10.
func TestProcessResponse_BatchSizeZeroUsesDefault(t *testing.T) {
	routerCfg := createMockRouterConfig("http://localhost:8080")
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 0, store)
	require.NotNil(t, extractor)

	assert.Equal(t, 10, extractor.batchSize, "batch size 0 should default to 10")
}

// TestProcessResponse_BatchSizeNegativeUsesDefault tests that BatchSize < 0 defaults to 10.
func TestProcessResponse_BatchSizeNegativeUsesDefault(t *testing.T) {
	routerCfg := createMockRouterConfig("http://localhost:8080")
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, -1, store)
	require.NotNil(t, extractor)

	assert.Equal(t, 10, extractor.batchSize, "negative batch size should default to 10")
}

// TestProcessResponse_MultipleSessionsIndependent tests that turn counts are tracked
// independently per session.
func TestProcessResponse_MultipleSessionsIndependent(t *testing.T) {
	server, callCount := createTrackingMockLLMServer(t, "[]")
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 5, store)
	require.NotNil(t, extractor)

	ctx := context.Background()
	history := []Message{{Role: "user", Content: "test"}, {Role: "assistant", Content: "response"}}

	// Session1: 5 turns (should extract at turn 5)
	for i := 1; i <= 5; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "session1 should extract at turn 5")

	// Session2: 3 turns (should NOT extract yet)
	for i := 1; i <= 3; i++ {
		err := extractor.ProcessResponse(ctx, "session2", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "session2 should not extract at turn 3")

	// Session2: 2 more turns to reach turn 5 (should extract)
	for i := 4; i <= 5; i++ {
		err := extractor.ProcessResponse(ctx, "session2", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(2), atomic.LoadInt32(callCount), "session2 should extract at turn 5")
}

// TestProcessResponse_BatchSelectionLogic tests that when history > batchSize+5,
// only the last batchSize+5 messages are used for extraction.
func TestProcessResponse_BatchSelectionLogic(t *testing.T) {
	var mu sync.Mutex
	var capturedBody []byte
	var readErr error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		body, err := io.ReadAll(r.Body)
		mu.Lock()
		capturedBody = body
		readErr = err
		mu.Unlock()

		response := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "[]"}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 5, store)
	require.NotNil(t, extractor)

	ctx := context.Background()

	history := make([]Message, 20)
	for i := 0; i < 20; i++ {
		if i%2 == 0 {
			history[i] = Message{Role: "user", Content: fmt.Sprintf("message_%d", i)}
		} else {
			history[i] = Message{Role: "assistant", Content: fmt.Sprintf("response_%d", i)}
		}
	}

	for i := 1; i <= 5; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}

	// With 20 messages and batchSize=5, batchStart = 20 - (5+5) = 10
	// So messages[10:] should be sent (indices 10-19)
	mu.Lock()
	require.NoError(t, readErr)
	require.NotNil(t, capturedBody, "LLM should have been called")
	bodyStr := string(capturedBody)
	mu.Unlock()

	assert.NotContains(t, bodyStr, "message_0", "early messages should not be in batch")
	assert.NotContains(t, bodyStr, "message_8", "message_8 should not be in batch")
	assert.Contains(t, bodyStr, "message_10", "message_10 should be in batch")
	assert.Contains(t, bodyStr, "message_18", "message_18 should be in batch")
}

// TestProcessResponse_BatchSelectionSmallHistory tests batch selection with history
// smaller than batchSize+5. Verifies all messages are included.
func TestProcessResponse_BatchSelectionSmallHistory(t *testing.T) {
	var mu sync.Mutex
	var capturedBody []byte
	var readErr error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		body, err := io.ReadAll(r.Body)
		mu.Lock()
		capturedBody = body
		readErr = err
		mu.Unlock()

		response := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "[]"}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, store)
	require.NotNil(t, extractor)

	ctx := context.Background()

	// 5 messages, less than batchSize+5=15
	history := []Message{
		{Role: "user", Content: "small_msg_0"},
		{Role: "assistant", Content: "small_msg_1"},
		{Role: "user", Content: "small_msg_2"},
		{Role: "assistant", Content: "small_msg_3"},
		{Role: "user", Content: "small_msg_4"},
	}

	for i := 1; i <= 10; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}

	mu.Lock()
	require.NoError(t, readErr)
	require.NotNil(t, capturedBody, "LLM should have been called")
	bodyStr := string(capturedBody)
	mu.Unlock()

	assert.Contains(t, bodyStr, "small_msg_0", "all messages should be included when history is small")
	assert.Contains(t, bodyStr, "small_msg_4", "all messages should be included when history is small")
}

// TestProcessResponse_TurnCountTracking tests turn count tracking across multiple turns.
// Verifies extraction happens at correct intervals for batch size 3.
func TestProcessResponse_TurnCountTracking(t *testing.T) {
	server, callCount := createTrackingMockLLMServer(t, "[]")
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	store := NewInMemoryStore()
	extractor := NewMemoryExtractorWithStore(routerCfg, 3, store)
	require.NotNil(t, extractor)

	ctx := context.Background()
	history := []Message{{Role: "user", Content: "test"}, {Role: "assistant", Content: "response"}}

	// Turns 1-2: no extraction
	for i := 1; i <= 2; i++ {
		err := extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(0), atomic.LoadInt32(callCount), "turns 1-2: no extraction")

	// Turn 3: extraction (3 % 3 == 0)
	err := extractor.ProcessResponse(ctx, "session1", "user1", history)
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "turn 3: extraction (3 % 3 == 0)")

	// Turns 4-5: no extraction
	for i := 4; i <= 5; i++ {
		err = extractor.ProcessResponse(ctx, "session1", "user1", history)
		require.NoError(t, err)
	}
	assert.Equal(t, int32(1), atomic.LoadInt32(callCount), "turns 4-5: no extraction")

	// Turn 6: extraction (6 % 3 == 0)
	err = extractor.ProcessResponse(ctx, "session1", "user1", history)
	require.NoError(t, err)
	assert.Equal(t, int32(2), atomic.LoadInt32(callCount), "turn 6: extraction (6 % 3 == 0)")
}

// TestProcessResponse_StoreDisabled tests behavior when store is disabled.
// Verifies ProcessResponse returns without error when store is not enabled.
func TestProcessResponse_StoreDisabled(t *testing.T) {
	// Create a disabled store
	disabledStore := &InMemoryStore{
		memories: make(map[string]*Memory),
		enabled:  false,
	}

	routerCfg := createMockRouterConfig("http://localhost:8080")
	extractor := NewMemoryExtractorWithStore(routerCfg, 10, disabledStore)
	require.NotNil(t, extractor)

	ctx := context.Background()
	history := []Message{{Role: "user", Content: "test"}}

	err := extractor.ProcessResponse(ctx, "session1", "user1", history)
	assert.NoError(t, err, "should return nil when store is disabled")
}

// TestProcessResponse_ExtractionDisabled tests behavior when extraction is disabled in config.
// Verifies NewMemoryExtractorWithStore returns nil when no extraction model is configured.
func TestProcessResponse_ExtractionDisabled(t *testing.T) {
	store := NewInMemoryStore()

	// With nil routerCfg (no extraction configured)
	extractor := NewMemoryExtractorWithStore(nil, 10, store)
	assert.Nil(t, extractor, "should return nil when extraction is disabled (nil config)")

	// With empty RouterConfig (no memory_extraction model role)
	extractor = NewMemoryExtractorWithStore(&config.RouterConfig{}, 10, store)
	assert.Nil(t, extractor, "should return nil when no memory_extraction model configured")
}

// TestProcessResponse_EmptyEndpoint tests behavior when endpoint is empty.
// Verifies ProcessResponse returns without error when endpoint is not configured.
func TestProcessResponse_EmptyEndpoint(t *testing.T) {
	store := NewInMemoryStore()

	// Create extractor directly with empty endpoint
	extractor := &MemoryExtractor{
		endpoint:   "",
		store:      store,
		turnCounts: make(map[string]int),
		batchSize:  10,
	}

	ctx := context.Background()
	history := []Message{{Role: "user", Content: "test"}}

	err := extractor.ProcessResponse(ctx, "session1", "user1", history)
	assert.NoError(t, err, "should return nil when endpoint is empty")
}

// TestProcessResponse_MultipleBatchSizes is a comprehensive test with multiple batch sizes.
// Validates extraction happens at correct turns for batch sizes 1, 3, 5, 10, 20.
func TestProcessResponse_MultipleBatchSizes(t *testing.T) {
	batchSizes := []int{1, 3, 5, 10, 20}

	for _, bs := range batchSizes {
		t.Run(fmt.Sprintf("batchSize_%d", bs), func(t *testing.T) {
			server, callCount := createTrackingMockLLMServer(t, "[]")
			defer server.Close()

			routerCfg := createMockRouterConfig(server.URL)
			store := NewInMemoryStore()
			extractor := NewMemoryExtractorWithStore(routerCfg, bs, store)
			require.NotNil(t, extractor)

			ctx := context.Background()
			history := []Message{{Role: "user", Content: "test"}, {Role: "assistant", Content: "response"}}

			// Call ProcessResponse for exactly batchSize turns
			for i := 1; i <= bs; i++ {
				err := extractor.ProcessResponse(ctx, "session1", "user1", history)
				require.NoError(t, err)
			}

			// Extraction should have happened exactly once at turn N
			assert.Equal(t, int32(1), atomic.LoadInt32(callCount),
				"extraction should happen exactly once at turn %d for batch size %d", bs, bs)
		})
	}
}

// =============================================================================
// Helper Functions
// =============================================================================

// createTrackingMockLLMServer creates a test server that returns the specified response
// and tracks the number of calls made to it via an atomic counter.
func createTrackingMockLLMServer(t *testing.T, factsJSON string) (*httptest.Server, *int32) {
	var callCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)

		response := map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]string{
						"content": factsJSON,
					},
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	return server, &callCount
}

// createMockLLMServer creates a test server that returns the specified response
func createMockLLMServer(t *testing.T, factsJSON string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		// Return mock response
		response := map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]string{
						"content": factsJSON,
					},
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
}
