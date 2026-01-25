package memory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
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
// Helper Functions
// =============================================================================

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
