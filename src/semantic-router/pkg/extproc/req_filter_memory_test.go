package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// =============================================================================
// ShouldSearchMemory Tests
// =============================================================================

func TestShouldSearchMemory_GeneralFactQuestions(t *testing.T) {
	tests := []struct {
		name  string
		query string
	}{
		{"general knowledge", "What is the capital of France?"},
		{"historical facts", "When was World War 2?"},
		{"scientific questions", "What is the speed of light?"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{FactCheckNeeded: true}
			result := ShouldSearchMemory(ctx, tt.query)
			assert.False(t, result, "should skip memory search for general fact queries")
		})
	}
}

func TestShouldSearchMemory_PersonalPronouns(t *testing.T) {
	tests := []struct {
		name  string
		query string
	}{
		{"my questions", "What is my budget?"},
		{"I questions", "What did I say about the project?"},
		{"me questions", "Tell me about my preferences"},
		{"mine questions", "Which project is mine?"},
		{"I'm contraction", "I'm planning a trip"},
		{"I've contraction", "I've told you my budget"},
		{"I'll contraction", "I'll need my preferences"},
		{"I'd contraction", "I'd like to know my status"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{FactCheckNeeded: true}
			result := ShouldSearchMemory(ctx, tt.query)
			assert.True(t, result, "should search memory for personal questions even with FactCheckNeeded")
		})
	}
}

func TestShouldSearchMemory_ToolQueries(t *testing.T) {
	t.Run("skip when tools available", func(t *testing.T) {
		ctx := &RequestContext{HasToolsForFactCheck: true}
		result := ShouldSearchMemory(ctx, "What's the weather today?")
		assert.False(t, result, "should skip memory search when tools are available")
	})

	t.Run("skip tool queries even with personal pronouns", func(t *testing.T) {
		ctx := &RequestContext{HasToolsForFactCheck: true}
		result := ShouldSearchMemory(ctx, "Search for my emails")
		assert.False(t, result, "should skip memory search for tool queries even with personal pronouns")
	})
}

func TestShouldSearchMemory_Greetings(t *testing.T) {
	greetings := []string{"Hi", "Hello there!", "Thanks"}

	for _, greeting := range greetings {
		t.Run(greeting, func(t *testing.T) {
			ctx := &RequestContext{}
			result := ShouldSearchMemory(ctx, greeting)
			assert.False(t, result, "should skip memory search for greetings")
		})
	}
}

func TestShouldSearchMemory_ShouldSearch(t *testing.T) {
	tests := []struct {
		name  string
		query string
	}{
		{"conversational questions", "What were we discussing?"},
		{"context-dependent questions", "Can you summarize what we talked about?"},
		{"follow-up questions", "And what about the deadline?"},
		{"vague questions needing context", "How much?"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{}
			result := ShouldSearchMemory(ctx, tt.query)
			assert.True(t, result, "should search memory for context-dependent queries")
		})
	}
}

func TestShouldSearchMemory_EdgeCases(t *testing.T) {
	ctx := &RequestContext{}

	t.Run("empty query", func(t *testing.T) {
		result := ShouldSearchMemory(ctx, "")
		assert.True(t, result, "empty query passes filters, let retrieval handle it")
	})

	t.Run("whitespace-only query", func(t *testing.T) {
		result := ShouldSearchMemory(ctx, "   ")
		assert.True(t, result, "whitespace passes filters")
	})

	t.Run("punctuation-only query", func(t *testing.T) {
		result := ShouldSearchMemory(ctx, "???")
		assert.True(t, result, "punctuation passes filters")
	})

	t.Run("very long query", func(t *testing.T) {
		longQuery := "This is a very long query that contains many words and should definitely trigger memory search because it likely contains important context about the conversation " +
			"and we want to make sure that the memory filter handles long queries correctly without any issues or performance problems"
		result := ShouldSearchMemory(ctx, longQuery)
		assert.True(t, result, "long queries should trigger memory search")
	})

	t.Run("unicode characters", func(t *testing.T) {
		result := ShouldSearchMemory(ctx, "מה התקציב שלי?") // Hebrew
		assert.True(t, result, "unicode queries should pass filters")
	})

	t.Run("emoji in query", func(t *testing.T) {
		result := ShouldSearchMemory(ctx, "What's my schedule? 📅")
		assert.True(t, result, "emoji queries with personal pronoun should search")
	})
}

func TestShouldSearchMemory_CombinedFlags(t *testing.T) {
	t.Run("both FactCheck and Tools true", func(t *testing.T) {
		ctx := &RequestContext{
			FactCheckNeeded:      true,
			HasToolsForFactCheck: true,
		}
		result := ShouldSearchMemory(ctx, "What is the weather?")
		assert.False(t, result, "should skip when both flags are true")
	})

	t.Run("FactCheck true, Tools true, with personal pronoun", func(t *testing.T) {
		ctx := &RequestContext{
			FactCheckNeeded:      true,
			HasToolsForFactCheck: true,
		}
		result := ShouldSearchMemory(ctx, "What's my weather forecast?")
		// Tools takes priority - even personal pronouns don't override tool queries
		assert.False(t, result, "tools should take priority over personal pronouns")
	})

	t.Run("all flags false", func(t *testing.T) {
		ctx := &RequestContext{
			FactCheckNeeded:      false,
			HasToolsForFactCheck: false,
		}
		result := ShouldSearchMemory(ctx, "Tell me about the project")
		assert.True(t, result, "should search when all flags false")
	})

	t.Run("only FactCheck false with personal pronoun", func(t *testing.T) {
		ctx := &RequestContext{
			FactCheckNeeded:      false,
			HasToolsForFactCheck: false,
		}
		result := ShouldSearchMemory(ctx, "What is my name?")
		assert.True(t, result, "should search with personal pronoun when FactCheck=false")
	})
}

func TestShouldSearchMemory_PriorityOrder(t *testing.T) {
	// Test that the priority order is: Tools > FactCheck > Greeting > Default

	t.Run("greeting with personal pronoun but greeting too short", func(t *testing.T) {
		ctx := &RequestContext{}
		// "Hi" is a greeting, but we're testing that personal pronouns in longer queries work
		result := ShouldSearchMemory(ctx, "Hi, what's my budget?")
		assert.True(t, result, "greeting with follow-up content should search")
	})

	t.Run("FactCheck query that looks like greeting", func(t *testing.T) {
		ctx := &RequestContext{FactCheckNeeded: true}
		result := ShouldSearchMemory(ctx, "What is hello in French?")
		assert.False(t, result, "fact check about greeting word should skip")
	})
}

// =============================================================================
// ContainsPersonalPronoun Tests
// =============================================================================

func TestContainsPersonalPronoun_WithPronouns(t *testing.T) {
	tests := []struct {
		name  string
		query string
	}{
		{"my", "What is my budget?"},
		{"I uppercase", "I need help"},
		{"i lowercase", "what did i say?"},
		{"me", "Tell me about it"},
		{"mine", "Is this mine?"},
		{"myself", "I did it myself"},
		{"I'm", "I'm going"},
		{"I've", "I've done it"},
		{"I'll", "I'll do it"},
		{"I'd", "I'd like to"},
		{"pronoun at end", "That belongs to me"},
		{"pronoun in middle", "Please tell me the time"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsPersonalPronoun(tt.query)
			assert.True(t, result, "should detect personal pronoun")
		})
	}
}

func TestContainsPersonalPronoun_WithoutPronouns(t *testing.T) {
	tests := []struct {
		name  string
		query string
	}{
		{"my in mythology", "What is mythology?"},
		{"I in AI", "What is AI?"},
		{"me in menu", "What is the menu?"},
		{"me in mechanism", "mechanism"},
		{"general question", "What is the capital of France?"},
		{"empty string", ""},
		{"third-person he", "He said that"},
		{"third-person she", "She wants it"},
		{"third-person they", "They are here"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsPersonalPronoun(tt.query)
			assert.False(t, result, "should not detect personal pronoun")
		})
	}
}

func TestContainsPersonalPronoun_MixedCases(t *testing.T) {
	// Note: "Tell me about Myanmar" contains "me" as a word
	assert.True(t, ContainsPersonalPronoun("Tell me about Myanmar"), "contains 'me' as word")
}

func TestContainsPersonalPronoun_CaseInsensitive(t *testing.T) {
	assert.True(t, ContainsPersonalPronoun("MY budget"))
	assert.True(t, ContainsPersonalPronoun("My budget"))
	assert.True(t, ContainsPersonalPronoun("my budget"))
	assert.True(t, ContainsPersonalPronoun("I think"))
	assert.True(t, ContainsPersonalPronoun("i think"))
}

func TestContainsPersonalPronoun_WordBoundaries(t *testing.T) {
	t.Run("pronoun at start of string", func(t *testing.T) {
		assert.True(t, ContainsPersonalPronoun("I am here"))
		assert.True(t, ContainsPersonalPronoun("My name is"))
		assert.True(t, ContainsPersonalPronoun("Me too"))
	})

	t.Run("pronoun at end of string", func(t *testing.T) {
		assert.True(t, ContainsPersonalPronoun("Give it to me"))
		assert.True(t, ContainsPersonalPronoun("That is mine"))
		assert.True(t, ContainsPersonalPronoun("said I"))
	})

	t.Run("pronoun as entire string", func(t *testing.T) {
		assert.True(t, ContainsPersonalPronoun("I"))
		assert.True(t, ContainsPersonalPronoun("me"))
		assert.True(t, ContainsPersonalPronoun("my"))
		assert.True(t, ContainsPersonalPronoun("mine"))
		assert.True(t, ContainsPersonalPronoun("myself"))
	})

	t.Run("multiple pronouns in query", func(t *testing.T) {
		assert.True(t, ContainsPersonalPronoun("Tell me about my project that I started"))
		assert.True(t, ContainsPersonalPronoun("I want my files that belong to me"))
	})
}

func TestContainsPersonalPronoun_SpecialCharacters(t *testing.T) {
	t.Run("with punctuation", func(t *testing.T) {
		assert.True(t, ContainsPersonalPronoun("What's my budget?"))
		assert.True(t, ContainsPersonalPronoun("I'm here!"))
		assert.True(t, ContainsPersonalPronoun("(my project)"))
	})

	t.Run("with quotes", func(t *testing.T) {
		assert.True(t, ContainsPersonalPronoun(`"my budget"`))
		assert.True(t, ContainsPersonalPronoun("'I said'"))
	})

	t.Run("with newlines", func(t *testing.T) {
		assert.True(t, ContainsPersonalPronoun("What is\nmy budget?"))
		assert.True(t, ContainsPersonalPronoun("I need\nhelp"))
	})
}

// =============================================================================
// IsGreeting Tests
// =============================================================================

func TestIsGreeting_SimpleGreetings(t *testing.T) {
	greetings := []string{
		"Hi", "hi", "HI",
		"Hello", "hello",
		"Hey", "hey!",
		"Howdy",
	}

	for _, g := range greetings {
		t.Run(g, func(t *testing.T) {
			assert.True(t, IsGreeting(g), "should detect simple greeting")
		})
	}
}

func TestIsGreeting_Variations(t *testing.T) {
	greetings := []string{
		"Hello there", "Hello there!",
		"Hi there",
		"Hi!", "Hello.", "Hey,",
		"  Hi  ", "Hello ",
	}

	for _, g := range greetings {
		t.Run(g, func(t *testing.T) {
			assert.True(t, IsGreeting(g), "should detect greeting variation")
		})
	}
}

func TestIsGreeting_TimeBased(t *testing.T) {
	greetings := []string{
		"Good morning", "good morning!",
		"Good afternoon",
		"Good evening",
		"Morning", "Evening!",
	}

	for _, g := range greetings {
		t.Run(g, func(t *testing.T) {
			assert.True(t, IsGreeting(g), "should detect time-based greeting")
		})
	}
}

func TestIsGreeting_Acknowledgments(t *testing.T) {
	greetings := []string{
		"Thanks", "thanks!",
		"Thank you",
		"Bye", "bye!",
		"Goodbye",
		"See you",
	}

	for _, g := range greetings {
		t.Run(g, func(t *testing.T) {
			assert.True(t, IsGreeting(g), "should detect acknowledgment")
		})
	}
}

func TestIsGreeting_ShortResponses(t *testing.T) {
	responses := []string{
		"Ok", "OK", "Okay",
		"Sure",
		"Yes", "No", "Yep", "Nope",
	}

	for _, r := range responses {
		t.Run(r, func(t *testing.T) {
			assert.True(t, IsGreeting(r), "should detect short response")
		})
	}
}

func TestIsGreeting_Informal(t *testing.T) {
	greetings := []string{
		"What's up", "What's up?",
		"Sup",
		"Yo",
	}

	for _, g := range greetings {
		t.Run(g, func(t *testing.T) {
			assert.True(t, IsGreeting(g), "should detect informal greeting")
		})
	}
}

func TestIsGreeting_NonGreetings(t *testing.T) {
	nonGreetings := []struct {
		name  string
		query string
	}{
		{"greeting with follow-up 1", "Hi, what's my budget?"},
		{"greeting with follow-up 2", "Hello, can you help me?"},
		{"greeting with follow-up 3", "Hey, I need assistance"},
		{"long query", "Hi there, I was wondering if you could help me with something important"},
		{"question 1", "What is the weather?"},
		{"question 2", "How are you?"},
		{"command 1", "Tell me a joke"},
		{"command 2", "Help me with this"},
		{"empty string", ""},
		{"partial greeting in text", "I said hi to everyone"},
	}

	for _, tt := range nonGreetings {
		t.Run(tt.name, func(t *testing.T) {
			assert.False(t, IsGreeting(tt.query), "should not detect as greeting")
		})
	}
}

func TestIsGreeting_LengthBoundary(t *testing.T) {
	// The IsGreeting function has a 25-character limit

	t.Run("short valid greeting", func(t *testing.T) {
		// "Good morning!" = 13 chars, well under limit
		assert.True(t, IsGreeting("Good morning!"), "short greeting should match")
	})

	t.Run("greeting with trailing spaces under limit", func(t *testing.T) {
		// Create a string with trailing spaces
		query := "Hello there!             " // with trailing spaces
		result := IsGreeting(query)
		assert.True(t, result, "greeting with trailing spaces should work")
	})

	t.Run("26 characters - should fail length check", func(t *testing.T) {
		// Create a 26-char string
		query := "abcdefghijklmnopqrstuvwxyz" // 26 chars
		assert.False(t, IsGreeting(query), "26 char query should fail length check")
	})

	t.Run("exactly 25 characters - passes length but not pattern", func(t *testing.T) {
		// 25 chars that don't match greeting pattern
		query25 := "1234567890123456789012345" // 25 chars of numbers
		assert.Len(t, query25, 25, "should be exactly 25 chars")
		assert.False(t, IsGreeting(query25), "passes length but not pattern")
	})

	t.Run("greeting at exactly 25 chars with padding", func(t *testing.T) {
		// "Hi" with spaces to make it 25 chars - trim should handle it
		query := "Hi                       " // Hi + 23 spaces = 25 chars
		assert.Len(t, query, 25, "should be exactly 25 chars")
		result := IsGreeting(query)
		assert.True(t, result, "Hi with padding should match after trim")
	})
}

func TestIsGreeting_Unicode(t *testing.T) {
	t.Run("non-ASCII characters", func(t *testing.T) {
		// These shouldn't match English greeting patterns
		assert.False(t, IsGreeting("שלום"), "Hebrew greeting shouldn't match")
		assert.False(t, IsGreeting("Bonjour"), "French greeting shouldn't match")
		assert.False(t, IsGreeting("Hola"), "Spanish greeting shouldn't match")
	})

	t.Run("emoji greetings", func(t *testing.T) {
		// Emojis alone shouldn't match
		assert.False(t, IsGreeting("👋"), "emoji wave shouldn't match")
		assert.False(t, IsGreeting("🙏"), "emoji pray shouldn't match")
	})
}

func TestIsGreeting_EdgePatterns(t *testing.T) {
	t.Run("greeting words in different context", func(t *testing.T) {
		assert.False(t, IsGreeting("The hello world program"), "hello in phrase")
		assert.False(t, IsGreeting("Say hi for me"), "hi in phrase")
		assert.False(t, IsGreeting("Thanks to everyone"), "thanks in phrase")
	})

	t.Run("multiple greeting words", func(t *testing.T) {
		// Short enough to pass length, but pattern may not match
		assert.False(t, IsGreeting("Hi hello hey"), "multiple greetings shouldn't match")
	})

	t.Run("greeting with numbers", func(t *testing.T) {
		assert.False(t, IsGreeting("Hi 123"), "greeting with numbers")
		assert.False(t, IsGreeting("Hello 2024"), "greeting with year")
	})
}

// =============================================================================
// BuildSearchQuery Tests
// =============================================================================

// createMockRouterConfig creates a RouterConfig with external_models pointing to the mock server.
// The serverURL should be from httptest.NewServer().URL
func createMockRouterConfig(serverURL string) *config.RouterConfig {
	// Parse the URL to extract host and port
	// httptest.Server.URL format: http://127.0.0.1:PORT
	var address string
	var port int
	if serverURL != "" {
		// Remove http:// prefix
		hostPort := serverURL[7:] // skip "http://"
		// Split by : to get host and port
		for i := len(hostPort) - 1; i >= 0; i-- {
			if hostPort[i] == ':' {
				address = hostPort[:i]
				portStr := hostPort[i+1:]
				// Parse port
				for _, c := range portStr {
					port = port*10 + int(c-'0')
				}
				break
			}
		}
	}

	return &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{
			{
				Provider:  "vllm",
				ModelRole: config.ModelRoleMemoryRewrite,
				ModelEndpoint: config.ClassifierVLLMEndpoint{
					Address: address,
					Port:    port,
				},
				ModelName:      "test-model",
				TimeoutSeconds: 5,
			},
		},
	}
}

func TestBuildSearchQuery_NoExternalModel(t *testing.T) {
	history := []ConversationMessage{
		{Role: "user", Content: "Planning vacation to Hawaii"},
	}

	// With nil routerCfg, should return original query
	result, err := BuildSearchQuery(context.Background(), history, "How much?", nil)
	require.NoError(t, err)
	assert.Equal(t, "How much?", result, "should return original when no routerCfg")

	// RouterConfig without external_models
	routerCfg := &config.RouterConfig{}
	result, err = BuildSearchQuery(context.Background(), nil, "test query", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "test query", result, "should return original when no external models")
}

func TestBuildSearchQuery_WithMockLLM(t *testing.T) {
	// Create mock LLM server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		// Parse request body
		var req llmChatRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, "test-model", req.Model)
		assert.Len(t, req.Messages, 2) // system + user
		assert.Equal(t, "system", req.Messages[0].Role)
		assert.Equal(t, "user", req.Messages[1].Role)
		assert.Contains(t, req.Messages[1].Content, "How much?")
		assert.Contains(t, req.Messages[1].Content, "Hawaii")

		// Return mock response
		resp := llmChatResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{Message: struct {
					Content string `json:"content"`
				}{Content: "What is the budget for the Hawaii vacation?"}},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	history := []ConversationMessage{
		{Role: "user", Content: "Planning vacation to Hawaii"},
		{Role: "assistant", Content: "Hawaii sounds great! What's your budget?"},
	}

	result, err := BuildSearchQuery(context.Background(), history, "How much?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is the budget for the Hawaii vacation?", result)
}

func TestBuildSearchQuery_SelfContainedQuery(t *testing.T) {
	// Create mock LLM server that returns the query unchanged
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := llmChatResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{Message: struct {
					Content string `json:"content"`
				}{Content: "What is the capital of France?"}}, // Unchanged
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	result, err := BuildSearchQuery(context.Background(), nil, "What is the capital of France?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is the capital of France?", result, "self-contained query should remain unchanged")
}

func TestBuildSearchQuery_LLMError_FallbackToOriginal(t *testing.T) {
	// Create mock LLM server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	result, err := BuildSearchQuery(context.Background(), nil, "original query", routerCfg)
	require.NoError(t, err) // Should not return error, just fallback
	assert.Equal(t, "original query", result, "should fallback to original on error")
}

func TestBuildSearchQuery_CleanupQuotes(t *testing.T) {
	// Create mock LLM server that returns quoted response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := llmChatResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{Message: struct {
					Content string `json:"content"`
				}{Content: `"What is my budget for Hawaii?"`}}, // With quotes
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	result, err := BuildSearchQuery(context.Background(), nil, "How much?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is my budget for Hawaii?", result, "should strip quotes")
}

func TestBuildSearchQuery_CleanupWhitespace(t *testing.T) {
	// Create mock LLM server that returns response with whitespace
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := llmChatResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{Message: struct {
					Content string `json:"content"`
				}{Content: "  What is the budget?  \n"}}, // With whitespace
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	result, err := BuildSearchQuery(context.Background(), nil, "How much?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is the budget?", result, "should trim whitespace")
}

func TestBuildSearchQuery_EmptyChoices(t *testing.T) {
	// Create mock LLM server that returns empty choices
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := llmChatResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{}, // Empty
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	result, err := BuildSearchQuery(context.Background(), nil, "original query", routerCfg)
	require.NoError(t, err) // Should fallback gracefully
	assert.Equal(t, "original query", result, "should fallback on empty choices")
}

// =============================================================================
// ExtractConversationHistory Tests
// =============================================================================

func TestExtractConversationHistory(t *testing.T) {
	messagesJSON := `[
		{"role": "system", "content": "You are a helpful assistant"},
		{"role": "user", "content": "Hello"},
		{"role": "assistant", "content": "Hi there!"},
		{"role": "user", "content": "What's the weather?"}
	]`

	history, err := ExtractConversationHistory([]byte(messagesJSON))
	require.NoError(t, err)

	// Should skip system message
	assert.Len(t, history, 3)
	assert.Equal(t, "user", history[0].Role)
	assert.Equal(t, "Hello", history[0].Content)
	assert.Equal(t, "assistant", history[1].Role)
	assert.Equal(t, "Hi there!", history[1].Content)
	assert.Equal(t, "user", history[2].Role)
	assert.Equal(t, "What's the weather?", history[2].Content)
}

func TestExtractConversationHistory_EmptyContent(t *testing.T) {
	messagesJSON := `[
		{"role": "user", "content": "Hello"},
		{"role": "assistant", "content": ""},
		{"role": "user", "content": "Test"}
	]`

	history, err := ExtractConversationHistory([]byte(messagesJSON))
	require.NoError(t, err)

	// Should skip empty content
	assert.Len(t, history, 2)
	assert.Equal(t, "Hello", history[0].Content)
	assert.Equal(t, "Test", history[1].Content)
}

func TestExtractConversationHistory_OnlySystemMessages(t *testing.T) {
	messagesJSON := `[
		{"role": "system", "content": "You are a helpful assistant"}
	]`

	history, err := ExtractConversationHistory([]byte(messagesJSON))
	require.NoError(t, err)
	assert.Empty(t, history, "should return empty for system-only messages")
}

func TestExtractConversationHistory_InvalidJSON(t *testing.T) {
	_, err := ExtractConversationHistory([]byte("invalid json"))
	assert.Error(t, err)
}

func TestExtractConversationHistory_EmptyArray(t *testing.T) {
	history, err := ExtractConversationHistory([]byte("[]"))
	require.NoError(t, err)
	assert.Empty(t, history)
}

func TestExtractConversationHistory_MissingRole(t *testing.T) {
	messagesJSON := `[
		{"content": "No role here"},
		{"role": "user", "content": "With role"}
	]`

	history, err := ExtractConversationHistory([]byte(messagesJSON))
	require.NoError(t, err)
	assert.Len(t, history, 1)
	assert.Equal(t, "With role", history[0].Content)
}

// =============================================================================
// FormatHistoryForPrompt Tests
// =============================================================================

func TestFormatHistoryForPrompt_Empty(t *testing.T) {
	result := formatHistoryForPrompt(nil)
	assert.Equal(t, "(no previous conversation)", result)
}

func TestFormatHistoryForPrompt_SingleMessage(t *testing.T) {
	history := []ConversationMessage{
		{Role: "user", Content: "Hello"},
	}

	result := formatHistoryForPrompt(history)
	assert.Equal(t, "[user]: Hello", result)
}

func TestFormatHistoryForPrompt_MultipleMessages(t *testing.T) {
	history := []ConversationMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
	}

	result := formatHistoryForPrompt(history)
	assert.Contains(t, result, "[user]: Hello")
	assert.Contains(t, result, "[assistant]: Hi there!")
}

func TestFormatHistoryForPrompt_LimitToLast5(t *testing.T) {
	history := []ConversationMessage{
		{Role: "user", Content: "Message 1"},
		{Role: "assistant", Content: "Response 1"},
		{Role: "user", Content: "Message 2"},
		{Role: "assistant", Content: "Response 2"},
		{Role: "user", Content: "Message 3"},
		{Role: "assistant", Content: "Response 3"},
		{Role: "user", Content: "Message 4"}, // Only last 5 should be included
	}

	result := formatHistoryForPrompt(history)

	// Should only include last 5 messages
	assert.NotContains(t, result, "Message 1")
	assert.NotContains(t, result, "Response 1")
	assert.Contains(t, result, "Message 2")
	assert.Contains(t, result, "Response 2")
	assert.Contains(t, result, "Message 3")
	assert.Contains(t, result, "Response 3")
	assert.Contains(t, result, "Message 4")
}

// =============================================================================
// TruncateForLog Tests
// =============================================================================

func TestTruncateForLog(t *testing.T) {
	assert.Equal(t, "short", truncateForLog("short", 10))
	assert.Equal(t, "this is a ...", truncateForLog("this is a long string", 10))
	assert.Empty(t, truncateForLog("", 10))
	assert.Equal(t, "exactly10!", truncateForLog("exactly10!", 10))
}

// =============================================================================
// InjectMemories Tests
// =============================================================================

func TestInjectMemories_NoMemories(t *testing.T) {
	originalRequest := []byte(`{"model":"test","messages":[{"role":"user","content":"Hello"}]}`)

	result, err := InjectMemories(originalRequest, nil)
	require.NoError(t, err)
	assert.Equal(t, originalRequest, result, "should return original when no memories")

	result, err = InjectMemories(originalRequest, []*memory.RetrieveResult{})
	require.NoError(t, err)
	assert.Equal(t, originalRequest, result, "should return original when empty memories")
}

func TestInjectMemories_SingleMemory(t *testing.T) {
	originalRequest := []byte(`{"model":"test","messages":[{"role":"user","content":"What's my budget?"}]}`)

	memories := []*memory.RetrieveResult{
		{
			Memory: &memory.Memory{Content: "Hawaii trip budget is $10,000"},
			Score:  0.85,
		},
	}

	result, err := InjectMemories(originalRequest, memories)
	require.NoError(t, err)

	// Parse result to verify
	var parsed map[string]interface{}
	err = json.Unmarshal(result, &parsed)
	require.NoError(t, err)

	messages, ok := parsed["messages"].([]interface{})
	require.True(t, ok)
	require.Len(t, messages, 2, "should have system + user message")

	// First message should be system with memory context
	firstMsg := messages[0].(map[string]interface{})
	assert.Equal(t, "system", firstMsg["role"])
	assert.Contains(t, firstMsg["content"], "User's Relevant Context")
	assert.Contains(t, firstMsg["content"], "Hawaii trip budget is $10,000")
}

func TestInjectMemories_MultipleMemories(t *testing.T) {
	originalRequest := []byte(`{"model":"test","messages":[{"role":"user","content":"Tell me about my trip"}]}`)

	memories := []*memory.RetrieveResult{
		{
			Memory: &memory.Memory{Content: "Hawaii trip budget is $10,000"},
			Score:  0.85,
		},
		{
			Memory: &memory.Memory{Content: "User prefers direct flights"},
			Score:  0.72,
		},
		{
			Memory: &memory.Memory{Content: "Trip planned for December 2025"},
			Score:  0.68,
		},
	}

	result, err := InjectMemories(originalRequest, memories)
	require.NoError(t, err)

	// Parse result to verify
	var parsed map[string]interface{}
	err = json.Unmarshal(result, &parsed)
	require.NoError(t, err)

	messages := parsed["messages"].([]interface{})
	firstMsg := messages[0].(map[string]interface{})
	content := firstMsg["content"].(string)

	assert.Contains(t, content, "Hawaii trip budget is $10,000")
	assert.Contains(t, content, "User prefers direct flights")
	assert.Contains(t, content, "Trip planned for December 2025")
}

func TestInjectMemories_ExistingSystemMessage(t *testing.T) {
	// Request already has a system message
	originalRequest := []byte(`{
		"model":"test",
		"messages":[
			{"role":"system","content":"You are a helpful assistant."},
			{"role":"user","content":"What's my budget?"}
		]
	}`)

	memories := []*memory.RetrieveResult{
		{
			Memory: &memory.Memory{Content: "Hawaii trip budget is $10,000"},
			Score:  0.85,
		},
	}

	result, err := InjectMemories(originalRequest, memories)
	require.NoError(t, err)

	// Parse result
	var parsed map[string]interface{}
	err = json.Unmarshal(result, &parsed)
	require.NoError(t, err)

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 2, "should still have 2 messages (context appended)")

	// System message should have both original content and memory
	systemMsg := messages[0].(map[string]interface{})
	content := systemMsg["content"].(string)
	assert.Contains(t, content, "You are a helpful assistant.")
	assert.Contains(t, content, "User's Relevant Context")
	assert.Contains(t, content, "Hawaii trip budget is $10,000")
}

func TestInjectMemories_InvalidJSON(t *testing.T) {
	invalidRequest := []byte(`not valid json`)

	memories := []*memory.RetrieveResult{
		{Memory: &memory.Memory{Content: "test memory"}, Score: 0.8},
	}

	// Should return original on error (graceful fallback)
	result, err := InjectMemories(invalidRequest, memories)
	require.NoError(t, err, "should not return error, just fallback")
	assert.Equal(t, invalidRequest, result, "should return original on parse error")
}

func TestInjectMemories_EmptyMessages(t *testing.T) {
	// Request with empty messages array
	originalRequest := []byte(`{"model":"test","messages":[]}`)

	memories := []*memory.RetrieveResult{
		{Memory: &memory.Memory{Content: "test memory"}, Score: 0.8},
	}

	result, err := InjectMemories(originalRequest, memories)
	require.NoError(t, err)

	var parsed map[string]interface{}
	err = json.Unmarshal(result, &parsed)
	require.NoError(t, err)

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 1, "should have system message added")

	systemMsg := messages[0].(map[string]interface{})
	assert.Equal(t, "system", systemMsg["role"])
	assert.Contains(t, systemMsg["content"], "test memory")
}

func TestInjectMemories_NilMemoryContent(t *testing.T) {
	originalRequest := []byte(`{"model":"test","messages":[{"role":"user","content":"test"}]}`)

	memories := []*memory.RetrieveResult{
		{Memory: nil, Score: 0.8},                         // nil memory
		{Memory: &memory.Memory{Content: ""}, Score: 0.7}, // empty content
		{Memory: &memory.Memory{Content: "valid memory"}, Score: 0.6},
	}

	result, err := InjectMemories(originalRequest, memories)
	require.NoError(t, err)

	var parsed map[string]interface{}
	err = json.Unmarshal(result, &parsed)
	require.NoError(t, err)

	messages := parsed["messages"].([]interface{})
	systemMsg := messages[0].(map[string]interface{})
	content := systemMsg["content"].(string)

	// Should only contain the valid memory
	assert.Contains(t, content, "valid memory")
	assert.NotContains(t, content, "nil") // No nil memory content
}

// =============================================================================
// FormatMemoriesAsContext Tests
// =============================================================================

func TestFormatMemoriesAsContext_Empty(t *testing.T) {
	result := FormatMemoriesAsContext(nil)
	assert.Empty(t, result)

	result = FormatMemoriesAsContext([]*memory.RetrieveResult{})
	assert.Empty(t, result)
}

func TestFormatMemoriesAsContext_SingleMemory(t *testing.T) {
	memories := []*memory.RetrieveResult{
		{Memory: &memory.Memory{Content: "Budget is $10,000"}, Score: 0.85},
	}

	result := FormatMemoriesAsContext(memories)

	assert.Contains(t, result, "## User's Relevant Context")
	assert.Contains(t, result, "- Budget is $10,000")
}

func TestFormatMemoriesAsContext_MultipleMemories(t *testing.T) {
	memories := []*memory.RetrieveResult{
		{Memory: &memory.Memory{Content: "Budget is $10,000"}, Score: 0.85},
		{Memory: &memory.Memory{Content: "Prefers direct flights"}, Score: 0.72},
	}

	result := FormatMemoriesAsContext(memories)

	assert.Contains(t, result, "## User's Relevant Context")
	assert.Contains(t, result, "- Budget is $10,000")
	assert.Contains(t, result, "- Prefers direct flights")
}

func TestFormatMemoriesAsContext_SkipsNilAndEmpty(t *testing.T) {
	memories := []*memory.RetrieveResult{
		{Memory: nil, Score: 0.9},
		{Memory: &memory.Memory{Content: ""}, Score: 0.8},
		{Memory: &memory.Memory{Content: "Valid content"}, Score: 0.7},
	}

	result := FormatMemoriesAsContext(memories)

	assert.Contains(t, result, "- Valid content")
	// Should only have one bullet point
	assert.Equal(t, 1, countOccurrences(result, "- "))
}

// =============================================================================
// Integration-like Tests
// =============================================================================

func TestInjectMemories_RealisticScenario(t *testing.T) {
	// Simulate a realistic request with system prompt, history, and user query
	originalRequest := []byte(`{
		"model": "qwen3-7b",
		"messages": [
			{"role": "system", "content": "You are a helpful travel assistant."},
			{"role": "user", "content": "I want to plan a trip to Hawaii"},
			{"role": "assistant", "content": "Great choice! Hawaii is beautiful. What's your budget?"},
			{"role": "user", "content": "How much did I say my budget was?"}
		],
		"temperature": 0.7,
		"max_tokens": 1024
	}`)

	memories := []*memory.RetrieveResult{
		{
			Memory: &memory.Memory{
				Content: "User's budget for Hawaii trip is $10,000",
				Type:    memory.MemoryTypeSemantic,
			},
			Score: 0.92,
		},
		{
			Memory: &memory.Memory{
				Content: "User prefers direct flights over connections",
				Type:    memory.MemoryTypeSemantic,
			},
			Score: 0.78,
		},
	}

	result, err := InjectMemories(originalRequest, memories)
	require.NoError(t, err)

	// Parse and validate
	var parsed map[string]interface{}
	err = json.Unmarshal(result, &parsed)
	require.NoError(t, err)

	// Verify model and other fields preserved
	assert.Equal(t, "qwen3-7b", parsed["model"])
	assert.Equal(t, float64(0.7), parsed["temperature"])
	assert.Equal(t, float64(1024), parsed["max_tokens"])

	// Verify messages
	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 4, "should have same number of messages")

	// System message should be enhanced
	systemMsg := messages[0].(map[string]interface{})
	content := systemMsg["content"].(string)
	assert.Contains(t, content, "You are a helpful travel assistant")
	assert.Contains(t, content, "User's Relevant Context")
	assert.Contains(t, content, "budget for Hawaii trip is $10,000")
	assert.Contains(t, content, "prefers direct flights")
}

func TestInjectMemories_PreservesMessageOrder(t *testing.T) {
	originalRequest := []byte(`{
		"model": "test",
		"messages": [
			{"role": "user", "content": "First message"},
			{"role": "assistant", "content": "Response"},
			{"role": "user", "content": "Second message"}
		]
	}`)

	memories := []*memory.RetrieveResult{
		{Memory: &memory.Memory{Content: "Memory content"}, Score: 0.8},
	}

	result, err := InjectMemories(originalRequest, memories)
	require.NoError(t, err)

	var parsed map[string]interface{}
	err = json.Unmarshal(result, &parsed)
	require.NoError(t, err)

	messages := parsed["messages"].([]interface{})
	require.Len(t, messages, 4, "should have system + 3 original messages")

	// Verify order: system, user, assistant, user
	assert.Equal(t, "system", messages[0].(map[string]interface{})["role"])
	assert.Equal(t, "user", messages[1].(map[string]interface{})["role"])
	assert.Equal(t, "assistant", messages[2].(map[string]interface{})["role"])
	assert.Equal(t, "user", messages[3].(map[string]interface{})["role"])

	// Verify content preserved
	assert.Equal(t, "First message", messages[1].(map[string]interface{})["content"])
	assert.Equal(t, "Response", messages[2].(map[string]interface{})["content"])
	assert.Equal(t, "Second message", messages[3].(map[string]interface{})["content"])
}

// Helper function
func countOccurrences(s, substr string) int {
	count := 0
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			count++
		}
	}
	return count
}
