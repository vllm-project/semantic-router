package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
		"What's up", "Whats up?",
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
		assert.Equal(t, 25, len(query25), "should be exactly 25 chars")
		assert.False(t, IsGreeting(query25), "passes length but not pattern")
	})

	t.Run("greeting at exactly 25 chars with padding", func(t *testing.T) {
		// "Hi" with spaces to make it 25 chars - trim should handle it
		query := "Hi                       " // Hi + 23 spaces = 25 chars
		assert.Equal(t, 25, len(query), "should be exactly 25 chars")
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
