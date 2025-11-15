package candle_binding

import (
	"strings"
	"testing"
	"time"
)

func TestNewRegexProvider(t *testing.T) {
	t.Run("ValidConfig", func(t *testing.T) {
		cfg := RegexProviderConfig{
			MaxPatterns:      10,
			MaxPatternLength: 100,
			MaxInputLength:   1000,
			DefaultTimeoutMs: 50,
			Patterns: []RegexPattern{
				{ID: "email", Pattern: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`},
			},
		}
		_, err := NewRegexProvider(cfg)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
	})

	t.Run("TooManyPatterns", func(t *testing.T) {
		cfg := RegexProviderConfig{
			MaxPatterns: 1,
			Patterns: []RegexPattern{
				{ID: "p1", Pattern: "a"},
				{ID: "p2", Pattern: "b"},
			},
		}
		_, err := NewRegexProvider(cfg)
		if err == nil {
			t.Fatal("expected an error for too many patterns, got nil")
		}
	})

	t.Run("PatternTooLong", func(t *testing.T) {
		cfg := RegexProviderConfig{
			MaxPatterns:      10,
			MaxPatternLength: 5,
			Patterns: []RegexPattern{
				{ID: "long", Pattern: "abcdef"},
			},
		}
		_, err := NewRegexProvider(cfg)
		if err == nil {
			t.Fatal("expected an error for pattern too long, got nil")
		}
	})

	t.Run("InvalidRegex", func(t *testing.T) {
		cfg := RegexProviderConfig{
			MaxPatterns:      10,
			MaxPatternLength: 100,
			Patterns: []RegexPattern{
				{ID: "invalid", Pattern: `[`},
			},
		}
		_, err := NewRegexProvider(cfg)
		if err == nil {
			t.Fatal("expected an error for invalid regex, got nil")
		}
	})
}

func TestRegexProvider_Scan(t *testing.T) {
	cfg := RegexProviderConfig{
		MaxPatterns:      10,
		MaxPatternLength: 100,
		MaxInputLength:   1000,
		DefaultTimeoutMs: 100,
		Patterns: []RegexPattern{
			{ID: "email", Pattern: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`, Category: "pii"},
			{ID: "word", Pattern: "hello", Category: "greeting"},
			{ID: "case", Pattern: "World", Flags: "i", Category: "case-test"},
		},
	}
	rp, err := NewRegexProvider(cfg)
	if err != nil {
		t.Fatalf("failed to create regex provider: %v", err)
	}

	t.Run("SimpleMatch", func(t *testing.T) {
		input := "say hello to the world"
		matches, err := rp.Scan(input)
		if err != nil {
			t.Fatalf("scan failed: %v", err)
		}
		if len(matches) != 2 { // "hello" and "world" (case-insensitive)
			t.Fatalf("expected 2 matches, got %d", len(matches))
		}
	})

	t.Run("CaseInsensitiveMatch", func(t *testing.T) {
		input := "hello WORLD"
		matches, err := rp.Scan(input)
		if err != nil {
			t.Fatalf("scan failed: %v", err)
		}
		if len(matches) != 2 {
			t.Fatalf("expected 2 matches, got %d", len(matches))
		}
	})

	t.Run("MultipleMatches", func(t *testing.T) {
		input := "my email is test@example.com, say hello"
		matches, err := rp.Scan(input)
		if err != nil {
			t.Fatalf("scan failed: %v", err)
		}
		if len(matches) != 2 {
			t.Fatalf("expected 2 matches, got %d", len(matches))
		}
	})

	t.Run("NoMatch", func(t *testing.T) {
		input := "nothing to see here"
		matches, err := rp.Scan(input)
		if err != nil {
			t.Fatalf("scan failed: %v", err)
		}
		if len(matches) != 0 {
			t.Fatalf("expected 0 matches, got %d", len(matches))
		}
	})

	t.Run("InputTooLong", func(t *testing.T) {
		rp.maxInputLength = 5
		_, err := rp.Scan("abcdef")
		if err == nil {
			t.Fatal("expected an error for input too long, got nil")
		}
		rp.maxInputLength = 1000 // reset
	})

	t.Run("Timeout", func(t *testing.T) {
		cfg := RegexProviderConfig{
			MaxPatterns:      1,
			MaxPatternLength: 100,
			MaxInputLength:   1000,
			DefaultTimeoutMs: 10, // 10ms
			Patterns: []RegexPattern{
				{ID: "any", Pattern: `.`},
			},
		}
		// Create a provider with a 20ms delay, which is longer than the timeout
		rp, err := NewRegexProvider(cfg, WithTestDelay(20*time.Millisecond))
		if err != nil {
			t.Fatalf("failed to create regex provider: %v", err)
		}

		_, err = rp.Scan("a")
		if err == nil {
			t.Fatal("expected a timeout error, got nil")
		}
		if !strings.Contains(err.Error(), "timed out") {
			t.Errorf("expected timeout error, got: %v", err)
		}
	})

	t.Run("ReDoSAttackVector", func(t *testing.T) {
		// This pattern is a known ReDoS vector for backtracking regex engines.
		// Go's engine is not vulnerable, so this should execute quickly.
		cfg := RegexProviderConfig{
			MaxPatterns:      1,
			MaxPatternLength: 100,
			MaxInputLength:   1000,
			DefaultTimeoutMs: 500, // 500ms timeout
			Patterns: []RegexPattern{
				{ID: "redos", Pattern: `(a+)+$`},
			},
		}
		rp, err := NewRegexProvider(cfg)
		if err != nil {
			t.Fatalf("failed to create regex provider: %v", err)
		}

		// A long string of 'a's followed by a non-matching character.
		// In a vulnerable engine, this would cause catastrophic backtracking.
		input := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"

		_, err = rp.Scan(input)
		if err != nil {
			t.Fatalf("scan failed for ReDoS pattern: %v", err)
		}
	})
}
