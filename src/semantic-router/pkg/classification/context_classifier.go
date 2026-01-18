package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TokenCounter defines the interface for counting tokens in text
type TokenCounter interface {
	CountTokens(text string) (int, error)
}

// CandleTokenCounter implements TokenCounter using the Candle binding tokenizer
type CandleTokenCounter struct{}

// CountTokens counts the number of tokens in the given text using the initialized Candle model
func (c *CandleTokenCounter) CountTokens(text string) (int, error) {
	// Use a large max length to ensure we count all tokens (up to 1M)
	// This avoids truncation for most practical context sizes
	const MaxTokenLimit = 1000000
	result, err := candle_binding.TokenizeText(text, MaxTokenLimit)
	if err != nil {
		return 0, fmt.Errorf("failed to count tokens: %w", err)
	}
	return len(result.TokenIDs), nil
}

// ContextClassifier classifies text based on token count rules
type ContextClassifier struct {
	tokenCounter TokenCounter
	rules        []config.ContextRule
}

// NewContextClassifier creates a new ContextClassifier
func NewContextClassifier(tokenCounter TokenCounter, rules []config.ContextRule) *ContextClassifier {
	return &ContextClassifier{
		tokenCounter: tokenCounter,
		rules:        rules,
	}
}

// Classify determines which context rules match the given text's token count
// Returns matched rule names, the actual token count, and any error
func (c *ContextClassifier) Classify(text string) ([]string, int, error) {
	tokenCount, err := c.tokenCounter.CountTokens(text)
	if err != nil {
		return nil, 0, err
	}

	var matchedRules []string
	for _, rule := range c.rules {
		min, err := rule.MinTokens.Value()
		if err != nil {
			// Skip rules with invalid token counts, log warning in real app
			continue
		}
		max, err := rule.MaxTokens.Value()
		if err != nil {
			continue
		}

		if tokenCount >= min && tokenCount <= max {
			matchedRules = append(matchedRules, rule.Name)
		}
	}

	return matchedRules, tokenCount, nil
}
