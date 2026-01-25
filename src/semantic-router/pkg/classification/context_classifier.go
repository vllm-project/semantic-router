package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TokenCounter defines the interface for counting tokens in text
type TokenCounter interface {
	CountTokens(text string) (int, error)
}

const CharactersPerToken = 4

// CharacterBasedTokenCounter implements TokenCounter using a fast character-based heuristic.
// It estimates token count as: len(text) / CharactersPerToken
// This provides O(1) performance compared to full tokenization.
type CharacterBasedTokenCounter struct{}

// CountTokens estimates the number of tokens using the 1:4 character-to-token heuristic.
// This is a fast O(1) operation that avoids the overhead of full tokenization.
// The heuristic is based on OpenAI's guidance that 1 token ≈ 4 characters for English text.
func (c *CharacterBasedTokenCounter) CountTokens(text string) (int, error) {
	// len(text) returns byte count, which for UTF-8 may be higher than character count.
	// For mixed-language text, this provides a conservative (higher) estimate.
	byteLen := len(text)
	if byteLen == 0 {
		return 0, nil
	}
	// Integer division rounds down, add 1 to ensure we don't underestimate
	return (byteLen + CharactersPerToken - 1) / CharactersPerToken, nil
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
