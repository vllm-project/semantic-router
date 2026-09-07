package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

// compiledContextRule holds a context rule's parsed band so the hot path does
// no string parsing. Rules whose limits fail to parse are kept with ok=false so
// their names stay visible in diagnostics but never match.
type compiledContextRule struct {
	name   string
	bounds config.ContextBounds
	ok     bool
}

// ContextClassifier classifies text based on token count rules
type ContextClassifier struct {
	tokenCounter TokenCounter
	rules        []compiledContextRule
}

// NewContextClassifier creates a new ContextClassifier. Rules with invalid
// token limits are reported once here and are skipped during classification;
// config validation normally rejects them before this point.
func NewContextClassifier(tokenCounter TokenCounter, rules []config.ContextRule) *ContextClassifier {
	compiled := make([]compiledContextRule, 0, len(rules))
	for _, rule := range rules {
		bounds, err := rule.Bounds()
		if err != nil {
			logging.Warnf("context rule %q has an invalid token range and will never match: %v", rule.Name, err)
			compiled = append(compiled, compiledContextRule{name: rule.Name})
			continue
		}
		compiled = append(compiled, compiledContextRule{name: rule.Name, bounds: bounds, ok: true})
	}
	return &ContextClassifier{
		tokenCounter: tokenCounter,
		rules:        compiled,
	}
}

// Classify determines which context rules match the given text's token count
// Returns matched rule names, the actual token count, and any error
func (c *ContextClassifier) Classify(text string) ([]string, int, error) {
	return c.ClassifyWithTokenFloor(text, 0)
}

// ClassifyWithTokenFloor applies the larger of the calibrated text estimate
// and a conservative request-envelope floor. The floor lets routing account
// for prompt-bearing components that must not be copied into semantic signal
// text (tool schemas/results and image payloads, for example).
//
// Bands are inclusive on both ends. Every matching rule is returned, in
// configuration order, so overlapping bands report all of their names.
// An open-ended band (no max_tokens) matches any count at or above its
// minimum, which is how overflow above the largest bounded band is routed.
func (c *ContextClassifier) ClassifyWithTokenFloor(text string, tokenFloor int) ([]string, int, error) {
	tokenCount, err := c.tokenCounter.CountTokens(text)
	if err != nil {
		return nil, 0, err
	}
	if tokenFloor > tokenCount {
		tokenCount = tokenFloor
	}

	var matchedRules []string
	for _, rule := range c.rules {
		if !rule.ok || !rule.bounds.Matches(tokenCount) {
			continue
		}
		if rule.bounds.Unbounded {
			logging.Debugf("[Signal Computation] context rule %q matched %d tokens via its open-ended band (min=%d)",
				rule.name, tokenCount, rule.bounds.Min)
		}
		matchedRules = append(matchedRules, rule.name)
	}

	return matchedRules, tokenCount, nil
}
