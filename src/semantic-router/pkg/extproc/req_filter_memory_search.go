package extproc

import (
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// personalPronounPattern matches personal pronouns that indicate user-specific context.
// These override the fact-check signal for personal questions like "What is my budget?"
var personalPronounPattern = regexp.MustCompile(`(?i)\b(my|i|me|mine|i'm|i've|i'll|i'd|myself)\b`)

// greetingPatterns match standalone greetings that don't need memory context.
var greetingPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^(hi|hello|hey|howdy)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(hi|hello|hey)[\s\,]*(there)?[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(good\s+)?(morning|afternoon|evening|night)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(thanks|thank\s+you|thx)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(bye|goodbye|see\s+you|later)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(ok|okay|sure|yes|no|yep|nope)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(what'?s?\s+up|sup|yo)[\s\!\?\.\,]*$`),
}

// ShouldSearchMemory decides if a query should trigger memory search.
// It reuses existing pipeline classification signals with a personal-fact override.
func ShouldSearchMemory(ctx *RequestContext, query string) bool {
	hasPersonalIndicator := ContainsPersonalPronoun(query)

	if ctx.FactCheckNeeded && !hasPersonalIndicator {
		logging.Debugf("Memory: Skipping - general fact query (FactCheckNeeded=%v, hasPersonalIndicator=%v)",
			ctx.FactCheckNeeded, hasPersonalIndicator)
		return false
	}

	if ctx.HasToolsForFactCheck {
		logging.Debugf("Memory: Skipping - tool query (HasToolsForFactCheck=%v)", ctx.HasToolsForFactCheck)
		return false
	}

	if IsGreeting(query) {
		logging.Debugf("Memory: Skipping - greeting detected")
		return false
	}

	logging.Debugf("Memory: Will search - query passed all filters")
	return true
}

// ContainsPersonalPronoun checks if the query contains personal pronouns
// that indicate user-specific context (my, I, me, mine, etc.).
func ContainsPersonalPronoun(query string) bool {
	return personalPronounPattern.MatchString(query)
}

// IsGreeting checks if the query is a standalone greeting that doesn't need
// memory context. Only matches short, simple greetings, not greetings followed
// by actual questions.
func IsGreeting(query string) bool {
	trimmed := strings.TrimSpace(query)

	if len(trimmed) > 25 {
		return false
	}

	for _, pattern := range greetingPatterns {
		if pattern.MatchString(trimmed) {
			return true
		}
	}

	return false
}
