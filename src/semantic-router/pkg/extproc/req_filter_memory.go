package extproc

import (
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Memory Decision Filter
//
// This filter decides whether a query should trigger a memory search.
// It reuses existing pipeline classification signals (FactCheckNeeded, HasToolsForFactCheck)
// to avoid redundant memory searches for queries that:
// - Are general fact questions (answered by LLM's knowledge)
// - Require tools (tool provides the answer)
// - Are simple greetings (no context needed)
//
// See: https://github.com/yehudit1987/semantic-router/issues/2

// personalPronounPattern matches personal pronouns that indicate user-specific context
// These override the fact-check signal for personal questions like "What is my budget?"
var personalPronounPattern = regexp.MustCompile(`(?i)\b(my|i|me|mine|i'm|i've|i'll|i'd|myself)\b`)

// greetingPatterns match standalone greetings that don't need memory context
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
//
// Decision logic:
//   - If FactCheckNeeded AND no personal pronouns → SKIP (general knowledge question)
//   - If HasToolsForFactCheck → SKIP (tool provides the answer)
//   - If isGreeting(query) → SKIP (no context needed)
//   - Otherwise → SEARCH MEMORY (conservative - don't miss context)
//
// The personal-indicator check overrides FactCheckNeeded because the fact-check
// classifier was designed for general-knowledge questions (e.g., "What is the capital of France?")
// and may incorrectly classify personal-fact questions ("What is my budget?") as fact queries.
func ShouldSearchMemory(ctx *RequestContext, query string) bool {
	// Check for personal indicators (overrides FactCheckNeeded for personal questions)
	hasPersonalIndicator := ContainsPersonalPronoun(query)

	// 1. Fact query → skip UNLESS it contains personal pronouns
	if ctx.FactCheckNeeded && !hasPersonalIndicator {
		logging.Debugf("Memory: Skipping - general fact query (FactCheckNeeded=%v, hasPersonalIndicator=%v)",
			ctx.FactCheckNeeded, hasPersonalIndicator)
		return false
	}

	// 2. Tool required → skip (tool provides answer)
	if ctx.HasToolsForFactCheck {
		logging.Debugf("Memory: Skipping - tool query (HasToolsForFactCheck=%v)", ctx.HasToolsForFactCheck)
		return false
	}

	// 3. Greeting/social → skip (no context needed)
	if IsGreeting(query) {
		logging.Debugf("Memory: Skipping - greeting detected")
		return false
	}

	// 4. Default: search memory (conservative - don't miss context)
	logging.Debugf("Memory: Will search - query passed all filters")
	return true
}

// ContainsPersonalPronoun checks if the query contains personal pronouns
// that indicate user-specific context (my, I, me, mine, etc.)
//
// Examples:
//   - "What is my budget?" → true
//   - "What is the capital of France?" → false
//   - "Tell me about my preferences" → true
//   - "I need help with my project" → true
func ContainsPersonalPronoun(query string) bool {
	return personalPronounPattern.MatchString(query)
}

// IsGreeting checks if the query is a standalone greeting that doesn't need
// memory context. Only matches short, simple greetings - not greetings
// followed by actual questions.
//
// Examples:
//   - "Hi" → true
//   - "Hello there!" → true
//   - "Good morning" → true
//   - "Thanks" → true
//   - "Hi, what's my budget?" → false (has content after greeting)
//   - "Hello, can you help me?" → false (has content after greeting)
func IsGreeting(query string) bool {
	// Trim and normalize
	trimmed := strings.TrimSpace(query)

	// Short greetings only (< 25 chars) - longer queries likely have actual content
	if len(trimmed) > 25 {
		return false
	}

	// Check against greeting patterns
	for _, pattern := range greetingPatterns {
		if pattern.MatchString(trimmed) {
			return true
		}
	}

	return false
}
