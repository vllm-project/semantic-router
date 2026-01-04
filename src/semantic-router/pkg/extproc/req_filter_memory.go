package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// =============================================================================
// Memory Decision Filter
// =============================================================================

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

// =============================================================================
// Query Rewriting for Memory Search
// =============================================================================

// Query Rewriting for Memory Search
//
// This module rewrites vague/context-dependent queries using an LLM to produce
// self-contained queries suitable for semantic search in a memory database.
//
// Example:
//   History: ["Planning vacation to Hawaii"]
//   Query: "How much?"
//   Rewritten: "What is the budget for the Hawaii vacation?"

// ConversationMessage represents a message in conversation history
type ConversationMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// getTimeout returns the timeout duration from config
func getTimeout(cfg *config.QueryRewriteConfig) time.Duration {
	if cfg.TimeoutSeconds > 0 {
		return time.Duration(cfg.TimeoutSeconds) * time.Second
	}
	return 5 * time.Second // default
}

// getMaxTokens returns max tokens with default
func getMaxTokens(cfg *config.QueryRewriteConfig) int {
	if cfg.MaxTokens > 0 {
		return cfg.MaxTokens
	}
	return 50 // default
}

// getTemperature returns temperature with default
func getTemperature(cfg *config.QueryRewriteConfig) float64 {
	if cfg.Temperature > 0 {
		return cfg.Temperature
	}
	return 0.1 // default
}

// queryRewriteSystemPrompt is the system prompt for query rewriting
const queryRewriteSystemPrompt = `You are a query rewriter for semantic search in a memory database.

Given conversation history and a user query, rewrite the query to be self-contained 
for searching memories. Include relevant context from history if the query references 
previous conversation.

Rules:
- Keep the rewritten query concise (under 50 words)
- Preserve the user's intent exactly
- Include specific entities/topics from history if referenced (e.g., "it", "that", "the second one")
- If the query is already self-contained, return it unchanged
- Return ONLY the rewritten query, no explanation or quotes`

// BuildSearchQuery rewrites a query with conversation context for semantic search.
// It uses an LLM to understand context and produce a self-contained query.
//
// Example:
//
//	History: ["Planning vacation to Hawaii"]
//	Query: "How much?"
//	Result: "What is the budget for the Hawaii vacation?"
func BuildSearchQuery(ctx context.Context, history []ConversationMessage, query string, cfg *config.QueryRewriteConfig) (string, error) {
	if cfg == nil || !cfg.Enabled || cfg.Endpoint == "" {
		logging.Debugf("Memory: Query rewriting disabled, using original query")
		return query, nil
	}

	// Format history for the prompt
	historyText := formatHistoryForPrompt(history)

	// Build user prompt
	userPrompt := fmt.Sprintf("History:\n%s\n\nQuery: %s\n\nRewritten query:", historyText, query)

	// TODO: Remove debug logs after POC demo
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║                    MEMORY QUERY REWRITING                        ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ HISTORY:                                                         ║")
	for _, msg := range history {
		logging.Infof("║   [%s]: %s", msg.Role, truncateForLog(msg.Content, 50))
	}
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ ORIGINAL QUERY: %s", query)
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	// Call LLM for rewriting
	rewrittenQuery, err := callLLMForQueryRewrite(ctx, cfg, userPrompt)
	if err != nil {
		logging.Errorf("Memory: Query rewriting failed, using original: %v", err)
		// Fallback to original query on error
		return query, nil
	}

	// Clean up the response
	rewrittenQuery = strings.TrimSpace(rewrittenQuery)
	rewrittenQuery = strings.Trim(rewrittenQuery, "\"'")

	// TODO: Remove debug logs after POC demo
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║ REWRITTEN QUERY: %s", rewrittenQuery)
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	return rewrittenQuery, nil
}

// formatHistoryForPrompt formats conversation history for the LLM prompt
func formatHistoryForPrompt(history []ConversationMessage) string {
	if len(history) == 0 {
		return "(no previous conversation)"
	}

	var lines []string
	// Only use last 5 messages to keep context manageable
	startIdx := 0
	if len(history) > 5 {
		startIdx = len(history) - 5
	}

	for _, msg := range history[startIdx:] {
		lines = append(lines, fmt.Sprintf("[%s]: %s", msg.Role, msg.Content))
	}

	return strings.Join(lines, "\n")
}

// truncateForLog truncates a string for logging purposes
func truncateForLog(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// =============================================================================
// LLM Client for Query Rewriting
// =============================================================================

// llmChatRequest represents an OpenAI-compatible chat request
type llmChatRequest struct {
	Model       string           `json:"model"`
	Messages    []llmChatMessage `json:"messages"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature float64          `json:"temperature,omitempty"`
	Stream      bool             `json:"stream"`
}

type llmChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// llmChatResponse represents an OpenAI-compatible chat response
type llmChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// callLLMForQueryRewrite calls the LLM endpoint for query rewriting
func callLLMForQueryRewrite(ctx context.Context, cfg *config.QueryRewriteConfig, userPrompt string) (string, error) {
	// Build request
	reqBody := llmChatRequest{
		Model: cfg.Model,
		Messages: []llmChatMessage{
			{Role: "system", Content: queryRewriteSystemPrompt},
			{Role: "user", Content: userPrompt},
		},
		MaxTokens:   getMaxTokens(cfg),
		Temperature: getTemperature(cfg),
		Stream:      false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimSuffix(cfg.Endpoint, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Send request
	client := &http.Client{Timeout: getTimeout(cfg)}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("LLM request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("LLM returned status %d", resp.StatusCode)
	}

	// Parse response
	var llmResp llmChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if len(llmResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in LLM response")
	}

	return llmResp.Choices[0].Message.Content, nil
}

// =============================================================================
// Helper: Extract History from OpenAI Messages
// =============================================================================

// ExtractConversationHistory extracts conversation history from raw request messages.
// It filters out system messages and returns user/assistant messages for context.
func ExtractConversationHistory(messagesJSON []byte) ([]ConversationMessage, error) {
	var messages []map[string]interface{}
	if err := json.Unmarshal(messagesJSON, &messages); err != nil {
		return nil, fmt.Errorf("failed to parse messages: %w", err)
	}

	var history []ConversationMessage
	for _, msg := range messages {
		role, ok := msg["role"].(string)
		// Skip if no role or system messages (not conversation history)
		if !ok || role == "system" {
			continue
		}

		// Extract content - skip empty
		content, ok := msg["content"].(string)
		if !ok || content == "" {
			continue
		}

		history = append(history, ConversationMessage{
			Role:    role,
			Content: content,
		})
	}

	return history, nil
}
