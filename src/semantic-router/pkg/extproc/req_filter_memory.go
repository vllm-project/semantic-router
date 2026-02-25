package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
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

// getMaxTokens returns max tokens from resolved config with default
func getMaxTokens(resolved *ResolvedLLMConfig, defaultValue int) int {
	if resolved != nil && resolved.MaxTokens > 0 {
		return resolved.MaxTokens
	}
	return defaultValue
}

// getTemperature returns temperature from resolved config with default
func getTemperature(resolved *ResolvedLLMConfig, defaultValue float64) float64 {
	if resolved != nil && resolved.Temperature > 0 {
		return resolved.Temperature
	}
	return defaultValue
}

// getTimeout returns timeout from resolved config (from external_models), or default 5s
func getTimeout(resolved *ResolvedLLMConfig) time.Duration {
	if resolved != nil && resolved.TimeoutSeconds > 0 {
		return time.Duration(resolved.TimeoutSeconds) * time.Second
	}
	return 5 * time.Second
}

// queryRewriteSystemPrompt is the system prompt for query rewriting
const queryRewriteSystemPrompt = `You are a query rewriter for semantic search in a memory database.

Given conversation history and a user query, rewrite the query to be self-contained
for searching memories. Include relevant context from history if the query references
previous conversation.

Do NOT use <think> tags or show your reasoning. Output ONLY the rewritten query.

CRITICAL RULES:
- PRESERVE THE QUERY TYPE: If the user is stating a fact, keep it as a statement. If asking a question, keep it as a question. NEVER convert statements to questions!
- Use ONLY facts explicitly stated in the history. NEVER invent or hallucinate values!
- If history says "$10,000" - use "$10,000" (not $1,000,000)
- Keep the rewritten query concise (under 50 words)
- Preserve the user's intent exactly
- Replace vague references (e.g., "it", "that", "my budget") with specific context from history
- Include CONSTRAINTS when relevant (cannot use X, must use Y, excluded, limitations)
- For tech/deployment queries, include any mentioned technologies or platforms
- If the query is already self-contained, return it unchanged
- Return ONLY the rewritten query, no explanation or quotes

EXAMPLES:
History: [user]: My project budget is $50,000 and deadline is March 15th
Query: When is the deadline?
Rewritten: When is the deadline for my $50,000 project?

History: [user]: I'm building an e-commerce platform
Query: I prefer React for frontend and Go for backend
Rewritten: I prefer React for frontend and Go for backend for my e-commerce platform

History: [user]: I prefer React for frontend and Go for backend
Query: What tech should I use?
Rewritten: What tech stack should I use considering my preference for React frontend and Go backend?

History: [user]: We cannot use AWS, must deploy on Azure
Query: Where can I deploy?
Rewritten: Where can I deploy my project given I cannot use AWS and must use Azure?

History: [user]: Building an e-commerce platform with PostgreSQL database
Query: What database?
Rewritten: What database should I use for my e-commerce platform using PostgreSQL?

History: (no relevant context)
Query: What is my budget?
Rewritten: What is my project budget?`

// BuildSearchQuery rewrites a query with conversation context for semantic search.
// It uses an LLM to understand context and produce a self-contained query.
//
// Example:
//
//	History: ["Planning vacation to Hawaii"]
//	Query: "How much?"
//	Result: "What is the budget for the Hawaii vacation?"
func BuildSearchQuery(ctx context.Context, history []ConversationMessage, query string, routerCfg *config.RouterConfig) (string, error) {
	// Query rewriting is enabled if external model with role "memory_rewrite" exists
	resolved := ResolveQueryRewriteConfig(routerCfg)
	if resolved == nil || resolved.Endpoint == "" {
		logging.Debugf("Memory: Query rewriting not configured, using original query")
		return query, nil
	}

	// Format history for the prompt
	historyText := formatHistoryForPrompt(history)

	// Build user prompt
	userPrompt := fmt.Sprintf("History:\n%s\n\nQuery: %s\n\nRewritten query:", historyText, query)

	logging.Debugf("Memory: query rewrite: original=%q, history_len=%d", truncateForLog(query, 80), len(history))

	// Call LLM for rewriting
	rewrittenQuery, err := callLLMForQueryRewrite(ctx, resolved, userPrompt)
	if err != nil {
		logging.Errorf("Memory: Query rewriting failed, using original: %v", err)
		// Fallback to original query on error
		return query, nil
	}

	// Clean up the response
	rewrittenQuery = strings.TrimSpace(rewrittenQuery)
	rewrittenQuery = strings.Trim(rewrittenQuery, "\"'")

	logging.Debugf("Memory: query rewrite: result=%q", truncateForLog(rewrittenQuery, 80))

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
// LLM Client for Query Rewriting (uses openai-go typed params)
// =============================================================================

// ResolvedLLMConfig holds resolved LLM endpoint configuration from external_models.
type ResolvedLLMConfig struct {
	Endpoint       string
	Model          string
	AccessKey      string
	TimeoutSeconds int
	MaxTokens      int
	Temperature    float64
}

// ResolveQueryRewriteConfig resolves the LLM endpoint configuration for query rewriting.
// Looks up the external model by ModelRole (defaults to "memory_rewrite").
func ResolveQueryRewriteConfig(routerCfg *config.RouterConfig) *ResolvedLLMConfig {
	if routerCfg == nil {
		return nil
	}

	externalCfg := routerCfg.FindExternalModelByRole(config.ModelRoleMemoryRewrite)
	if externalCfg == nil || externalCfg.ModelEndpoint.Address == "" {
		return nil
	}

	endpoint := fmt.Sprintf("http://%s:%d", externalCfg.ModelEndpoint.Address, externalCfg.ModelEndpoint.Port)
	return &ResolvedLLMConfig{
		Endpoint:       endpoint,
		Model:          externalCfg.ModelName,
		AccessKey:      externalCfg.AccessKey,
		TimeoutSeconds: externalCfg.TimeoutSeconds,
		MaxTokens:      externalCfg.MaxTokens,
		Temperature:    externalCfg.Temperature,
	}
}

// callLLMForQueryRewrite calls the LLM endpoint for query rewriting.
// Uses openai-go typed params. No response_format since rewrite returns plain text.
func callLLMForQueryRewrite(ctx context.Context, resolved *ResolvedLLMConfig, userPrompt string) (string, error) {
	params := openai.ChatCompletionNewParams{
		Model: resolved.Model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(queryRewriteSystemPrompt),
				},
			}},
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(userPrompt),
				},
			}},
		},
		MaxTokens:   openai.Int(int64(getMaxTokens(resolved, 512))),
		Temperature: openai.Float(getTemperature(resolved, 0.1)),
	}

	jsonData, err := json.Marshal(params)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Explicitly set stream=false (SDK omits it)
	jsonData, err = setJSONField(jsonData, "stream", false)
	if err != nil {
		return "", fmt.Errorf("failed to set stream param: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimSuffix(resolved.Endpoint, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if resolved.AccessKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+resolved.AccessKey)
	}

	timeout := getTimeout(resolved)
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("LLM request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("LLM returned status %d: %s", resp.StatusCode, truncateForLog(string(body), 200))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	// Parse with openai-go typed response — vLLM separates reasoning
	// into reasoning_content field, so message.content is clean.
	var completion openai.ChatCompletion
	if err := json.Unmarshal(body, &completion); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("no choices in LLM response")
	}

	// Safety net for non-vLLM backends
	content := memory.StripThinkTags(completion.Choices[0].Message.Content)
	return content, nil
}

// setJSONField sets a field on a marshalled JSON object.
func setJSONField(data []byte, key string, value interface{}) ([]byte, error) {
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	m[key] = value
	return json.Marshal(m)
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

// =============================================================================
// Memory Injection into LLM Request
// =============================================================================

// FormatMemoriesAsContext formats retrieved memories as a context block
// for injection into the LLM request.
//
// Output format:
//
//	## User's Relevant Context
//
//	- Hawaii trip budget is $10,000 (relevance: 0.85)
//	- User prefers direct flights (relevance: 0.72)
func FormatMemoriesAsContext(memories []*memory.RetrieveResult) string {
	if len(memories) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("The following is relevant context from previous conversations with the user:\n\n")

	for _, result := range memories {
		if result.Memory != nil && result.Memory.Content != "" {
			sb.WriteString(fmt.Sprintf("- %s\n", result.Memory.Content))
		}
	}

	sb.WriteString("\nUse this context to personalize your response when relevant. Do not repeat it verbatim unless asked.")

	return sb.String()
}

// injectMemoryMessages inserts memory context as a separate message in the
// conversation, following the openai-agents-python pattern where context is
// injected as conversation items rather than appended to the system prompt.
//
// The memory message is inserted right after any system/developer messages
// but before user messages, keeping it clearly separated from instructions.
func injectMemoryMessages(requestBody []byte, content string) ([]byte, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	messages, ok := request["messages"].([]interface{})
	if !ok {
		messages = []interface{}{}
	}

	memoryMessage := map[string]interface{}{
		"role":    "user",
		"content": content,
	}

	// Find insertion point: after the last system/developer message
	insertIdx := 0
	for i, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		if role, ok := msgMap["role"].(string); ok && (role == "system" || role == "developer") {
			insertIdx = i + 1
		}
	}

	// Insert memory message at the computed position
	newMessages := make([]interface{}, 0, len(messages)+1)
	newMessages = append(newMessages, messages[:insertIdx]...)
	newMessages = append(newMessages, memoryMessage)
	newMessages = append(newMessages, messages[insertIdx:]...)

	request["messages"] = newMessages

	modifiedBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified request: %w", err)
	}

	logging.Debugf("Memory: Injected memory as separate message at position %d", insertIdx)
	return modifiedBody, nil
}
