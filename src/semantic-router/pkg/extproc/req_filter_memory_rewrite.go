package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ConversationMessage represents a message in conversation history.
type ConversationMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ResolvedLLMConfig holds resolved LLM endpoint configuration from external_models.
type ResolvedLLMConfig struct {
	Endpoint       string
	Model          string
	AccessKey      string
	TimeoutSeconds int
	MaxTokens      int
	Temperature    float64
}

// queryRewriteSystemPrompt is the system prompt for query rewriting.
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

func getMaxTokens(resolved *ResolvedLLMConfig, defaultValue int) int {
	if resolved != nil && resolved.MaxTokens > 0 {
		return resolved.MaxTokens
	}
	return defaultValue
}

func getTemperature(resolved *ResolvedLLMConfig, defaultValue float64) float64 {
	if resolved != nil && resolved.Temperature > 0 {
		return resolved.Temperature
	}
	return defaultValue
}

func getTimeout(resolved *ResolvedLLMConfig) time.Duration {
	if resolved != nil && resolved.TimeoutSeconds > 0 {
		return time.Duration(resolved.TimeoutSeconds) * time.Second
	}
	return 5 * time.Second
}

// BuildSearchQuery rewrites a query with conversation context for semantic search.
// It uses an LLM to understand context and produce a self-contained query.
func BuildSearchQuery(ctx context.Context, history []ConversationMessage, query string, routerCfg *config.RouterConfig) (string, error) {
	resolved := ResolveQueryRewriteConfig(routerCfg)
	if resolved == nil || resolved.Endpoint == "" {
		logging.Debugf("Memory: Query rewriting not configured, using original query")
		return query, nil
	}

	historyText := formatHistoryForPrompt(history)
	userPrompt := fmt.Sprintf("History:\n%s\n\nQuery: %s\n\nRewritten query:", historyText, query)

	logging.Debugf("Memory: query rewrite: original=%q, history_len=%d", truncateForLog(query, 80), len(history))

	rewrittenQuery, err := callLLMForQueryRewrite(ctx, resolved, userPrompt)
	if err != nil {
		logging.Errorf("Memory: Query rewriting failed, using original: %v", err)
		return query, nil
	}

	rewrittenQuery = strings.TrimSpace(rewrittenQuery)
	rewrittenQuery = strings.Trim(rewrittenQuery, "\"'")

	logging.Debugf("Memory: query rewrite: result=%q", truncateForLog(rewrittenQuery, 80))

	return rewrittenQuery, nil
}

func formatHistoryForPrompt(history []ConversationMessage) string {
	if len(history) == 0 {
		return "(no previous conversation)"
	}

	var lines []string
	startIdx := 0
	if len(history) > 5 {
		startIdx = len(history) - 5
	}

	for _, msg := range history[startIdx:] {
		lines = append(lines, fmt.Sprintf("[%s]: %s", msg.Role, msg.Content))
	}

	return strings.Join(lines, "\n")
}

func truncateForLog(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
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

	scheme := externalCfg.ModelEndpoint.Protocol
	if scheme == "" {
		scheme = "http"
	}
	endpoint := fmt.Sprintf("%s://%s:%d", scheme, externalCfg.ModelEndpoint.Address, externalCfg.ModelEndpoint.Port)
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

	client := &http.Client{Timeout: getTimeout(resolved)}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("LLM request failed: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("LLM returned status %d: %s", resp.StatusCode, truncateForLog(string(body), 200))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	var completion openai.ChatCompletion
	if err := json.Unmarshal(body, &completion); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("no choices in LLM response")
	}

	return memory.StripThinkTags(completion.Choices[0].Message.Content), nil
}

func setJSONField(data []byte, key string, value interface{}) ([]byte, error) {
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	m[key] = value
	return json.Marshal(m)
}

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
		if !ok || role == "system" {
			continue
		}

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
