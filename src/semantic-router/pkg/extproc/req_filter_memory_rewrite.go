package extproc

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// ConversationMessage represents a message in conversation history.
type ConversationMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// memoryRewriteServiceConfig describes the optional internal system service
// used to improve vector-memory queries. It is not a request-selected logical
// Model backend and cannot participate in routing, fallback, or usage dispatch.
type memoryRewriteServiceConfig struct {
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

func getMaxTokens(resolved *memoryRewriteServiceConfig, defaultValue int) int {
	if resolved != nil && resolved.MaxTokens > 0 {
		return resolved.MaxTokens
	}
	return defaultValue
}

func getTemperature(resolved *memoryRewriteServiceConfig, defaultValue float64) float64 {
	if resolved != nil && resolved.Temperature > 0 {
		return resolved.Temperature
	}
	return defaultValue
}

func getTimeout(resolved *memoryRewriteServiceConfig) time.Duration {
	if resolved != nil && resolved.TimeoutSeconds > 0 {
		return time.Duration(resolved.TimeoutSeconds) * time.Second
	}
	return 5 * time.Second
}

// BuildSearchQuery rewrites a query with conversation context for semantic search.
// It uses an LLM to understand context and produce a self-contained query.
func BuildSearchQuery(ctx context.Context, history []ConversationMessage, query string, routerCfg *config.RouterConfig) (string, error) {
	resolved := resolveMemoryRewriteServiceConfig(routerCfg)
	if resolved == nil || resolved.Endpoint == "" {
		logging.Debugf("Memory: Query rewriting not configured, using original query")
		return query, nil
	}

	historyText := formatHistoryForPrompt(history)
	userPrompt := fmt.Sprintf("History:\n%s\n\nQuery: %s\n\nRewritten query:", historyText, query)

	logging.Debugf("Memory: query rewrite: original=%s, history_len=%d", logging.ContentDescriptor(query), len(history))

	rewrittenQuery, err := callMemoryRewriteService(ctx, resolved, userPrompt)
	if err != nil {
		logging.Errorf("Memory: Query rewriting failed, using original (error_class=%T)", err)
		return query, nil
	}

	rewrittenQuery = strings.TrimSpace(rewrittenQuery)
	rewrittenQuery = strings.Trim(rewrittenQuery, "\"'")

	logging.Debugf("Memory: query rewrite: result=%s", logging.ContentDescriptor(rewrittenQuery))

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

// resolveMemoryRewriteServiceConfig resolves only the internal
// memory_rewrite system-model role. Request-facing Model assignments and
// physical backend selection never enter this adapter.
func resolveMemoryRewriteServiceConfig(routerCfg *config.RouterConfig) *memoryRewriteServiceConfig {
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
	return &memoryRewriteServiceConfig{
		Endpoint:       endpoint,
		Model:          externalCfg.ModelName,
		AccessKey:      externalCfg.AccessKey,
		TimeoutSeconds: externalCfg.TimeoutSeconds,
		MaxTokens:      externalCfg.MaxTokens,
		Temperature:    externalCfg.Temperature,
	}
}

// callMemoryRewriteService invokes the dedicated internal memory helper. It is
// intentionally outside BackendInvoker because it cannot serve a public Model
// request and its output is only an input to vector-memory retrieval.
func callMemoryRewriteService(ctx context.Context, resolved *memoryRewriteServiceConfig, userPrompt string) (string, error) {
	maxOutput := int64(getMaxTokens(resolved, 512))
	temperature := getTemperature(resolved, 0.1)
	request := llmprotocol.Request{
		Generation: 1,
		Model:      resolved.Model,
		Instructions: []llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: queryRewriteSystemPrompt}},
		}},
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: userPrompt}},
		}},
		Sampling: llmprotocol.Sampling{
			MaxOutputTokens: &maxOutput,
			Temperature:     &temperature,
		},
	}
	engine := protocolcodec.NewBuiltinEngine()
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	if err != nil {
		return "", fmt.Errorf("encode memory rewrite request: %w", err)
	}
	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimSuffix(resolved.Endpoint, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(encoded.Body))
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
		return "", fmt.Errorf("LLM returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	decoded, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil,
	)
	if err != nil {
		return "", fmt.Errorf("decode memory rewrite response: %w", err)
	}
	content := semanticResponseText(decoded.Response)
	if content == "" {
		return "", fmt.Errorf("no choices in LLM response")
	}
	return memory.StripThinkTags(content), nil
}

func semanticResponseText(response llmprotocol.Response) string {
	var text strings.Builder
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentText {
				text.WriteString(content.Text)
			}
		}
	}
	return text.String()
}
