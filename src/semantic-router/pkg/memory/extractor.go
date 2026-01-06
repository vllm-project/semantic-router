package memory

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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// =============================================================================
// Memory Extractor
// =============================================================================

// MemoryExtractor extracts facts from conversation history using an LLM.
// It analyzes conversation messages and identifies important information
// to store in long-term memory (facts, preferences, procedural knowledge).
//
// Usage:
//
//	extractor := NewMemoryExtractor(cfg)
//	facts, err := extractor.ExtractFacts(ctx, messages)
type MemoryExtractor struct {
	config *config.ExtractionConfig
	client *http.Client // Reused for connection pooling
}

// NewMemoryExtractor creates a new MemoryExtractor with the given configuration.
// The http.Client is reused across requests for connection pooling.
func NewMemoryExtractor(cfg *config.ExtractionConfig) *MemoryExtractor {
	return &MemoryExtractor{
		config: cfg,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

// =============================================================================
// LLM-Based Fact Extraction
// =============================================================================

// extractionSystemPrompt is the system prompt for fact extraction
const extractionSystemPrompt = `You are a memory extraction system. Extract important information from conversations.

RULES:
1. Extract facts, preferences, decisions, and procedural knowledge
2. ALWAYS include context - never extract isolated values
3. Use self-contained phrases that make sense without the conversation
4. Return ONLY valid JSON - no explanations or markdown

EXAMPLES:
BAD:  {"type": "semantic", "content": "budget is $10,000"}
GOOD: {"type": "semantic", "content": "User's budget for Hawaii vacation is $10,000"}

BAD:  {"type": "procedural", "content": "run npm build"}
GOOD: {"type": "procedural", "content": "To deploy payment-service: run npm build, then docker push"}

TYPES:
- "semantic": Facts, preferences, knowledge (e.g., "User prefers direct flights")
- "procedural": Instructions, how-to, steps (e.g., "To reset password: click forgot, enter email")

Return JSON array. Empty array [] if nothing worth remembering.`

// ExtractFacts extracts memorable facts from a conversation using an LLM.
//
// Error handling:
//   - Returns empty slice on any error (graceful degradation)
//   - Logs warnings for debugging but doesn't fail the response
//
// Example:
//
//	messages := []Message{
//	    {Role: "user", Content: "My budget for Hawaii is $10,000"},
//	    {Role: "assistant", Content: "Great! That's a good budget for Hawaii."},
//	}
//	facts, err := extractor.ExtractFacts(ctx, messages)
//	// facts = [{Type: "semantic", Content: "User's budget for Hawaii vacation is $10,000"}]
func (e *MemoryExtractor) ExtractFacts(ctx context.Context, messages []Message) ([]ExtractedFact, error) {
	if e.config == nil || !e.config.Enabled || e.config.Endpoint == "" {
		logging.Debugf("Memory: Fact extraction disabled or not configured")
		return nil, nil
	}

	if len(messages) == 0 {
		return nil, nil
	}

	// Format messages for the prompt
	conversationText := formatMessagesForExtraction(messages)

	// Build user prompt
	userPrompt := fmt.Sprintf("Extract important information from this conversation:\n\n%s\n\nReturn JSON array:", conversationText)

	// TODO: Remove debug logs after POC demo
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║                    MEMORY FACT EXTRACTION                        ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ MESSAGES TO EXTRACT FROM (%d messages):                          ║", len(messages))
	for _, msg := range messages {
		logging.Infof("║   [%s]: %s", msg.Role, truncateForLog(msg.Content, 50))
	}
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	// Call LLM for extraction
	facts, err := e.callLLMForExtraction(ctx, userPrompt)
	if err != nil {
		logging.Warnf("Memory: Fact extraction failed: %v", err)
		return nil, nil // Graceful degradation
	}

	// TODO: Remove debug logs after POC demo
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║ EXTRACTED FACTS (%d):                                            ║", len(facts))
	for i, fact := range facts {
		logging.Infof("║   %d. [%s] %s", i+1, fact.Type, truncateForLog(fact.Content, 45))
	}
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	return facts, nil
}

// callLLMForExtraction calls the configured LLM endpoint for fact extraction
func (e *MemoryExtractor) callLLMForExtraction(ctx context.Context, userPrompt string) ([]ExtractedFact, error) {
	// Build request
	reqBody := llmChatRequest{
		Model: e.config.Model,
		Messages: []llmChatMessage{
			{Role: "system", Content: extractionSystemPrompt},
			{Role: "user", Content: userPrompt},
		},
		MaxTokens:   getMaxTokensForExtraction(e.config),
		Temperature: getTemperatureForExtraction(e.config),
		Stream:      false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request with timeout
	timeout := getTimeoutForExtraction(e.config)
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimSuffix(e.config.Endpoint, "/"))
	httpReq, err := http.NewRequestWithContext(reqCtx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Send request (using reused client for connection pooling)
	resp, err := e.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("LLM request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("LLM returned status %d", resp.StatusCode)
	}

	// Parse response
	var llmResp llmChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(llmResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in LLM response")
	}

	// Parse extracted facts from LLM response
	content := llmResp.Choices[0].Message.Content
	return parseExtractedFacts(content)
}

// =============================================================================
// Response Parsing
// =============================================================================

// parseExtractedFacts parses the LLM response into ExtractedFact structs
func parseExtractedFacts(content string) ([]ExtractedFact, error) {
	// Clean up the response - remove markdown code blocks if present
	content = strings.TrimSpace(content)
	content = cleanJSONResponse(content)

	if content == "" || content == "[]" {
		return nil, nil
	}

	// Parse JSON array
	var facts []ExtractedFact
	if err := json.Unmarshal([]byte(content), &facts); err != nil {
		return nil, fmt.Errorf("failed to parse facts JSON: %w (content: %s)", err, truncateForLog(content, 100))
	}

	// Validate and filter facts
	validFacts := make([]ExtractedFact, 0, len(facts))
	for _, fact := range facts {
		// Skip empty content
		if strings.TrimSpace(fact.Content) == "" {
			continue
		}

		// Normalize type
		normalizedType := normalizeMemoryType(string(fact.Type))
		if normalizedType == "" {
			logging.Warnf("Memory: Skipping fact with invalid type: %s", fact.Type)
			continue
		}

		validFacts = append(validFacts, ExtractedFact{
			Type:    normalizedType,
			Content: strings.TrimSpace(fact.Content),
		})
	}

	return validFacts, nil
}

// cleanJSONResponse removes markdown code blocks and other formatting from LLM response
func cleanJSONResponse(content string) string {
	// Remove markdown code blocks
	// Match ```json ... ``` or ``` ... ```
	codeBlockPattern := regexp.MustCompile("(?s)```(?:json)?\\s*(.+?)\\s*```")
	if matches := codeBlockPattern.FindStringSubmatch(content); len(matches) > 1 {
		content = matches[1]
	}

	// Trim whitespace
	content = strings.TrimSpace(content)

	return content
}

// normalizeMemoryType converts string to MemoryType, returns empty string if invalid
func normalizeMemoryType(typeStr string) MemoryType {
	switch strings.ToLower(strings.TrimSpace(typeStr)) {
	case "semantic":
		return MemoryTypeSemantic
	case "procedural":
		return MemoryTypeProcedural
	case "episodic":
		return MemoryTypeEpisodic
	default:
		return ""
	}
}

// =============================================================================
// Helper Functions
// =============================================================================

// formatMessagesForExtraction formats messages for the LLM extraction prompt
func formatMessagesForExtraction(messages []Message) string {
	var lines []string
	for _, msg := range messages {
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

// getTimeoutForExtraction returns timeout with default (30s for extraction)
func getTimeoutForExtraction(cfg *config.ExtractionConfig) time.Duration {
	if cfg.TimeoutSeconds > 0 {
		return time.Duration(cfg.TimeoutSeconds) * time.Second
	}
	return 30 * time.Second // default for extraction (longer than query rewrite)
}

// getMaxTokensForExtraction returns max tokens with default
func getMaxTokensForExtraction(cfg *config.ExtractionConfig) int {
	if cfg.MaxTokens > 0 {
		return cfg.MaxTokens
	}
	return 500 // default - extraction can produce multiple facts
}

// getTemperatureForExtraction returns temperature with default
func getTemperatureForExtraction(cfg *config.ExtractionConfig) float64 {
	if cfg.Temperature > 0 {
		return cfg.Temperature
	}
	return 0.1 // default - low temperature for consistent extraction
}

// =============================================================================
// LLM Client Types (shared with req_filter_memory.go)
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

// =============================================================================
// Utility Functions
// =============================================================================

// ExtractFactsFromReader extracts facts from a reader (e.g., for testing)
func (e *MemoryExtractor) ExtractFactsFromReader(reader io.Reader) ([]ExtractedFact, error) {
	content, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read content: %w", err)
	}

	return parseExtractedFacts(string(content))
}
