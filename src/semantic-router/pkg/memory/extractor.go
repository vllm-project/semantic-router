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
	"sync"
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
// The LLM endpoint is configured via external_models with model_role="memory_extraction".
//
// It supports two modes:
//  1. Extraction only: Use ExtractFacts() to extract facts without storing them
//  2. Extraction + Storage with deduplication: Use ProcessResponse() to extract and store with deduplication
//
// Usage:
//
//	// Extraction + Storage with deduplication:
//	extractorWithStore := NewMemoryExtractorWithStore(routerCfg, batchSize, store)
//	err := extractorWithStore.ProcessResponse(ctx, sessionID, userID, history)
type MemoryExtractor struct {
	endpoint    string       // Resolved LLM endpoint
	model       string       // Resolved model name
	client      *http.Client // Reused for connection pooling
	store       Store        // Optional: for ProcessResponse with deduplication
	turnCounts  map[string]int
	mu          sync.Mutex
	dedupConfig DeduplicationConfig
	// LLM generation parameters from external_models config
	maxTokens   int
	temperature float64
	batchSize   int
}

// NewMemoryExtractorWithStore creates a new MemoryExtractor with router config for external model resolution and store.
// This enables ProcessResponse which handles extraction + storage with deduplication.
// The LLM endpoint is resolved from external_models using model_role="memory_extraction".
// batchSize controls how often extraction runs (every N turns). Use 0 for default (10).
func NewMemoryExtractorWithStore(routerCfg *config.RouterConfig, batchSize int, store Store) *MemoryExtractor {
	// Resolve LLM endpoint and params from external_models
	resolved := resolveExtractionConfig(routerCfg)
	if resolved == nil {
		return nil
	}

	if batchSize <= 0 {
		batchSize = 10 // default
	}

	return &MemoryExtractor{
		endpoint:    resolved.endpoint,
		model:       resolved.model,
		client:      &http.Client{Timeout: resolved.timeout},
		store:       store,
		turnCounts:  make(map[string]int),
		dedupConfig: DefaultDeduplicationConfig(),
		maxTokens:   resolved.maxTokens,
		temperature: resolved.temperature,
		batchSize:   batchSize,
	}
}

// resolvedExtractionConfig holds resolved extraction LLM configuration
type resolvedExtractionConfig struct {
	endpoint    string
	model       string
	timeout     time.Duration
	maxTokens   int
	temperature float64
}

// resolveExtractionConfig resolves the LLM endpoint and params from external_models.
func resolveExtractionConfig(routerCfg *config.RouterConfig) *resolvedExtractionConfig {
	if routerCfg == nil {
		return nil
	}

	externalCfg := routerCfg.FindExternalModelByRole(config.ModelRoleMemoryExtraction)
	if externalCfg == nil || externalCfg.ModelEndpoint.Address == "" {
		return nil
	}

	timeout := 30 * time.Second
	if externalCfg.TimeoutSeconds > 0 {
		timeout = time.Duration(externalCfg.TimeoutSeconds) * time.Second
	}

	maxTokens := externalCfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 500 // default for extraction
	}

	temperature := externalCfg.Temperature
	if temperature <= 0 {
		temperature = 0.1 // default
	}

	return &resolvedExtractionConfig{
		endpoint:    fmt.Sprintf("http://%s:%d", externalCfg.ModelEndpoint.Address, externalCfg.ModelEndpoint.Port),
		model:       externalCfg.ModelName,
		timeout:     timeout,
		maxTokens:   maxTokens,
		temperature: temperature,
	}
}

// SetDeduplicationConfig sets the deduplication configuration.
func (e *MemoryExtractor) SetDeduplicationConfig(config DeduplicationConfig) {
	e.dedupConfig = config
}

// =============================================================================
// LLM-Based Fact Extraction
// =============================================================================

// extractionSystemPrompt is the system prompt for fact extraction
const extractionSystemPrompt = `You are a memory extraction system. Extract important USER information from conversations.

CRITICAL RULES:
1. Extract ONLY facts stated by or about the USER
2. DO NOT extract assistant suggestions, recommendations, or general knowledge
3. ALWAYS include context - never extract isolated values
4. Use self-contained phrases that make sense without the conversation
5. Return ONLY valid JSON - no explanations or markdown
6. ALWAYS phrase facts as STATEMENTS, never as questions
7. Include CONSTRAINTS and LIMITATIONS explicitly (cannot, must not, excluded, etc.)

MEMORY TYPES:

"semantic" - User's facts, preferences, constraints, knowledge:
  - Personal info: "User's name is Alex", "User works at Acme Corp"
  - Preferences: "User prefers window seats", "User likes spicy food"
  - Constraints: "User is allergic to shellfish", "User's budget is $5000"
  - Limitations: "User cannot use AWS", "User must deploy on Azure only"
  - Tech stack: "User's project uses React frontend and Go backend"
  - Knowledge: "User knows Python and Go", "User studied at MIT"

"procedural" - User's personal workflows, routines, or processes they EXPLICITLY describe:
  - "User's morning routine: check Slack, review PRs, then standup at 9am"
  - "User deploys code by: running tests, then pushing to staging, then production"
  - "User prefers to debug by: adding logs first, then using breakpoints"
  NOTE: This is for USER's own processes, NOT assistant recommendations!

WHAT NOT TO EXTRACT:
- Assistant suggestions ("You should try the seafood restaurant")
- General knowledge ("Python is a programming language")
- Hypotheticals ("If I had more time, I would...")
- Questions (never phrase as "What is user's budget?" - use statements!)

EXAMPLES:
GOOD: [{"type": "semantic", "content": "User is lactose intolerant"}]
GOOD: [{"type": "semantic", "content": "User's project deadline is March 15th"}]
GOOD: [{"type": "semantic", "content": "User cannot use AWS due to company policy"}]
GOOD: [{"type": "semantic", "content": "User's tech stack is React, Go, and PostgreSQL"}]
GOOD: [{"type": "procedural", "content": "User's code review process: check tests, review logic, then check style"}]
BAD:  [{"type": "semantic", "content": "What is the user's budget?"}] (question form - use statement!)
BAD:  [{"type": "procedural", "content": "To improve code: add more tests"}] (assistant advice, not user's process)

Return JSON array. Empty array [] if nothing worth remembering about the USER.`

// ExtractFacts extracts memorable facts from a conversation using an LLM.
// This is a pure extraction function - it does NOT store the facts.
// Use ProcessResponse if you want extraction + storage with deduplication.
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
	if e == nil || e.endpoint == "" {
		logging.Debugf("Memory: Fact extraction not configured")
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
		logging.Infof("║   %d. [%s] %s", i+1, fact.Type, fact.Content) // Full content for demo
	}
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	return facts, nil
}

// callLLMForExtraction calls the configured LLM endpoint for fact extraction
func (e *MemoryExtractor) callLLMForExtraction(ctx context.Context, userPrompt string) ([]ExtractedFact, error) {
	// Build request
	reqBody := llmChatRequest{
		Model: e.model,
		Messages: []llmChatMessage{
			{Role: "system", Content: extractionSystemPrompt},
			{Role: "user", Content: userPrompt},
		},
		MaxTokens:   e.maxTokens,
		Temperature: e.temperature,
		Stream:      false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request (timeout is set on http.Client during construction)
	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimSuffix(e.endpoint, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
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
// Extraction + Storage with Deduplication
// =============================================================================

// ProcessResponse extracts and stores memories from conversation history.
// It runs extraction every N turns (batchSize) to avoid excessive LLM calls.
// This method combines extraction (using ExtractFacts) + storage with deduplication.
func (e *MemoryExtractor) ProcessResponse(
	ctx context.Context,
	sessionID string,
	userID string,
	history []Message,
) error {
	// TODO: Remove demo logging after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║              MEMORY EXTRACTION: ProcessResponse                  ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ sessionID: %s", sessionID)
	logging.Infof("║ userID: %s", userID)
	logging.Infof("║ historyLen: %d", len(history))
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	if e.store == nil || !e.store.IsEnabled() {
		logging.Infof("Memory extraction: SKIPPED - store not enabled (store=%v)", e.store != nil)
		return nil // Store not enabled, skip extraction
	}

	if e == nil || e.endpoint == "" {
		logging.Infof("Memory extraction: SKIPPED - extraction not configured")
		return nil // Extraction not enabled
	}

	// Track turn count for this session
	e.mu.Lock()
	e.turnCounts[sessionID]++
	turnCount := e.turnCounts[sessionID]
	e.mu.Unlock()

	// Use batch size from struct (set at construction)
	batchSize := e.batchSize

	// TODO: Remove demo logging after POC
	logging.Infof("Memory extraction: turnCount=%d, batchSize=%d, shouldExtract=%v",
		turnCount, batchSize, turnCount%batchSize == 0)

	// Only extract every N turns
	if turnCount%batchSize != 0 {
		logging.Infof("Memory extraction: SKIPPED - not batch turn (turn %d, batch every %d)", turnCount, batchSize)
		return nil
	}

	// Get recent batch (last N+5 messages for context)
	batchStart := 0
	if len(history) > batchSize+5 {
		batchStart = len(history) - batchSize - 5
	}
	batch := history[batchStart:]

	// Use ExtractFacts for extraction
	extracted, err := e.ExtractFacts(ctx, batch)
	if err != nil {
		logging.Warnf("Memory extraction failed: %v", err)
		return err // Return error but don't block response
	}

	if len(extracted) == 0 {
		logging.Debugf("Memory extraction: no facts extracted from batch")
		return nil
	}

	// Store with deduplication
	for _, fact := range extracted {
		if err := e.storeWithDeduplication(ctx, userID, fact); err != nil {
			logging.Warnf("Failed to store memory with deduplication: %v", err)
			// Continue with other facts even if one fails
		}
	}

	return nil
}

// storeWithDeduplication stores a fact with deduplication logic.
// It checks for similar existing memories and either updates or creates new ones.
func (e *MemoryExtractor) storeWithDeduplication(
	ctx context.Context,
	userID string,
	fact ExtractedFact,
) error {
	// TODO: Remove demo logging after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║              DEDUPLICATION CHECK                                 ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ userID: %s", userID)
	logging.Infof("║ fact.Type: %s", fact.Type)
	logging.Infof("║ fact.Content: %s", fact.Content) // Full content for demo
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	// Check for similar memories using deduplication logic
	result := CheckDeduplication(ctx, e.store, userID, fact.Content, fact.Type, e.dedupConfig)

	// TODO: Remove demo logging after POC
	logging.Infof("Deduplication result: action=%s, similarity=%.3f, existingID=%v",
		result.Action, result.Similarity, result.ExistingMemory != nil)

	switch result.Action {
	case "update":
		// Very similar → UPDATE existing memory
		if result.ExistingMemory == nil {
			// Should not happen, but handle gracefully
			logging.Warnf("Memory deduplication: update action but no existing memory")
			return e.createNewMemory(ctx, userID, fact)
		}

		// Update existing memory with new content
		result.ExistingMemory.Content = fact.Content // Use newer content
		result.ExistingMemory.UpdatedAt = time.Now()

		if err := e.store.Update(ctx, result.ExistingMemory.ID, result.ExistingMemory); err != nil {
			return fmt.Errorf("failed to update memory: %w", err)
		}

		logging.Infof("Memory deduplication: UPDATED memory id=%s (similarity=%.3f)",
			result.ExistingMemory.ID, result.Similarity)
		return nil

	case "create":
		// Create new memory (either no similar found, or in gray zone)
		return e.createNewMemory(ctx, userID, fact)

	default:
		// Unknown action - default to create
		logging.Warnf("Memory deduplication: unknown action '%s', defaulting to create", result.Action)
		return e.createNewMemory(ctx, userID, fact)
	}
}

// createNewMemory creates a new memory from an extracted fact.
func (e *MemoryExtractor) createNewMemory(
	ctx context.Context,
	userID string,
	fact ExtractedFact,
) error {
	mem := &Memory{
		ID:         generateMemoryID(),
		Type:       fact.Type,
		Content:    fact.Content,
		UserID:     userID,
		Source:     "conversation",
		CreatedAt:  time.Now(),
		Importance: 0.5, // Default importance
	}

	if err := e.store.Store(ctx, mem); err != nil {
		return fmt.Errorf("failed to store memory: %w", err)
	}

	logging.Infof("Memory deduplication: CREATED new memory id=%s, type=%s", mem.ID, mem.Type)
	return nil
}

// generateMemoryID generates a unique memory ID
func generateMemoryID() string {
	return fmt.Sprintf("mem_%d", time.Now().UnixNano())
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
