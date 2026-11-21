package memory

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SDKMessageContent extracts the plain-text content from an SDK message union.
func SDKMessageContent(msg openai.ChatCompletionMessageParamUnion) string {
	switch {
	case msg.OfUser != nil:
		return extractUserContentString(msg.OfUser.Content)
	case msg.OfSystem != nil:
		return extractSystemContentString(msg.OfSystem.Content)
	case msg.OfAssistant != nil:
		return extractAssistantContentString(msg.OfAssistant.Content)
	default:
		return ""
	}
}

func extractUserContentString(c openai.ChatCompletionUserMessageParamContentUnion) string {
	if c.OfString.Value != "" {
		return c.OfString.Value
	}
	var parts []string
	for _, p := range c.OfArrayOfContentParts {
		if p.OfText != nil {
			parts = append(parts, p.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}

func extractSystemContentString(c openai.ChatCompletionSystemMessageParamContentUnion) string {
	if c.OfString.Value != "" {
		return c.OfString.Value
	}
	var parts []string
	for _, p := range c.OfArrayOfContentParts {
		if p.Text != "" {
			parts = append(parts, p.Text)
		}
	}
	return strings.Join(parts, " ")
}

func extractAssistantContentString(c openai.ChatCompletionAssistantMessageParamContentUnion) string {
	if c.OfString.Value != "" {
		return c.OfString.Value
	}
	var parts []string
	for _, p := range c.OfArrayOfContentParts {
		if p.OfText != nil {
			parts = append(parts, p.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}

// sdkHistoryMessages adapts the string-based extractor API to the same
// ordered content used by the router. Other roles still separate turns.
func sdkHistoryMessages(history []openai.ChatCompletionMessageParamUnion) []llmprotocol.Message {
	messages := make([]llmprotocol.Message, len(history))
	for i, msg := range history {
		switch {
		case msg.OfUser != nil:
			messages[i] = llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText, Text: SDKMessageContent(msg),
			}}}
		case msg.OfAssistant != nil:
			messages[i] = llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: sdkAssistantParts(msg.OfAssistant)}
		}
	}
	return messages
}

func sdkAssistantParts(msg *openai.ChatCompletionAssistantMessageParam) []llmprotocol.Content {
	var parts []llmprotocol.Content
	if text := msg.Content.OfString.Value; text != "" {
		parts = append(parts, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: text})
	}
	for _, part := range msg.Content.OfArrayOfContentParts {
		switch {
		case part.OfText != nil:
			parts = append(parts, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.OfText.Text})
		case part.OfRefusal != nil:
			parts = append(parts, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: part.OfRefusal.Refusal})
		}
	}
	if refusal := msg.Refusal.Value; refusal != "" {
		parts = append(parts, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: refusal})
	}
	for _, call := range msg.ToolCalls {
		parts = append(parts, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
			Name: call.Function.Name, Arguments: call.Function.Arguments,
		}})
	}
	return parts
}

// sdkUserMessage constructs an SDK user message from a plain string.
func sdkUserMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String(content),
			},
		},
	}
}

// sdkAssistantMessage constructs an SDK assistant message from a plain string.
func sdkAssistantMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			Content: openai.ChatCompletionAssistantMessageParamContentUnion{
				OfString: openai.String(content),
			},
		},
	}
}

// SDKMessageForRole constructs an SDK message for the given role and content.
func SDKMessageForRole(role, content string) openai.ChatCompletionMessageParamUnion {
	switch role {
	case "system":
		return openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(content),
				},
			},
		}
	case "assistant":
		return sdkAssistantMessage(content)
	default:
		return sdkUserMessage(content)
	}
}

// =============================================================================
// Memory Chunk Store
// =============================================================================

// DefaultSessionWindowSize is the number of turns included in each session
// chunk. A sliding window of this size is stored every DefaultSessionStride
// turns, creating overlapping chunks for better multi-hop retrieval coverage.
const DefaultSessionWindowSize = 5

// DefaultSessionStride is how often (in turns) a session chunk is stored.
// Stride < WindowSize creates overlapping windows so facts at window
// boundaries still appear together in at least one chunk.
const DefaultSessionStride = 3

// MemoryExtractor stores conversation turns directly in the vector store.
// No LLM extraction is performed -- the original user question and assistant
// response (with think tags stripped from prose) are embedded and stored as-is, preserving
// full conversation fidelity with zero additional inference cost.
//
// In addition to per-turn chunks, a session-level rolling window chunk is
// stored every N turns (default 5) to improve multi-hop retrieval.
//
// Usage:
//
//	store := NewMemoryChunkStore(milvusStore)
//	count, err := store.ProcessResponseWithHistoryCount(ctx, sessionID, userID, userMsg, assistantMsg, history)
type MemoryExtractor struct {
	store             Store
	sessionWindowSize int
	sessionStride     int
}

// NewMemoryChunkStore creates a MemoryExtractor backed by the given Store.
// Unlike the previous LLM-based extractor, this requires no external model
// endpoint -- conversation chunks are embedded and stored directly.
func NewMemoryChunkStore(store Store) *MemoryExtractor {
	if store == nil {
		return nil
	}
	return &MemoryExtractor{
		store:             store,
		sessionWindowSize: DefaultSessionWindowSize,
		sessionStride:     DefaultSessionStride,
	}
}

// =============================================================================
// Think-Tag Stripping
// =============================================================================

var (
	thinkClosedPattern   = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)
	thinkUnclosedPattern = regexp.MustCompile(`(?s)<think>.*`)
	memoryIDSequence     atomic.Uint64
)

// StripThinkTags removes <think>...</think> blocks (and unclosed <think> tails)
// from LLM output. This is a safety net for backends that embed reasoning in
// the content field instead of the separate reasoning_content field.
func StripThinkTags(s string) string {
	s = thinkClosedPattern.ReplaceAllString(s, "")
	s = thinkUnclosedPattern.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

// =============================================================================
// Direct Chunk Storage
// =============================================================================

// ProcessResponse stores the current conversation turn without prior history.
func (e *MemoryExtractor) ProcessResponse(
	ctx context.Context,
	sessionID string,
	userID string,
	userMessage string,
	assistantResponse string,
) error {
	return e.ProcessResponseWithHistory(ctx, sessionID, userID, userMessage, assistantResponse, nil)
}

// ProcessResponseWithHistory stores the current turn and any due session chunk.
// Call ProcessResponseWithHistoryCount when the successful-write count is needed.
func (e *MemoryExtractor) ProcessResponseWithHistory(
	ctx context.Context,
	sessionID string,
	userID string,
	userMessage string,
	assistantResponse string,
	history []openai.ChatCompletionMessageParamUnion,
) error {
	_, err := e.ProcessResponseWithHistoryCount(ctx, sessionID, userID, userMessage, assistantResponse, history)
	return err
}

// ProcessResponseWithHistoryCount stores the current conversation turn directly in
// the vector store (Q: ... / A: ...) and, every N turns, a session-level
// rolling window chunk that concatenates recent turns for multi-hop retrieval.
// assistantResponse is raw prose. Call StoreAssistantTurn with an answer
// rendered from structured prose and tool calls.
//
// The session chunk is built from the conversation history (obtained from the
// Response API's previous_response_id chain) plus the current turn. This gives
// retrieval a broader context window to match multi-hop queries that span
// multiple turns.
//
// Low-entropy turns are skipped for per-turn storage but still counted toward
// the session window trigger.
// The returned count is the number of chunks successfully stored. A write failure
// is returned even if an earlier chunk succeeded; successful writes are not rolled back.
func (e *MemoryExtractor) ProcessResponseWithHistoryCount(
	ctx context.Context,
	_ string, // sessionID (unused, kept for interface compatibility)
	userID string,
	userMessage string,
	assistantResponse string,
	history []openai.ChatCompletionMessageParamUnion,
) (int, error) {
	return e.storeAssistantTurn(ctx, userID, userMessage, func() (string, []llmprotocol.Message) {
		return StripThinkTags(assistantResponse), sdkHistoryMessages(history)
	})
}

// StoreAssistantTurn stores a rendered assistant answer and any due session
// window. The caller has already stripped reasoning from the answer's prose
// while it could distinguish prose from tool arguments. Prior messages retain
// their structured content until the rolling window needs them.
func (e *MemoryExtractor) StoreAssistantTurn(
	ctx context.Context, userID, userMessage, assistantResponse string,
	history []llmprotocol.Message,
) (int, error) {
	return e.storeAssistantTurn(ctx, userID, userMessage, func() (string, []llmprotocol.Message) {
		return assistantResponse, history
	})
}

func (e *MemoryExtractor) storeAssistantTurn(
	ctx context.Context, userID, userMessage string, prepare func() (string, []llmprotocol.Message),
) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if e == nil || e.store == nil || !e.store.IsEnabled() {
		logging.Infof("Memory chunk store: SKIPPED - store not enabled (store=%v)", e != nil && e.store != nil)
		return 0, nil
	}

	assistantResponse, history := prepare()
	if userMessage == "" && assistantResponse == "" {
		logging.Debugf("Memory chunk store: SKIPPED - empty turn")
		return 0, nil
	}

	startTime := time.Now()
	factsCount := 0
	status := "success"
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryExtraction(status, duration, factsCount, "episodic")
	}()

	// --- Per-turn chunk ---
	if isLowEntropy(userMessage, assistantResponse) {
		logging.Debugf("Memory chunk store: SKIPPED low-entropy turn for user=%s", userID)
	} else if stored, err := e.storeTurnChunk(ctx, userMessage, assistantResponse, userID); err != nil {
		status = "error"
		// Negative count skips the facts-count histogram for failed extraction attempts.
		factsCount = -1
		return 0, err
	} else if stored {
		factsCount++
	}

	// --- Session-level rolling window chunk ---
	if stored, err := e.maybeStoreSessionChunk(ctx, userID, userMessage, assistantResponse, history); err != nil {
		status = "error"
		storedCount := factsCount
		// Preserve the successful-write count while excluding failed attempts
		// from the extraction facts-count histogram.
		factsCount = -1
		return storedCount, err
	} else if stored {
		// The session window is stored as a separate episodic memory chunk.
		factsCount++
	}

	return factsCount, nil
}

func (e *MemoryExtractor) storeTurnChunk(ctx context.Context, userMessage, assistantResponse, userID string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	chunk := formatTurnChunk(userMessage, assistantResponse)
	sanitized, err := sanitizeMemoryContent(chunk)
	if err != nil {
		logging.Debugf("Memory chunk store: sanitization rejected: not stored - %v (user=%s)", err, userID)
		return false, nil
	}
	mem := &Memory{
		ID:         generateMemoryID(),
		Type:       MemoryTypeEpisodic,
		Content:    sanitized,
		UserID:     userID,
		Source:     "conversation",
		CreatedAt:  time.Now(),
		Importance: 0.5,
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if err := e.store.Store(ctx, mem); err != nil {
		return false, fmt.Errorf("failed to store conversation chunk: %w", err)
	}
	logging.Infof("Memory chunk store: stored turn for user=%s (len=%d)", userID, len(sanitized))
	return true, nil
}

// maybeStoreSessionChunk stores a session-level chunk every `stride` turns.
// Each chunk covers the last `windowSize` turns, creating overlapping windows
// (stride < windowSize) so facts near window boundaries still co-occur in at
// least one chunk -- critical for multi-hop retrieval. It returns true when a
// session chunk was actually stored, and propagates backend write failures.
func (e *MemoryExtractor) maybeStoreSessionChunk(
	ctx context.Context,
	userID string,
	userMessage string,
	assistantResponse string,
	history []llmprotocol.Message,
) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if len(history) == 0 {
		return false, nil
	}

	windowSize := e.sessionWindowSize
	if windowSize <= 0 {
		windowSize = DefaultSessionWindowSize
	}
	stride := e.sessionStride
	if stride <= 0 {
		stride = DefaultSessionStride
	}

	// history contains prior turns; +1 for the current turn
	totalTurns := countTurns(history) + 1
	if totalTurns < stride {
		return false, nil
	}
	// Fire on every stride-th turn (once we have enough turns for a window)
	if totalTurns%stride != 0 {
		return false, nil
	}

	// Build session chunk from the last windowSize turns of history + current
	sessionChunk := buildSessionChunk(history, userMessage, assistantResponse, windowSize)
	if sessionChunk == "" {
		return false, nil
	}

	sanitized, err := sanitizeMemoryContent(sessionChunk)
	if err != nil {
		logging.Debugf("Memory session chunk: REJECTED - %v (user=%s)", err, userID)
		return false, nil
	}

	mem := &Memory{
		ID:         generateMemoryID(),
		Type:       MemoryTypeEpisodic,
		Content:    sanitized,
		UserID:     userID,
		Source:     "session_window",
		CreatedAt:  time.Now(),
		Importance: 0.7,
	}

	if err := ctx.Err(); err != nil {
		return false, err
	}
	if err := e.store.Store(ctx, mem); err != nil {
		return false, fmt.Errorf("failed to store session chunk: %w", err)
	}

	logging.Infof("Memory session chunk: stored %d-turn window (stride=%d) for user=%s (len=%d)",
		windowSize, stride, userID, len(sanitized))
	return true, nil
}

// countTurns counts user turns in a message history. Each user message is one turn.
func countTurns(history []llmprotocol.Message) int {
	n := 0
	for _, m := range history {
		if m.Role == llmprotocol.RoleUser {
			n++
		}
	}
	return n
}

// buildSessionChunk concatenates the last windowSize turns from history plus
// the current turn into a single multi-turn context chunk.
func buildSessionChunk(history []llmprotocol.Message, userMsg, assistantResp string, windowSize int) string {
	var chunks []string
	for i := len(history) - 1; i >= 0 && len(chunks) < windowSize-1; i-- {
		if msg := history[i]; msg.Role == llmprotocol.RoleUser {
			chunks = append(chunks, formatTurnChunk(llmprotocol.TextContent(msg.Content), historyAssistantText(history[i+1:])))
		}
	}
	slices.Reverse(chunks)
	chunks = append(chunks, formatTurnChunk(userMsg, assistantResp))
	return strings.Join(chunks, "\n---\n")
}

// Responses can represent the prose and tool calls of one assistant turn as
// consecutive items. Keep their boundaries until the complete answer is rendered.
func historyAssistantText(history []llmprotocol.Message) string {
	var output []llmprotocol.OutputItem
	for _, msg := range history {
		if msg.Role != llmprotocol.RoleAssistant {
			break
		}
		output = append(output, llmprotocol.OutputItem{Role: msg.Role, Content: msg.Content})
	}
	return (&llmprotocol.Response{Output: output}).AssistantText(StripThinkTags)
}

// formatTurnChunk combines a user message and assistant response into a single
// searchable chunk. Both halves are included so that semantic search can match
// on either the question or the answer.
func formatTurnChunk(userMessage, assistantResponse string) string {
	var parts []string
	if userMessage != "" {
		parts = append(parts, "Q: "+userMessage)
	}
	if assistantResponse != "" {
		parts = append(parts, "A: "+assistantResponse)
	}
	return strings.Join(parts, "\n")
}

// =============================================================================
// Entropy Gate (SimpleMem-inspired)
// =============================================================================

// lowEntropyPatterns match user messages that carry no retrievable information.
var lowEntropyPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^(hi|hello|hey|howdy|yo|sup)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(good\s+)?(morning|afternoon|evening|night)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(thanks|thank\s+you|thx|ty)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(bye|goodbye|see\s+you|later|cheers)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(ok|okay|sure|yes|no|yep|nope|yea|nah|k|alright|got\s+it)[\s\!\.\,]*$`),
	regexp.MustCompile(`(?i)^(cool|great|nice|awesome|perfect|sounds\s+good)[\s\!\.\,]*$`),
}

// refusalPatterns match assistant responses that carry no retrievable information.
var refusalPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^i('m|\s+am)\s+(sorry|unable|not\s+able|afraid\s+i\s+can)`),
	regexp.MustCompile(`(?i)^(as\s+an?\s+ai|i\s+don'?t\s+have\s+(access|the\s+ability))`),
	regexp.MustCompile(`(?i)^i\s+can'?t\s+(help|assist|provide)\s+with\s+that`),
}

const minTurnLength = 30

// isLowEntropy returns true if a conversation turn is unlikely to contain
// useful information for future retrieval. Skipping these avoids polluting
// the memory store with greetings, acknowledgments, and refusals.
func isLowEntropy(userMessage, assistantResponse string) bool {
	userTrimmed := strings.TrimSpace(userMessage)
	assistantTrimmed := strings.TrimSpace(assistantResponse)

	combinedLen := len(userTrimmed) + len(assistantTrimmed)
	if combinedLen < minTurnLength {
		return true
	}

	if userTrimmed != "" {
		for _, pat := range lowEntropyPatterns {
			if pat.MatchString(userTrimmed) {
				return true
			}
		}
	}

	if assistantTrimmed != "" {
		for _, pat := range refusalPatterns {
			if pat.MatchString(assistantTrimmed) {
				return true
			}
		}
	}

	return false
}

// =============================================================================
// Helpers
// =============================================================================

func generateMemoryID() string {
	return fmt.Sprintf("mem_%d_%d", time.Now().UnixNano(), memoryIDSequence.Add(1))
}
