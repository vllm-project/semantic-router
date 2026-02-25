package memory

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

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
// response (with think tags stripped) are embedded and stored as-is, preserving
// full conversation fidelity with zero additional inference cost.
//
// In addition to per-turn chunks, a session-level rolling window chunk is
// stored every N turns (default 5) to improve multi-hop retrieval.
//
// Usage:
//
//	store := NewMemoryChunkStore(milvusStore)
//	err := store.ProcessResponse(ctx, sessionID, userID, userMsg, assistantMsg)
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

var thinkClosedPattern = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)
var thinkUnclosedPattern = regexp.MustCompile(`(?s)<think>.*`)

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

// ProcessResponse stores the current conversation turn. Delegates to
// ProcessResponseWithHistory with nil history for backward compatibility.
func (e *MemoryExtractor) ProcessResponse(
	ctx context.Context,
	sessionID string,
	userID string,
	userMessage string,
	assistantResponse string,
) error {
	return e.ProcessResponseWithHistory(ctx, sessionID, userID, userMessage, assistantResponse, nil)
}

// ProcessResponseWithHistory stores the current conversation turn directly in
// the vector store (Q: ... / A: ...) and, every N turns, a session-level
// rolling window chunk that concatenates recent turns for multi-hop retrieval.
//
// The session chunk is built from the conversation history (obtained from the
// Response API's previous_response_id chain) plus the current turn. This gives
// retrieval a broader context window to match multi-hop queries that span
// multiple turns.
//
// Low-entropy turns are skipped for per-turn storage but still counted toward
// the session window trigger.
func (e *MemoryExtractor) ProcessResponseWithHistory(
	ctx context.Context,
	_ string, // sessionID (unused, kept for interface compatibility)
	userID string,
	userMessage string,
	assistantResponse string,
	history []Message,
) error {
	if e == nil || e.store == nil || !e.store.IsEnabled() {
		logging.Infof("Memory chunk store: SKIPPED - store not enabled (store=%v)", e != nil && e.store != nil)
		return nil
	}

	assistantResponse = StripThinkTags(assistantResponse)

	if userMessage == "" && assistantResponse == "" {
		logging.Debugf("Memory chunk store: SKIPPED - empty turn")
		return nil
	}

	startTime := time.Now()
	turnStored := false
	status := "success"
	defer func() {
		duration := time.Since(startTime).Seconds()
		count := 1
		if turnStored {
			count = 1
		}
		RecordMemoryExtraction(status, duration, count, "episodic")
	}()

	// --- Per-turn chunk ---
	if !isLowEntropy(userMessage, assistantResponse) {
		chunk := formatTurnChunk(userMessage, assistantResponse)
		sanitized, err := sanitizeMemoryContent(chunk)
		if err != nil {
			logging.Debugf("Memory chunk store: REJECTED - %v (user=%s)", err, userID)
		} else {
			mem := &Memory{
				ID:         generateMemoryID(),
				Type:       MemoryTypeEpisodic,
				Content:    sanitized,
				UserID:     userID,
				Source:     "conversation",
				CreatedAt:  time.Now(),
				Importance: 0.5,
			}
			if err := e.store.Store(ctx, mem); err != nil {
				status = "error"
				return fmt.Errorf("failed to store conversation chunk: %w", err)
			}
			turnStored = true
			logging.Infof("Memory chunk store: stored turn for user=%s (len=%d)", userID, len(sanitized))
		}
	} else {
		logging.Debugf("Memory chunk store: SKIPPED low-entropy turn for user=%s", userID)
	}

	// --- Session-level rolling window chunk ---
	e.maybeStoreSessionChunk(ctx, userID, userMessage, assistantResponse, history)

	return nil
}

// maybeStoreSessionChunk stores a session-level chunk every `stride` turns.
// Each chunk covers the last `windowSize` turns, creating overlapping windows
// (stride < windowSize) so facts near window boundaries still co-occur in at
// least one chunk -- critical for multi-hop retrieval.
func (e *MemoryExtractor) maybeStoreSessionChunk(
	ctx context.Context,
	userID string,
	userMessage string,
	assistantResponse string,
	history []Message,
) {
	if len(history) == 0 {
		return
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
		return
	}
	// Fire on every stride-th turn (once we have enough turns for a window)
	if totalTurns%stride != 0 {
		return
	}

	// Build session chunk from the last windowSize turns of history + current
	sessionChunk := buildSessionChunk(history, userMessage, assistantResponse, windowSize)
	if sessionChunk == "" {
		return
	}

	sanitized, err := sanitizeMemoryContent(sessionChunk)
	if err != nil {
		logging.Debugf("Memory session chunk: REJECTED - %v (user=%s)", err, userID)
		return
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

	if err := e.store.Store(ctx, mem); err != nil {
		logging.Warnf("Memory session chunk: failed to store for user=%s: %v", userID, err)
		return
	}

	logging.Infof("Memory session chunk: stored %d-turn window (stride=%d) for user=%s (len=%d)",
		windowSize, stride, userID, len(sanitized))
}

// countTurns counts user turns in a message history. Each user message is one turn.
func countTurns(history []Message) int {
	n := 0
	for _, m := range history {
		if m.Role == "user" {
			n++
		}
	}
	return n
}

// buildSessionChunk concatenates the last windowSize turns from history plus
// the current turn into a single multi-turn context chunk.
func buildSessionChunk(history []Message, userMsg, assistantResp string, windowSize int) string {
	// Collect the tail of history: last (windowSize-1) user-assistant pairs
	// plus the current turn = windowSize total turns.
	var pairs []string

	// Walk history backwards to find the last (windowSize-1) user messages
	// and their following assistant responses.
	type turn struct{ user, assistant string }
	var turns []turn
	for i := len(history) - 1; i >= 0 && len(turns) < windowSize-1; i-- {
		if history[i].Role == "user" {
			t := turn{user: history[i].Content}
			// Look ahead for the assistant response
			if i+1 < len(history) && history[i+1].Role == "assistant" {
				t.assistant = StripThinkTags(history[i+1].Content)
			}
			turns = append(turns, t)
		}
	}

	// Reverse to chronological order
	for i, j := 0, len(turns)-1; i < j; i, j = i+1, j-1 {
		turns[i], turns[j] = turns[j], turns[i]
	}

	// Format each historical turn
	for _, t := range turns {
		pairs = append(pairs, formatTurnChunk(t.user, t.assistant))
	}

	// Append current turn
	pairs = append(pairs, formatTurnChunk(userMsg, assistantResp))

	return strings.Join(pairs, "\n---\n")
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
	return fmt.Sprintf("mem_%d", time.Now().UnixNano())
}
