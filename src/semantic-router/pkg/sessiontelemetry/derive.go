package sessiontelemetry

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// ChatMessage is a minimal role/content pair for session correlation (matches extproc.ChatCompletionMessage fields used for hashing).
type ChatMessage struct {
	Role    string
	Content string
}

// DeriveChatCompletionsSessionID matches extproc.deriveSessionIDFromMessages so multi-turn chat sessions correlate with memory.
func DeriveChatCompletionsSessionID(messages []ChatMessage, userID string) string {
	var builder strings.Builder
	builder.WriteString(userID)
	builder.WriteString(":")

	for _, msg := range messages {
		if msg.Role == "user" {
			content := msg.Content
			if len(content) > 100 {
				content = content[:100]
			}
			builder.WriteString(content)
			break
		}
	}

	hash := sha256.Sum256([]byte(builder.String()))
	return "cc-" + hex.EncodeToString(hash[:])[:16]
}

// ChatTurnNumber is 1-based: next assistant reply index for the current request.
func ChatTurnNumber(messages []ChatMessage) int {
	assistants := 0
	for _, m := range messages {
		if m.Role == "assistant" {
			assistants++
		}
	}
	return assistants + 1
}
