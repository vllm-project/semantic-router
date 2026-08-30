package protocolcodec

import (
	"crypto/rand"
	"encoding/base64"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// streamUsageRequested is deliberately opt-in. Chat Completions omits its
// terminal usage-only chunk unless the client explicitly requests it. Router
// accounting may still force backend usage independently of this public option.
func streamUsageRequested(context llmprotocol.StreamContext) bool {
	return context.Options.IncludeUsage != nil && *context.Options.IncludeUsage
}

func streamObfuscationRequested(context llmprotocol.StreamContext) bool {
	return context.Options.IncludeObfuscation != nil && *context.Options.IncludeObfuscation
}

func newStreamObfuscation(context llmprotocol.StreamContext) (string, error) {
	if !streamObfuscationRequested(context) {
		return "", nil
	}
	// The public contract requires random padding but does not prescribe a
	// length. Six random bytes render as eight URL-safe characters, matching the
	// compact shape used by the official Chat examples without introducing
	// provider state into neutral events.
	buffer := make([]byte, 6)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buffer), nil
}
