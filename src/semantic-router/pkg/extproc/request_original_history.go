package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"

// originalConversation decodes the original-history snapshot at most once per
// request and memoizes it. The second result is false when no snapshot was
// captured. The returned value is shared and read-only by contract:
// ConversationHistory holds slices, so callers must not mutate it.
func originalConversation(ctx *RequestContext) (contextcompression.ConversationHistory, bool) {
	if ctx.OriginalContextHistory == nil {
		return contextcompression.ConversationHistory{}, false
	}
	if !ctx.originalConversationLoaded {
		decoded := ctx.OriginalContextHistory.Conversation()
		ctx.originalConversation = &decoded
		ctx.originalConversationLoaded = true
	}
	return *ctx.originalConversation, true
}
