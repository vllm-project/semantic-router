package protocolcodec

import (
	"bytes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ValidateEncodedResponse enforces limits on final client-visible bytes, after
// any trusted transport extension is rendered. SSE delimiters count toward both
// the complete-body limit and each frame limit, just as on provider input.
func (engine *Engine) ValidateEncodedResponse(body []byte, streaming bool) error {
	if len(body) > engine.policy.Limits.BodyBytes {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "response_body_limit", "encoded response exceeds the configured body limit", nil)
	}
	if !streaming {
		return nil
	}
	for len(body) > 0 {
		end, complete := completeSSEFrame(body)
		if !complete {
			if len(bytes.TrimSpace(body)) == 0 {
				return nil
			}
			end = len(body)
		}
		if err := validateSSEFrameBytes(body[:end], engine.policy.Limits.SSEFrameBytes); err != nil {
			return err
		}
		body = body[end:]
	}
	return nil
}

// NewChatPublicStreamFilter shares this engine's configured frame limit, the
// output limit of the request whose stream it filters and the configured JSON
// depth.
func (engine *Engine) NewChatPublicStreamFilter(includeUsage bool) *ChatUsageStreamFilter {
	filter := NewChatPublicStreamFilter(engine.policy.Limits.SSEFrameBytes, includeUsage, engine.policy.Limits.JSONDepth)
	filter.outputLimit = engine.policy.RequestOutputLimit
	return filter
}
