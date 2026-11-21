package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// routerResponseID is the id of an answer the Router makes itself, such as a
// fast_response, a block or a cache hit, derived from seed and the request.
// It has the shape the client's protocol gives a model's answer: the
// Responses object's public id or resp_, msg_ on Messages, chatcmpl- on Chat.
// The Responses encoder names the answer's items by their types.
func routerResponseID(ctx *RequestContext, seed string) string {
	digest := strings.TrimPrefix(llmprotocol.StableID(seed, ctx.RequestID), responseapi.ItemIDPrefix)
	switch ctx.SourceFormat {
	case llmprotocol.OpenAIResponsesV1:
		if public := responseObjectPublicID(ctx); public != "" {
			return public
		}
		return responseapi.ResponseIDPrefix + digest
	case llmprotocol.AnthropicMessagesV1:
		return "msg_" + digest
	}
	return "chatcmpl-" + digest
}
