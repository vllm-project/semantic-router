package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Chat client of a Chat upstream gets the nested usage details the buffered
// reply re-encodes, never the provider's flat cached_tokens and
// reasoning_tokens, however the upstream splits its reads.
func TestChatStreamClientGetsNestedUsageDetailsForFlatProviderFields(t *testing.T) {
	content := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[{"index":0,"delta":{"content":"ok"},"logprobs":null,"finish_reason":"stop"}]}` + "\n\n"
	usage := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"cached_tokens":1,"reasoning_tokens":1}}` + "\n\n"
	want := `data: {"choices":[],"created":1,"id":"c1","model":"provider-model","object":"chat.completion.chunk","usage":{"completion_tokens":2,"completion_tokens_details":{"reasoning_tokens":1},"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":1},"total_tokens":5}}` + "\n\n"
	done := "data: [DONE]\n\n"
	for name, reads := range map[string][]string{
		"one read":           {content + usage + done},
		"usage split":        {content + usage[:40], usage[40:], done},
		"usage and done end": {content, usage + done},
	} {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, content+want+done, chatUpstreamClientBody(t, llmprotocol.OpenAIChatV1, false, true, reads))
		})
	}
}
