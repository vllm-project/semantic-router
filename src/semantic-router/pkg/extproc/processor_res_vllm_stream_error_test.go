package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vLLM sends its error event and then [DONE], in one read or in two. Either
// way every client gets the Router's error frame with vLLM's message and the
// status as a string code where the protocol has one (#5107).
func TestVLLMStreamErrorReachesClientInOneOrTwoReads(t *testing.T) {
	event := `data: {"error": {"message": "Grammar error: bad regex", "type": "BadRequestError", "param": null, "code": 400}}` + "\n\n"
	done := "data: [DONE]\n\n"
	want := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        `data: {"error":{"type":"invalid_request_error","code":"400","message":"Grammar error: bad regex","param":null}}` + "\n\n",
		llmprotocol.OpenAIResponsesV1:   "event: error\n" + `data: {"type":"error","code":"400","message":"Grammar error: bad regex","param":null,"sequence_number":0}` + "\n\n",
		llmprotocol.AnthropicMessagesV1: "event: error\n" + `data: {"type":"error","error":{"type":"invalid_request_error","message":"Grammar error: bad regex"}}` + "\n\n",
	}
	for _, client := range extProcMatrixFormats {
		t.Run(string(client), func(t *testing.T) {
			require.Equal(t, want[client], chatUpstreamClientBody(t, client, false, true, []string{event + done}), "one read")
			require.Equal(t, want[client], chatUpstreamClientBody(t, client, false, true, []string{event, done}), "two reads")
		})
	}
}
