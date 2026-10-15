package extproc

import (
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vLLM sends its error event and then [DONE], in one read or in two. Every
// client gets vLLM's message either way. Messages and Responses clients, and
// Chat clients when [DONE] shares the error's read, get the Router's error
// frame with the status as a string code where the protocol has one. When
// [DONE] comes in a later read, a Chat client gets vLLM's event as sent, as
// the Router already does for an error with a string code. Which form it
// should get there is not settled, so that case checks only for vLLM's
// message, once, and no [DONE].
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
			require.Equal(t, want[client], vllmStreamClientBody(t, client, []string{event + done}), "one read")
			twoReads := vllmStreamClientBody(t, client, []string{event, done})
			if client == llmprotocol.OpenAIChatV1 {
				require.Equal(t, 1, strings.Count(twoReads, "Grammar error: bad regex"), "two reads")
				require.NotContains(t, twoReads, "[DONE]", "two reads")
				return
			}
			require.Equal(t, want[client], twoReads, "two reads")
		})
	}
}

// vllmStreamClientBody runs a Chat upstream's chunks through the public
// response header and body handlers and returns what the client receives.
func vllmStreamClientBody(t *testing.T, client llmprotocol.WireFormat, chunks []string) string {
	t.Helper()
	router := &OpenAIRouter{}
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, true)
	ctx.LooperRequest = false // the public client path, not the internal Looper hop
	ctx.SourceFormat = client
	_, err := router.handleResponseHeaders(looperTransportHeaders(200, "text/event-stream"), ctx)
	require.NoError(t, err)
	var out strings.Builder
	for i, chunk := range chunks {
		response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(chunk), EndOfStream: i == len(chunks)-1},
		}, ctx)
		require.NoError(t, err)
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			chunk = string(mutation.GetBody())
		}
		out.WriteString(chunk)
	}
	return out.String()
}
