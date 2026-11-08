package extproc

import (
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const magistralOpening = `data: {"id":"chatcmpl-ad7f2d419b6862ca","object":"chat.completion.chunk","created":1791748219,"model":"mistralai/Magistral-Small-2509","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null,"prompt_text":null}

data: {"id":"chatcmpl-ad7f2d419b6862ca","object":"chat.completion.chunk","created":1791748219,"model":"mistralai/Magistral-Small-2509","choices":[{"index":0,"delta":{"content":"Hi! Here are five short sentences about the ocean.","reasoning":"Okay, the user wants me to say hi first."},"logprobs":null,"finish_reason":null,"token_ids":null}]}

`

// When the read after the opening delta ends the stream, a Chat client of a
// Chat upstream should still get the delta's text once.
func TestChatOpeningDeltaReachesChatClientOnce(t *testing.T) {
	done := "data: [DONE]\n\n"
	stringCode := `data: {"error":{"message":"mock provider stream failed","type":"server_error","param":null,"code":"provider_overloaded"}}` + "\n\n"
	vllmError := `data: {"error": {"message": "Internal server error", "type": "InternalServerError", "param": null, "code": 500}}` + "\n\n"
	for _, tc := range []struct {
		name  string
		reads []string
	}{
		{"stream ends in the next read", []string{magistralOpening, ""}},
		{"string-code error in the next read", []string{magistralOpening, stringCode + done}},
		{"vLLM error in the next read", []string{magistralOpening, vllmError + done}},
		{"malformed frame in the next read", []string{magistralOpening, "data: {not json\n\n"}},
		{"error in the same read", []string{magistralOpening + stringCode + done}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := openingDeltaChatClientBody(t, tc.reads)
			require.Equal(t, 1, strings.Count(body, "Hi! Here are five short sentences"), "answer:\n%s", body)
			require.Equal(t, 1, strings.Count(body, "the user wants me to say hi first"), "reasoning:\n%s", body)
		})
	}
}

// openingDeltaChatClientBody drives the response header and body handlers, in
// that order, and returns what a Chat client of a Chat upstream receives.
func openingDeltaChatClientBody(t *testing.T, reads []string) string {
	t.Helper()
	router := &OpenAIRouter{}
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, true)
	ctx.LooperRequest = false
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.OpenAIChatV1
	_, err := router.handleResponseHeaders(looperTransportHeaders(200, "text/event-stream"), ctx)
	require.NoError(t, err)
	var out strings.Builder
	for i, read := range reads {
		response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(read), EndOfStream: i == len(reads)-1},
		}, ctx)
		require.NoError(t, err)
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			read = string(mutation.GetBody())
		}
		out.WriteString(read)
	}
	return out.String()
}
