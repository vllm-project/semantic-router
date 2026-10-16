package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// The reconstructed stream feeds cache, memory, and Router Replay through the
// same buffered encoder, which rejects stop_sequence without the matched text.
func TestSemanticStreamKeepsMatchedStopSequence(t *testing.T) {
	backends := map[llmprotocol.WireFormat]string{
		// vLLM reports the matched stop string in choices[].stop_reason.
		llmprotocol.OpenAIChatV1: strings.Join([]string{
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ALPHA BRAVO \"},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\",\"stop_reason\":\"CHARLIE\"}]}\n\n",
			"data: [DONE]\n\n",
		}, ""),
		llmprotocol.AnthropicMessagesV1: strings.Replace(
			string(extProcStreamFixture(llmprotocol.AnthropicMessagesV1)),
			`"stop_reason":"end_turn","stop_sequence":null`,
			`"stop_reason":"stop_sequence","stop_sequence":"CHARLIE"`, 1),
	}
	for backend, fixture := range backends {
		t.Run(string(backend)+"_backend", func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: backend,
				RequestModel: "public-model", TraceContext: t.Context(),
			}
			if err := (&OpenAIRouter{}).ensureSemanticResponseStream(ctx); err != nil {
				t.Fatal(err)
			}
			pushExtProcStreamFixture(t, ctx, []byte(fixture))
			semantic, err := ctx.SemanticStreamState.response()
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := protocolcodec.NewBuiltinEngine().EncodeResponse(llmprotocol.AnthropicMessagesV1, *semantic, llmprotocol.Envelope{})
			if err != nil {
				t.Fatalf("reconstructed stream does not re-encode: %v", err)
			}
			if !strings.Contains(string(encoded.Body), `"stop_reason":"stop_sequence","stop_sequence":"CHARLIE"`) {
				t.Fatalf("replayed response lost the matched stop sequence: %s", encoded.Body)
			}
		})
	}
}

// vLLM reports a matched stop string longer than 128 bytes as sent, and the
// request side accepts one, so the reply must reach the client either way.
func TestLongMatchedStopSequenceReachesClient(t *testing.T) {
	reason, _ := json.Marshal(strings.Repeat("x", 129))
	anthropicStop := `"stop_reason":"stop_sequence","stop_sequence":` + string(reason)

	t.Run("buffered", func(t *testing.T) {
		body := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":"BEGIN "},"finish_reason":"stop","stop_reason":` +
			string(reason) + `}],"usage":{"prompt_tokens":77,"completion_tokens":36,"total_tokens":113}}`)
		for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
			ctx := &RequestContext{
				SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1,
				RequestModel: "public-model", TraceContext: t.Context(),
			}
			response := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, ctx, 0)
			if response.GetImmediateResponse() != nil || response.GetResponseBody() == nil {
				t.Fatalf("%s client: the reply was replaced with an error: %+v", client, response.GetImmediateResponse())
			}
			if client == llmprotocol.AnthropicMessagesV1 &&
				!strings.Contains(string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()), anthropicStop) {
				t.Fatalf("Anthropic client lost the matched stop sequence: %+v", response.GetResponseBody())
			}
		}
	})

	t.Run("streamed", func(t *testing.T) {
		ctx := &RequestContext{
			SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
			RequestModel: "public-model", TraceContext: t.Context(),
		}
		if err := (&OpenAIRouter{}).ensureSemanticResponseStream(ctx); err != nil {
			t.Fatal(err)
		}
		fixture := strings.Join([]string{
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"BE\"},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"GIN \"},\"finish_reason\":\"stop\",\"stop_reason\":" + string(reason) + "}]}\n\n",
			"data: [DONE]\n\n",
		}, "")
		clientWire := pushExtProcStreamFixture(t, ctx, []byte(fixture))
		if !strings.Contains(clientWire.String(), anthropicStop) {
			t.Fatalf("streamed Anthropic client lost the matched stop sequence:\n%s", clientWire.String())
		}
	})
}
