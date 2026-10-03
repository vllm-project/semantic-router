package extproc

import (
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
