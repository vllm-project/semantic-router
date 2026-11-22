package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A call the output limit cut reaches the client as a successful stream, and
// it is never cached: a hit would give another request the cut call.
func TestSemanticStreamKeepsCutToolCallOutOfCache(t *testing.T) {
	chunk := func(delta, finish string) string {
		return `data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + finish + "}]}\n\n"
	}
	call := chunk(`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]}`, "null")
	ends := map[string]string{
		"cut":      chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"\"Par"}}]}`, `"length"`),
		"complete": chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}`, `"tool_calls"`),
	}
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		for name, end := range ends {
			t.Run(string(client)+"_"+name, func(t *testing.T) {
				mockCache, router, decision := statusCacheRouter()
				ctx := withSelectedDecision(&RequestContext{
					RequestID: "req-" + name, UpstreamStatusCode: 200,
					SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1,
					RequestModel: "public-model", RequestQuery: "weather",
					SemanticRequest: testNeutralRequest("public-model", "weather"),
					TraceContext:    t.Context(),
				}, decision)
				if err := router.ensureSemanticResponseStream(ctx); err != nil {
					t.Fatal(err)
				}
				pushExtProcStreamFixture(t, ctx, []byte(strings.Join([]string{call, end, "data: [DONE]\n\n"}, "")))
				router.finalizeSemanticStreamingResponse(ctx, nil)
				if ctx.StreamingAborted {
					t.Fatal("the stream did not end as a success")
				}
				if want := name == "complete"; mockCache.addEntryCalled != want {
					t.Fatalf("cache written = %v, want %v", mockCache.addEntryCalled, want)
				}
			})
		}
	}
}
