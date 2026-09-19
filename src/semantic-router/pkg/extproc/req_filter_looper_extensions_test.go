package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestLooperTraceExtensionsCrossResponseBoundary(t *testing.T) {
	for _, field := range []string{"flow", "fusion", "reasoning_mom_responses"} {
		for _, streaming := range []bool{false, true} {
			name := field + "/buffered"
			if streaming {
				name = field + "/streaming"
			}
			t.Run(name, func(t *testing.T) {
				const answer = `{"id":"c1","object":"chat.completion","created":1,"model":"judge","choices":[{"index":0,"message":{"role":"assistant","content":"Answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
				var fields map[string]json.RawMessage
				if err := json.Unmarshal([]byte(answer), &fields); err != nil {
					t.Fatal(err)
				}
				fields[field] = json.RawMessage(`{"summary":"router trace"}`)
				snapshot, err := json.Marshal(fields)
				if err != nil {
					t.Fatal(err)
				}
				// A provider body still has to satisfy the strict public wire contract.
				if _, err := protocolcodec.NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, snapshot, nil); err == nil {
					t.Fatal("upstream extension unexpectedly accepted")
				}
				response := &looper.Response{Body: snapshot, Model: "judge", ContentType: "application/json"}
				if streaming {
					response.ContentType = "text/event-stream"
					response.Body = []byte("data: " + `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"judge","choices":[{"index":0,"delta":{"role":"assistant","content":"Answer"},"finish_reason":"stop"}],"` + field + `":{"summary":"router trace"}}` + "\n\ndata: [DONE]\n\n")
					response.BufferedBody = snapshot
				}
				ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmWorkflows}}}
				out, semantic, _, err := (&OpenAIRouter{}).prepareLooperResponse(response, ctx)
				if err != nil {
					t.Fatal(err)
				}
				if out.GetImmediateResponse().GetStatus().GetCode() != 200 || semantic.Model != "judge" {
					t.Fatalf("invalid translated response: %+v", out)
				}
				if !strings.Contains(string(out.GetImmediateResponse().GetBody()), `"`+field+`"`) {
					t.Fatal("public trace was lost")
				}
			})
		}
	}
}
