package extproc

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Both requests of a captured Codex CLI tool loop must pass Router ingress and
// reach a Chat backend: the first turn, and the turn carrying the tool result.
func TestExtProcDispatchesCodexToolLoopToChatBackend(t *testing.T) {
	raw, err := os.ReadFile("../protocolcodec/testdata/clients/codex-cli-0.156.1-tool-loop.json")
	if err != nil {
		t.Fatal(err)
	}
	var turns []json.RawMessage
	if err := json.Unmarshal(raw, &turns); err != nil {
		t.Fatal(err)
	}
	for turn, body := range turns {
		router := &OpenAIRouter{}
		ctx := &RequestContext{
			SourceFormat: llmprotocol.OpenAIResponsesV1,
			RequestID:    "request_codex_tool_loop",
			TraceContext: t.Context(),
		}
		request, immediate := router.prepareProtocolRequest(body, ctx)
		if immediate != nil || request == nil {
			t.Fatalf("turn %d rejected at ingress: %+v", turn+1, ctx.ImmediateProtocolError)
		}
		if changed, err := router.materializeResponseObjectContext(request, ctx); err != nil {
			t.Fatal(err)
		} else if changed {
			request.Generation++
		}
		request.Model = "routed-model"
		request.Generation++
		ctx.TargetFormat = llmprotocol.OpenAIChatV1

		dispatch, err := router.encodeDispatchRequest(ctx)
		if err != nil {
			t.Fatalf("turn %d dispatch: %v", turn+1, err)
		}
		var chat map[string]json.RawMessage
		if err := json.Unmarshal(dispatch, &chat); err != nil {
			t.Fatal(err)
		}
		if string(chat["prompt_cache_key"]) != `"00000000-0000-4000-8000-000000000004"` {
			t.Fatalf("turn %d dispatch lost prompt_cache_key: %s", turn+1, dispatch)
		}
		for _, field := range []string{"client_metadata", "include", "store"} {
			if _, ok := chat[field]; ok {
				t.Fatalf("turn %d dispatch forwarded %s: %s", turn+1, field, dispatch)
			}
		}
		dropped := map[string]bool{}
		for _, diagnostic := range ctx.ProtocolDiagnostics {
			if diagnostic.Action == llmprotocol.DiagnosticDropped {
				dropped[diagnostic.Field] = true
			}
		}
		if !dropped["client_metadata"] || !dropped["include"] {
			t.Fatalf("turn %d did not report the dropped fields: %+v", turn+1, ctx.ProtocolDiagnostics)
		}
	}
}
