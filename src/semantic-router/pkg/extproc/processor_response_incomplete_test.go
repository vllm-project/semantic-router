package extproc

import (
	"encoding/json"
	"testing"

	"github.com/alicebob/miniredis/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

func TestStoredResponseKeepsIncompleteDetails(t *testing.T) {
	for _, backend := range []string{"memory", "redis"} {
		t.Run(backend, func(t *testing.T) {
			store := incompleteResponseTestStore(t, backend)
			for _, tc := range []struct {
				name, status, reason string
				tool, stream         bool
			}{
				{name: "completed text", status: "completed"},
				{name: "completed tool", status: "completed", tool: true},
				{name: "output-limited text", status: "incomplete", reason: "max_output_tokens"},
				{name: "output-limited tool", status: "incomplete", reason: "max_output_tokens", tool: true},
				{name: "filtered text", status: "incomplete", reason: "content_filter"},
				{name: "streamed completed text", status: "completed", stream: true},
				{name: "streamed output-limited text", status: "incomplete", reason: "max_output_tokens", stream: true},
				{name: "streamed filtered text", status: "incomplete", reason: "content_filter", stream: true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(store)}
					ctx := &RequestContext{
						SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
						Headers: map[string]string{}, TraceContext: t.Context(), RequestID: t.Name(), RequestModel: "m", UpstreamStatusCode: 200,
					}
					request, immediate := router.prepareProtocolRequest([]byte(`{"model":"m","input":"answer","tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{}}}]}`), ctx)
					if request == nil || immediate != nil {
						t.Fatalf("request rejected: %v", immediate)
					}
					body := incompleteResponseFixture(t, tc.status, tc.reason, tc.tool)
					engine := protocolcodec.NewBuiltinEngine()
					source, _, _, sourceErr := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
					if sourceErr != nil {
						t.Fatalf("invalid source response: %v", sourceErr)
					}
					if tc.stream {
						streamingClient(ctx, llmprotocol.OpenAIResponsesV1)
						stream, _, streamErr := engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, source, llmprotocol.StreamContext{ResponseID: "resp_provider", PublicModel: "m"})
						if streamErr != nil {
							t.Fatal(streamErr)
						}
						router.handleSemanticStreamingResponseBody(stream, true, ctx)
						if ctx.StreamingAborted || ctx.SemanticResponse == nil {
							t.Fatal("live response stream failed")
						}
					} else {
						reply := router.handleNonStreamingResponseBody(body, ctx, 0, false)
						if reply.GetImmediateResponse() != nil || ctx.SemanticResponse == nil {
							t.Fatalf("live response failed: %v", reply)
						}
					}
					live, liveErr := router.encodeClientResponse(*ctx.SemanticResponse, ctx)
					if liveErr != nil {
						t.Fatal(liveErr)
					}
					assertResponseIncompleteDetails(t, live, tc.status, tc.reason)
					router.persistResponseObject(ctx)
					get, getErr := router.ResponseAPIFilter.HandleGetResponse(t.Context(), ctx.SemanticResponse.ID)
					if getErr != nil || get.GetImmediateResponse().GetStatus().GetCode() != 200 {
						t.Fatalf("stored GET failed: %v %v", get, getErr)
					}
					retrieved := get.GetImmediateResponse().GetBody()
					assertResponseIncompleteDetails(t, retrieved, tc.status, tc.reason)
					if _, _, _, decodeErr := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, retrieved); decodeErr != nil {
						t.Fatalf("stored object is invalid: %v", decodeErr)
					}
				})
			}
		})
	}
}

func incompleteResponseTestStore(t *testing.T, backend string) responsestore.ResponseStore {
	t.Helper()
	var store responsestore.ResponseStore
	var err error
	if backend == "redis" {
		server := miniredis.RunT(t)
		store, err = responsestore.NewRedisStore(responsestore.StoreConfig{
			Enabled: true, Redis: responsestore.RedisStoreConfig{Address: server.Addr()},
		})
	} else {
		store, err = responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func incompleteResponseFixture(t *testing.T, status, reason string, tool bool) []byte {
	t.Helper()
	output := json.RawMessage(`[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"answer","annotations":[]}]}]`)
	if tool {
		arguments := "{}"
		if reason == "max_output_tokens" {
			arguments = `{"city":`
		}
		encoded, err := json.Marshal([]map[string]any{{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": arguments, "status": status}})
		if err != nil {
			t.Fatal(err)
		}
		output = encoded
	}
	wire := map[string]any{"id": "resp_provider", "object": "response", "model": "m", "status": status, "output": output}
	if reason != "" {
		wire["incomplete_details"] = map[string]string{"reason": reason}
	}
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func assertResponseIncompleteDetails(t *testing.T, body []byte, status, reason string) {
	t.Helper()
	var response responseapi.ResponseAPIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatal(err)
	}
	if response.Status != status {
		t.Fatalf("status %q, want %q: %s", response.Status, status, body)
	}
	if reason == "" {
		if response.IncompleteDetails != nil {
			t.Fatalf("completed response acquired incomplete details: %s", body)
		}
	} else if response.IncompleteDetails == nil || response.IncompleteDetails.Reason != reason {
		t.Fatalf("incomplete reason %q not retained: %s", reason, body)
	}
}
