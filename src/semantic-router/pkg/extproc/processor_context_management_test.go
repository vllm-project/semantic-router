package extproc

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code sends a top-level context_management directive on every
// /v1/messages request. It asks the upstream Anthropic API to trim stale
// content from the billed prompt, so it must survive the production dispatch
// seam to an Anthropic-format backend. A field can survive the codec in
// isolation and still be lost here, where the Router either replays the client
// bytes or re-encodes the neutral request; both routes must carry it through.

// dispatchContextManagementBody is a minimal Anthropic Messages request with the
// directive embedded, used to drive prepareProtocolRequest at the ExtProc seam.
const dispatchContextManagementBody = `{"model":"public-model","max_tokens":64,` +
	`"messages":[{"role":"user","content":"hello"}],` +
	`"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}}`

// The unmutated request keeps decode-time Generation 1 and stays byte-replay
// eligible, so dispatch forwards the client bytes verbatim. This is a green
// guard: the directive is preserved trivially by replay.
func TestExtProcDispatchReplaysContextManagementToAnthropicBackend(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestID:    "request_context_management_replay",
		TraceContext: t.Context(),
	}
	request, immediate := router.prepareProtocolRequest([]byte(dispatchContextManagementBody), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("request carrying context_management was rejected: request=%+v immediate=%+v", request, immediate)
	}

	dispatch, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	assertDispatchCarriesContextManagement(t, dispatch)
}

// Mutating the neutral request bumps Generation, so dispatch re-encodes from the
// neutral model instead of replaying the client bytes. The directive survives
// only if the Router models and re-emits it, which is the behavior the fix adds.
func TestExtProcDispatchReEncodesContextManagementToAnthropicBackend(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestID:    "request_context_management_reencode",
		TraceContext: t.Context(),
	}
	request, immediate := router.prepareProtocolRequest([]byte(dispatchContextManagementBody), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("request carrying context_management was rejected: request=%+v immediate=%+v", request, immediate)
	}
	// A routing decision rewrites the model and bumps the generation, which
	// forces the dispatch encode off the byte-replay path.
	request.Model = "routed-model"
	request.Generation++

	dispatch, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	assertDispatchCarriesContextManagement(t, dispatch)
}

func assertDispatchCarriesContextManagement(t *testing.T, dispatch []byte) {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(dispatch, &object); err != nil {
		t.Fatalf("dispatch body is not a JSON object: %v\n%s", err, dispatch)
	}
	want := json.RawMessage(`{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`)
	var got, expected any
	if err := json.Unmarshal(object["context_management"], &got); err != nil {
		t.Fatalf("dispatch dropped context_management for the Anthropic backend: %s", dispatch)
	}
	if err := json.Unmarshal(want, &expected); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("context_management changed through dispatch: %s", dispatch)
	}
}
