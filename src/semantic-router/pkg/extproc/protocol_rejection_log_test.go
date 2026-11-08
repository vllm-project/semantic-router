package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func observeRejections(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.DebugLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	return logs
}

func rejectionFields(t *testing.T, logs *observer.ObservedLogs) map[string]interface{} {
	t.Helper()
	entries := logs.FilterMessage("protocol_request_rejected").All()
	if len(entries) != 1 {
		t.Fatalf("protocol_request_rejected events = %d, want 1", len(entries))
	}
	return entries[0].ContextMap()
}

// Claude Code's first call carries tools.defer_loading, which the Anthropic
// ingress refuses before routing, so the request never reaches replay.
func TestIngressRejectionLogsCodeAndMessage(t *testing.T) {
	logs := observeRejections(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string), RequestID: "req-1", SourceFormat: llmprotocol.AnthropicMessagesV1}
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"name":"Bash","input_schema":{"type":"object"},"defer_loading":true}]}`

	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body)},
	}, ctx)
	if err != nil {
		t.Fatalf("HandleRequestBody returned error: %v", err)
	}
	response = router.encodeImmediateResponseForClient(response, ctx)
	if response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_BadRequest {
		t.Fatalf("expected a 400, got %v", response.GetImmediateResponse().GetStatus())
	}
	fields := rejectionFields(t, logs)
	if fields["request_id"] != "req-1" || fields["code"] != "unsupported_tools_defer_loading" || fields["status"] != int64(400) {
		t.Fatalf("rejection fields = %v", fields)
	}
	if message, _ := fields["message"].(string); message != "tools.defer_loading is not supported by the protocol-neutral request contract" {
		t.Fatalf("rejection message = %q", message)
	}
}

// Toward a Chat backend Claude Code's thinking display is refused by the
// dispatch capability check, a path that never passes through the ingress or
// the dispatch encoder.
func TestDispatchCapabilityRejectionLogsCodeAndMessage(t *testing.T) {
	logs := observeRejections(t)
	router := routingTestRouter("chat")
	body := `{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"adaptive","display":"updates"}}`
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.RequestID = "req-2"
	_, err = router.prepareProviderDispatch(&request, "chat", "", false, ctx)
	if err == nil {
		t.Fatal("expected the thinking display to be refused toward a Chat backend")
	}
	response, converted := router.processBodyRoutingError(err, ctx)
	if !converted {
		t.Fatalf("refusal was not converted to an immediate response: %v", err)
	}
	router.encodeImmediateResponseForClient(response, ctx)
	fields := rejectionFields(t, logs)
	if fields["request_id"] != "req-2" || fields["code"] != "unsupported_reasoning_display" || fields["status"] != int64(400) {
		t.Fatalf("rejection fields = %v", fields)
	}
}
