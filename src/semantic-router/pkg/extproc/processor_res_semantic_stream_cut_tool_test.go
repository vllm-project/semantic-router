package extproc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
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

// A decoder copies a provider frame's code, message and parameter verbatim, and
// they can echo request content, so a provider-framed failure contributes only
// its category — even when it names a code the Router also authors.
func TestStreamFailureSummaryKeepsProviderTextOut(t *testing.T) {
	const providerText = "the user's secret file is /etc/passwd"
	for _, providerCode := range []string{"PRIVATE_REQUEST_CANARY", "stream_incomplete"} {
		failed := &llmprotocol.ProtocolError{
			Category:       llmprotocol.ErrorUpstreamUnavailable,
			Code:           providerCode,
			Message:        providerText,
			Cause:          fmt.Errorf("%s", providerText),
			ProviderFramed: true,
		}
		state := &semanticResponseStreamState{failed: failed}
		_, err := state.response()
		if err == nil {
			t.Fatal("a failed stream assembled a response")
		}
		if !strings.Contains(err.Error(), "category=upstream_unavailable") {
			t.Fatalf("the summary lost the category: %s", err.Error())
		}
		if strings.Contains(err.Error(), providerText) || strings.Contains(err.Error(), providerCode) {
			t.Fatalf("the summary retained provider text for %q: %s", providerCode, err.Error())
		}
	}
}

// A code the Router authored stays in the log with its diagnostic cause.
func TestStreamFailureSummaryNamesARouterDiagnosticCode(t *testing.T) {
	failed := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_tool_arguments",
		"upstream streamed tool arguments are not a JSON object",
		llmprotocol.ErrorDiagnostic("arguments length 15, shape plain text"))
	state := &semanticResponseStreamState{failed: failed}
	_, err := state.response()
	if err == nil {
		t.Fatal("a failed stream assembled a response")
	}
	for _, want := range []string{"category=upstream_unavailable", "code=invalid_stream_tool_arguments", "shape plain text"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("the summary lost %q: %s", want, err.Error())
		}
	}
}

// A parser error's text can name provider fields, so only a Router-authored
// ErrorDiagnostic is printed from the cause chain.
func TestStreamFailureSummaryKeepsProviderParserTextOut(t *testing.T) {
	const providerField = "PRIVATE_FIELD_CANARY"
	failed := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json",
		"upstream response JSON is invalid",
		fmt.Errorf("non-canonical JSON field: unknown field %q", providerField))
	state := &semanticResponseStreamState{failed: failed}
	_, err := state.response()
	if err == nil {
		t.Fatal("a failed stream assembled a response")
	}
	if !strings.Contains(err.Error(), "code=invalid_upstream_json") {
		t.Fatalf("the summary lost the Router code: %s", err.Error())
	}
	if strings.Contains(err.Error(), providerField) {
		t.Fatalf("the summary retained provider parser text: %s", err.Error())
	}
}

// A duplicate field name the provider chose reaches the codec's cause; the
// reconstruction-skipped summary still names only the Router diagnostic.
func TestStreamFailureSummaryKeepsADuplicateFieldNameOut(t *testing.T) {
	const canary = "PRIVATE_REQUEST_CANARY"
	payload := "data: {\"id\":\"response_1\",\"object\":\"chat.completion.chunk\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"" + canary + "\":1,\"" + canary + "\":2}\n\n"
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, pushErr := stream.Push([]byte(payload))
	var failed *llmprotocol.ProtocolError
	if !errors.As(pushErr, &failed) {
		t.Fatalf("duplicate provider field = %v, want a protocol error", pushErr)
	}
	state := &semanticResponseStreamState{failed: failed}
	_, err = state.response()
	if err == nil {
		t.Fatal("a failed stream assembled a response")
	}
	if !strings.Contains(err.Error(), "code=upstream_duplicate_json_field") {
		t.Fatalf("the summary lost the Router code: %s", err.Error())
	}
	if strings.Contains(err.Error(), canary) {
		t.Fatalf("the summary retained the provider field name: %s", err.Error())
	}
}
