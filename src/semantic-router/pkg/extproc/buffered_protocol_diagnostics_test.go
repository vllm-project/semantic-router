package extproc

import (
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestBufferedResponseDiagnosticsReachBodyHeaderAndMetrics(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	request := testNeutralRequest(model, "hello")
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	if _, err := router.prepareProviderDispatch(request, model, "", false, ctx); err != nil {
		t.Fatal(err)
	}
	const earlyReason = "already known before response headers"
	const lateReason = "structured refusal detail has no neutral representation"
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{
		Source: llmprotocol.AnthropicMessagesV1, Target: llmprotocol.AnthropicMessagesV1,
		Action: llmprotocol.DiagnosticDropped, Field: "early_test", Reason: earlyReason,
	})
	earlyCounter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "anthropic", "dropped", earlyReason)
	lateCounter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "anthropic", "dropped", lateReason)
	earlyBefore := testutil.ToFloat64(earlyCounter)
	lateBefore := testutil.ToFloat64(lateCounter)

	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"}, {Key: "content-type", Value: "application/json"},
		}},
	}}
	headerResponse, err := router.handleResponseHeaders(responseHeaders, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got := protocolDiagnosticsHeader(headerResponse.GetResponseHeaders().GetResponse().GetHeaderMutation()); !strings.Contains(got, "early_test") {
		t.Fatalf("early diagnostic missing from response header phase: %q", got)
	}
	body := []byte(`{"id":"msg1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"stop_details":{"type":"refusal"},"usage":{"input_tokens":1,"output_tokens":1}}`)
	bodyResponse, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
	}, ctx)
	if err != nil || bodyResponse.GetResponseBody() == nil {
		t.Fatalf("buffered response failed: response=%v error=%v", bodyResponse, err)
	}
	warning := protocolDiagnosticsHeader(bodyResponse.GetResponseBody().GetResponse().GetHeaderMutation())
	if !strings.Contains(warning, "early_test") || !strings.Contains(warning, ";stop_details") {
		t.Fatalf("body header omitted early or late diagnostic: %q", warning)
	}
	if len(warning) > lossinessHeaderSizeLimit || testutil.ToFloat64(earlyCounter) != earlyBefore+1 ||
		testutil.ToFloat64(lateCounter) != lateBefore+1 {
		t.Fatalf("diagnostic limit or accounting failed: warning=%q early=%v late=%v", warning,
			testutil.ToFloat64(earlyCounter)-earlyBefore, testutil.ToFloat64(lateCounter)-lateBefore)
	}
	for _, option := range bodyResponse.GetResponseBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.VSRProtocolWarnings &&
			option.GetAppendAction() != core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD {
			t.Fatal("late diagnostics must replace the early warning header")
		}
	}
}

func TestBufferedResponsePartialCacheWarningIsRecordedOnce(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
	}
	const reason = "Messages requires numeric cache buckets; the unreported bucket is zero-filled for representation, while settlement retains unknown usage"
	counter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "openai", "dropped", reason)
	before := testutil.ToFloat64(counter)
	body := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":1}}}`)
	response := router.handleNonStreamingResponseBody(body, ctx, 0)
	if response.GetResponseBody() == nil {
		t.Fatalf("buffered response failed: %+v", response)
	}
	warning := protocolDiagnosticsHeader(response.GetResponseBody().GetResponse().GetHeaderMutation())
	if strings.Count(warning, ";usage.cache") != 1 {
		t.Fatalf("partial cache usage should emit one warning: %q", warning)
	}
	if got := testutil.ToFloat64(counter); got != before+1 {
		t.Fatalf("partial cache warning counted %v times, want 1", got-before)
	}
	if ctx.SemanticResponse == nil || ctx.SemanticResponse.Usage.InputCacheWrite.Value != nil {
		t.Fatalf("unreported cache writes must remain unknown for settlement: %+v", ctx.SemanticResponse)
	}
}

func protocolDiagnosticsHeader(mutation *ext_proc.HeaderMutation) string {
	for _, option := range mutation.GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.VSRProtocolWarnings {
			return string(option.GetHeader().GetRawValue()) + option.GetHeader().GetValue()
		}
	}
	return ""
}
