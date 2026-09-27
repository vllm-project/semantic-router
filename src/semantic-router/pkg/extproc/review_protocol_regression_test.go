package extproc

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// A response-policy body warning renders the client response a second time.
// That render must not report the same translation loss a second time.
func TestReviewBufferedPolicyBodyRewriteReportsCacheWarningOnce(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat:          llmprotocol.AnthropicMessagesV1,
		TargetFormat:          llmprotocol.OpenAIChatV1,
		RequestModel:          "public-model",
		TraceContext:          t.Context(),
		HallucinationDetected: true,
		VSRSelectedDecision:   decisionWithHallucinationActions("body", "header"),
	}
	const reason = "Messages requires numeric cache buckets; the unreported bucket is zero-filled for representation, while settlement retains unknown usage"
	counter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "openai", "dropped", reason)
	before := testutil.ToFloat64(counter)
	body := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":1}}}`)

	response := router.handleNonStreamingResponseBody(body, ctx, 0)
	common := response.GetResponseBody().GetResponse()
	if common == nil {
		t.Fatalf("buffered response failed: %+v", response)
	}
	if got := string(common.GetBodyMutation().GetBody()); !strings.Contains(got, "[Hallucination Warning]") {
		t.Fatalf("response-policy body rewrite did not run: %q", got)
	}
	warning := protocolDiagnosticsHeader(common.GetHeaderMutation())
	if got := strings.Count(warning, ";usage.cache"); got != 1 {
		t.Fatalf("usage.cache appears %d times after body rewrite: %q", got, warning)
	}
	if got := testutil.ToFloat64(counter) - before; got != 1 {
		t.Fatalf("usage.cache metric increased by %v, want 1", got)
	}
}

func TestReviewAzureTransportErrorReportsLateDiagnostic(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat:       llmprotocol.AnthropicMessagesV1,
		TargetFormat:       llmprotocol.OpenAIChatV1,
		ResponseVendor:     llmprotocol.ResponseVendorAzure,
		UpstreamStatusCode: 429,
		RequestID:          "request_1",
		RequestModel:       "public-model",
	}
	const reason = "provider vendor extension field is not part of the canonical response contract"
	counter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "openai", "dropped", reason)
	before := testutil.ToFloat64(counter)
	body := []byte(`{"error":{"type":"invalid_request_error","code":"content_filter","message":"blocked","param":null,"innererror":{"content_filter_result":{"hate":{"filtered":true}}}}}`)

	response := router.handleUpstreamTransportError(body, ctx)
	common := response.GetResponseBody().GetResponse()
	if common == nil || common.GetBodyMutation() == nil {
		t.Fatalf("transport error was not translated: %+v", response)
	}
	warning := protocolDiagnosticsHeader(common.GetHeaderMutation())
	if got := strings.Count(warning, ";error.innererror"); got != 1 {
		t.Fatalf("error.innererror appears %d times in warning header: %q", got, warning)
	}
	if got := testutil.ToFloat64(counter) - before; got != 1 {
		t.Fatalf("transport diagnostic metric increased by %v, want 1", got)
	}
}

func TestReviewWarningHeaderEncodesControlBytes(t *testing.T) {
	diagnostics := llmprotocol.Diagnostics{{
		Action: llmprotocol.DiagnosticDropped,
		Reason: "vendor extension",
		Field:  "vendor\x00\x1f\x7f;%",
	}}
	value, included := formatProtocolDiagnostics(diagnostics)
	if included != 1 {
		t.Fatalf("included %d diagnostics, want 1", included)
	}
	for _, escaped := range []string{"%00", "%1F", "%7F", "%3B", "%25"} {
		if !strings.Contains(value, escaped) {
			t.Errorf("warning header does not encode %s: %q", escaped, value)
		}
	}
	for _, b := range []byte(value) {
		if b < 0x20 || b == 0x7f {
			t.Fatalf("warning header contains a control byte 0x%02x: %q", b, value)
		}
	}
}

func TestReviewBufferedDiagnosticTruncationCountsOnlyReportedEntries(t *testing.T) {
	const reason = "review bounded diagnostics count"
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
	}
	for i := 0; i < 100; i++ {
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{
			Action: llmprotocol.DiagnosticDropped,
			Reason: reason,
			Field:  "vendor." + strings.Repeat("x", 80),
		})
	}
	counter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "openai", "dropped", reason)
	before := testutil.ToFloat64(counter)
	warning, ok := recordBufferedProtocolDiagnostics(ctx, 0)
	if !ok {
		t.Fatal("buffered diagnostics were not reported")
	}
	formatted, included := formatProtocolDiagnostics(ctx.ProtocolDiagnostics)
	if included == 0 || included >= len(ctx.ProtocolDiagnostics) || warning != formatted ||
		!strings.Contains(warning, "diagnostics_truncated") {
		t.Fatalf("warning was not bounded: included=%d warning=%q", included, warning)
	}
	if got := testutil.ToFloat64(counter) - before; got != float64(included) {
		t.Fatalf("truncated diagnostics metric increased by %v, want %d", got, included)
	}
}
