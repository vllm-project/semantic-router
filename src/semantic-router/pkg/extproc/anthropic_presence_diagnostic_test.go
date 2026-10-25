package extproc

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestAnthropicOmittedStopSequenceReachesProtocolWarning(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	request := testNeutralRequest(model, "hello")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	if _, err := router.prepareProviderDispatch(request, model, "", false, ctx); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"id":"msg1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	response, err := router.decodeClientResponse(body, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || response.Output[0].Content[0].Text != "hello" {
		t.Fatalf("provider answer changed: %+v", response)
	}
	builder := newResponseHeaderMutationBuilder()
	builder.addProtocolDiagnostics(ctx, ctx.ProtocolDiagnostics)
	for _, header := range builder.setHeaders {
		if header.Header.Key != headers.VSRProtocolWarnings {
			continue
		}
		value := string(header.Header.RawValue)
		if !strings.Contains(value, "interpreted as null") || !strings.HasSuffix(value, ";stop_sequence") || len(value) > lossinessHeaderSizeLimit {
			t.Fatalf("compatibility warning not observable or bounded: %q", value)
		}
		return
	}
	t.Fatal("compatible provider omission had no caller-visible warning")
}

func TestAnthropicStreamOmissionRecordsWarningAfterHeaders(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	request := testNeutralRequest(model, "hello")
	request.Stream = true
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	if _, err := router.prepareProviderDispatch(request, model, "", true, ctx); err != nil {
		t.Fatal(err)
	}
	if err := router.ensureSemanticResponseStream(ctx); err != nil {
		t.Fatal(err)
	}
	counter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "anthropic", "approximated",
		"absent nullable stop_sequence interpreted as null for provider compatibility")
	before := testutil.ToFloat64(counter)
	var buffers semanticStreamBuffers
	buffers.push([]byte("event: message_start\ndata: "+`{"type":"message_start","message":{"id":"msg1","type":"message","role":"assistant","model":"provider-model","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`+"\n\n"), ctx)
	buffers.push([]byte("event: message_delta\ndata: "+`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}`+"\n\n"), ctx)
	if buffers.streamErr != nil {
		t.Fatal(buffers.streamErr)
	}
	if got := testutil.ToFloat64(counter); got != before+2 {
		t.Fatalf("late stream warnings not observed: before=%v after=%v", before, got)
	}
	if len(ctx.ProtocolDiagnostics) != 2 {
		t.Fatalf("stream diagnostics = %+v, want both missing-field warnings", ctx.ProtocolDiagnostics)
	}
}
