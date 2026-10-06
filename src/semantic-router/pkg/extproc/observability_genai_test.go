package extproc

import (
	"context"
	"fmt"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

func TestUpstreamSpanCarriesGenAIAttributes(t *testing.T) {
	for _, tc := range []struct {
		name         string
		backend      llmprotocol.WireFormat
		providerType string
		providerName string
	}{
		{name: "openai_chat", backend: llmprotocol.OpenAIChatV1, providerType: "openai", providerName: "openai"},
		{name: "openai_responses", backend: llmprotocol.OpenAIResponsesV1, providerType: "openai", providerName: "openai"},
		{name: "anthropic_messages", backend: llmprotocol.AnthropicMessagesV1, providerType: "anthropic", providerName: "anthropic"},
		{name: "azure_openai", backend: llmprotocol.OpenAIChatV1, providerType: "azure-openai", providerName: "azure.ai.openai"},
		{name: "bedrock", backend: llmprotocol.OpenAIChatV1, providerType: "bedrock", providerName: "aws.bedrock"},
		{name: "self_hosted_vllm", backend: llmprotocol.OpenAIChatV1, providerType: "vllm", providerName: "vllm"},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/streaming=%t", tc.name, streaming), func(t *testing.T) {
				attrs, logicalModel := exportedUpstreamSpanAttributes(t, tc.backend, tc.providerType, streaming)

				require.Equal(t, "chat", attrs["gen_ai.operation.name"].AsString())
				require.Equal(t, tc.providerName, attrs["gen_ai.provider.name"].AsString())
				require.Equal(t, "provider-model", attrs["gen_ai.request.model"].AsString())
				require.Equal(t, int64(2), attrs["gen_ai.usage.input_tokens"].AsInt64())
				require.Equal(t, int64(1), attrs["gen_ai.usage.output_tokens"].AsInt64())
				responseModel, hasResponseModel := attrs["gen_ai.response.model"]
				if streaming {
					require.False(t, hasResponseModel, "stream events carry the client-facing model, not the provider's")
				} else {
					require.Equal(t, "source-model", responseModel.AsString())
				}

				require.Equal(t, logicalModel, attrs[tracing.AttrModelName].AsString())
				require.Equal(t, "127.0.0.1:8000", attrs[tracing.AttrEndpointAddress].AsString())
			})
		}
	}
}

// exportedUpstreamSpanAttributes dispatches one request and feeds the
// provider's response through the same ExtProc handlers Process uses, then
// returns the attributes of the upstream span the SDK exported.
func exportedUpstreamSpanAttributes(
	t *testing.T,
	backend llmprotocol.WireFormat,
	providerType string,
	streaming bool,
) (map[string]attribute.Value, string) {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
	})

	router, logicalModel := routingTestRouterForFormat(backend)
	profile := router.Config.ProviderProfiles["provider"]
	profile.Type = providerType
	router.Config.ProviderProfiles["provider"] = profile
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1,
		RequestID: "genai-trace-request", StartTime: time.Now(),
	}
	ctx.TraceContext, ctx.RequestSpan = tracing.StartSpan(t.Context(), tracing.SpanRequest)
	clientBody := fmt.Sprintf(`{"model":"virtual","messages":[{"role":"user","content":"hello"}],"stream":%t}`, streaming)
	request, immediate := router.prepareProtocolRequest([]byte(clientBody), ctx)
	require.Nil(t, immediate)
	_, err := router.handleSpecifiedModelRouting(request, logicalModel, "", ctx)
	require.NoError(t, err)

	contentType, providerBody := "application/json", extProcResponseFixture(backend)
	if streaming {
		contentType, providerBody = "text/event-stream", extProcStreamFixture(backend)
	}
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseHeaders(stream, looperTransportHeaders(200, contentType), ctx))
	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: providerBody, EndOfStream: true},
	}, ctx))

	for _, span := range exporter.GetSpans() {
		if span.Name != tracing.SpanUpstreamRequest {
			continue
		}
		attrs := make(map[string]attribute.Value, len(span.Attributes))
		for _, attr := range span.Attributes {
			attrs[string(attr.Key)] = attr.Value
		}
		return attrs, logicalModel
	}
	t.Fatalf("upstream span was not exported; got %d spans", len(exporter.GetSpans()))
	return nil, ""
}
