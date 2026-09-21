package extproc

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

func routingTraceRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
	})
	return recorder
}

func traceAttributes(span sdktrace.ReadOnlySpan) map[string]attribute.Value {
	result := make(map[string]attribute.Value)
	for _, attr := range span.Attributes() {
		result[string(attr.Key)] = attr.Value
	}
	return result
}

func TestRoutingTraceSpansCoverStreamingAndPreserveSiblingParentage(t *testing.T) {
	recorder := routingTraceRecorder(t)
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: map[string]string{}, TraceContext: t.Context()}
	headers := buildMinimalHeaderRequest().GetRequestHeaders()
	_, err := router.handleRequestHeaders(&ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: headers}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(recorder.Ended()) != 0 || ctx.RequestSpan == nil {
		t.Fatal("request span ended at request headers")
	}
	rootID := ctx.RequestSpan.SpanContext().SpanID()
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "balanced"})
	observeRoutingIdentity(ctx, " \tentry\n")
	_, signal := tracing.StartSpan(ctx.TraceContext, tracing.SpanSignalEvaluation)
	ctx.VSRProjectionScores = map[string]float64{"difficulty": 0.25}
	ctx.VSRMatchedProjection = []string{"medium"}
	observeProjectionResults(ctx, signal)
	tracing.EndSignalSpan(signal, []string{"math"}, 0, 1, false)
	_, decisionSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanDecisionEvaluation)
	decision := &config.Decision{Name: "reason", Algorithm: &config.AlgorithmConfig{Type: "static"}}
	observeDecisionIdentity(ctx, decisionSpan, decision)
	tracing.EndDecisionSpan(decisionSpan, 0, []string{"math"}, "priority", false)
	router.startUpstreamSpanAndInjectHeaders("model-a", "provider.invalid", ctx)
	ctx.IsStreamingResponse = true
	ctx.UpstreamStatusCode = 200
	annotateUpstreamResponseSpan(ctx, responseHeaderOutcome{statusCode: 200, isSuccessful: true})
	if len(recorder.Ended()) != 2 {
		t.Fatal("root/upstream ended before streamed response body")
	}
	_, plugin := tracing.StartPluginSpan(ctx.TraceContext, "response_guard", "reason")
	plugin.End()
	ctx.StreamingComplete = true
	finishRequestTrace(ctx, nil)
	finishRequestTrace(ctx, nil)
	if len(recorder.Ended()) != 5 {
		t.Fatalf("ended spans = %d, want exactly 5", len(recorder.Ended()))
	}
	var root sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		attrs := traceAttributes(span)
		if span.SpanContext().SpanID() == rootID {
			root = span
			continue
		}
		if span.Parent().SpanID() != rootID {
			t.Fatalf("%s parent = %v, want live request root", span.Name(), span.Parent())
		}
		if attrs[tracing.AttrRecipe].AsString() != "balanced" {
			t.Fatalf("recipe absent on %s", span.Name())
		}
		if attrs[tracing.AttrEntrypoint].AsString() != "entry" {
			t.Fatalf("entrypoint was not normalized on %s", span.Name())
		}
		if span.Name() == tracing.SpanSignalEvaluation {
			if _, fabricated := attrs[tracing.AttrSignalConfidence]; fabricated {
				t.Fatal("aggregate signal phase invented confidence")
			}
			if len(span.Events()) != 1 || span.Events()[0].Name != "routing.projection.score" {
				t.Fatal("actual projection score missing")
			}
		}
		if span.Name() == tracing.SpanPluginExecution && attrs[tracing.AttrAlgorithm].AsString() != "static" {
			t.Fatal("plugin lost resolved algorithm identity")
		}
	}
	if root == nil || root.EndTime().Before(recorder.Ended()[0].EndTime()) {
		t.Fatal("request root does not encompass child execution")
	}
}

func TestRoutingTraceFinishesOnCancellationAndPanic(t *testing.T) {
	for _, err := range []error{context.Canceled, context.DeadlineExceeded, errors.New("private transport details")} {
		t.Run(err.Error(), func(t *testing.T) {
			recorder := routingTraceRecorder(t)
			ctx := &RequestContext{}
			ctx.TraceContext, ctx.RequestSpan = tracing.StartSpan(t.Context(), tracing.SpanRequest)
			_, ctx.UpstreamSpan = tracing.StartSpan(ctx.TraceContext, tracing.SpanUpstreamRequest)
			ctx.TraceReceiveError = err
			finishRequestTrace(ctx, nil)
			for _, span := range recorder.Ended() {
				if span.Status().Code != codes.Error || span.Status().Description == err.Error() {
					t.Fatalf("missing bounded error status: %+v", span.Status())
				}
			}
			if len(recorder.Ended()) != 2 {
				t.Fatal("cancellation leaked a span")
			}
		})
	}
	t.Run("process panic", func(t *testing.T) {
		recorder := routingTraceRecorder(t)
		router := &OpenAIRouter{Config: &config.RouterConfig{}}
		stream := &panicOnSendStream{MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{buildMinimalHeaderRequest()}), panicMsg: "fixture"}
		if err := router.Process(stream); err == nil {
			t.Fatal("panic not surfaced")
		}
		if len(recorder.Ended()) != 1 || recorder.Ended()[0].Status().Code != codes.Error {
			t.Fatal("panic leaked request root")
		}
	})
}

func TestRoutingSelectionTraceReportsRealDecisionName(t *testing.T) {
	recorder := routingTraceRecorder(t)
	ctx := &RequestContext{TraceContext: t.Context(), VSRSelectionMethod: "static"}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "recipe"})
	ctx.VSRSelectedDecision = &config.Decision{Name: "selected"}
	_, span := tracing.StartSpan(ctx.TraceContext, tracing.SpanAlgorithmSelection)
	observeAlgorithmSelection(ctx, span, "model", nil)
	observeRoutingStage(ctx, "algorithm", time.Now())
	span.End()
	if traceAttributes(recorder.Ended()[0])[tracing.AttrSelectedModel].AsString() != "model" {
		t.Fatal("selection identity missing")
	}
	if trace.SpanContextFromContext(ctx.TraceContext).IsValid() {
		t.Fatal("completed child replaced request context")
	}
}

type cancelAfterHeaderStream struct{ MockStream }

func (s *cancelAfterHeaderStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if s.RecvIndex > 0 {
		return nil, context.Canceled
	}
	return s.MockStream.Recv()
}

func TestRoutingTraceImmediateResponseKeepsTerminalStatusAfterTransportCancellation(t *testing.T) {
	before := requestOutcomeCount(t, "catalog", "success")
	recorder := routingTraceRecorder(t)
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	request := buildMinimalHeaderRequest()
	request.GetRequestHeaders().Headers.Headers = append(request.GetRequestHeaders().Headers.Headers,
		&core.HeaderValue{Key: ":method", Value: "GET"},
		&core.HeaderValue{Key: ":path", Value: "/v1/models"})
	stream := &cancelAfterHeaderStream{MockStream: *NewMockStream([]*ext_proc.ProcessingRequest{request})}
	if err := router.Process(stream); err != nil {
		t.Fatal(err)
	}
	spans := recorder.Ended()
	if len(spans) != 1 || spans[0].Status().Code == codes.Error {
		t.Fatalf("completed immediate response was changed by late gRPC cancellation: %+v", spans)
	}
	if traceAttributes(spans[0])["http.status_code"].AsInt64() != 200 {
		t.Fatal("immediate response status missing")
	}
	if requestOutcomeCount(t, "catalog", "success") != before+1 {
		t.Fatal("terminal response and late cancellation counted the same request twice")
	}
}

func requestOutcomeCount(t *testing.T, kind, outcome string) float64 {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != "llm_request_outcomes_total" {
			continue
		}
		for _, metric := range family.Metric {
			labels := map[string]string{}
			for _, label := range metric.Label {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["traffic_kind"] == kind && labels["outcome"] == outcome {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}

func TestRoutingTraceRoutesAreBoundedAndInternalTrafficIsDistinct(t *testing.T) {
	for _, tc := range []struct {
		path, route, kind string
		internal          bool
	}{
		{"/v1/chat/completions?api_key=private", "/v1/chat/completions", "inference", false},
		{"/v1/chat/completions", "/v1/chat/completions", "inference_internal", true},
		{"/v1/responses/private-id/input_items?token=secret", "/v1/responses/{response_id}/input_items", "response_object", false},
		{"/v1/models/private-model", "/v1/models/{model}", "catalog", false},
		{"/private-user/path?secret=value", "/{unmatched}", "other", false},
	} {
		t.Run(tc.kind+tc.path, func(t *testing.T) {
			recorder := routingTraceRecorder(t)
			ctx := &RequestContext{LooperRequest: tc.internal, Headers: map[string]string{"x-vsr-looper-request": "true"}}
			ctx.TraceContext, ctx.RequestSpan = tracing.StartSpan(t.Context(), tracing.SpanRequest)
			setRequestHeaderSpanAttributes(ctx.RequestSpan, ctx, "POST", tc.path)
			ctx.TraceStatusCode = 200
			finishRequestTrace(ctx, nil)
			attrs := traceAttributes(recorder.Ended()[0])
			if attrs[tracing.AttrHTTPPath].AsString() != tc.route || attrs["traffic.kind"].AsString() != tc.kind {
				t.Fatalf("unsafe or misclassified route: %+v", attrs)
			}
		})
	}
}

func TestRoutingTraceRAGErrorOmitsPrivateCause(t *testing.T) {
	recorder := routingTraceRecorder(t)
	_, span := tracing.StartSpan(t.Context(), tracing.SpanRAGRetrieval)
	secret := "private-provider?token=fixture-secret&prompt=private-input"
	ctx := &RequestContext{}
	if err := handleRAGRetrievalError(ctx, span, &config.RAGPluginConfig{Backend: "external_api", OnFailure: "skip"}, "route", errors.New(secret), .1); err != nil {
		t.Fatal(err)
	}
	span.End()
	got := recorder.Ended()[0]
	if got.Status().Description != "retrieval_failed" {
		t.Fatal("error reason was not bounded")
	}
	for _, event := range got.Events() {
		for _, attr := range event.Attributes {
			if strings.Contains(attr.Value.AsString(), "fixture-secret") || strings.Contains(attr.Value.AsString(), "private-input") {
				t.Fatal("private cause entered trace")
			}
		}
	}
}

func TestRoutingTracePrematureEOFDoesNotClaimCompletedResponse(t *testing.T) {
	recorder := routingTraceRecorder(t)
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	stream := NewMockStream([]*ext_proc.ProcessingRequest{buildMinimalHeaderRequest()})
	if err := router.Process(stream); err != nil {
		t.Fatalf("tracing changed the normal EOF return contract: %v", err)
	}
	spans := recorder.Ended()
	if len(spans) != 1 || spans[0].Status().Code != codes.Error ||
		spans[0].Status().Description != "stream_ended_before_terminal_response" {
		t.Fatalf("premature EOF was not identified: %+v", spans)
	}
}

func TestRoutingTraceLocalResponseErrorKeepsProviderStatus(t *testing.T) {
	for _, tc := range []struct {
		name string
		code int
		eos  bool
	}{{"guard block before EOS", 403, false}, {"guard block at EOS", 403, true}, {"decode error", 502, true}} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := routingTraceRecorder(t)
			server := newJailbreakScoreServer(t, 0.95, 0.05)
			router, ctx := newResponseStageRouter(t, server, "", "block")
			ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
			ctx.StartTime = time.Now()
			ctx.VSRSelectedModel = "provider-model"
			ctx.TraceContext, ctx.RequestSpan = tracing.StartSpan(t.Context(), tracing.SpanRequest)
			_, ctx.UpstreamSpan = tracing.StartSpan(ctx.TraceContext, tracing.SpanUpstreamRequest)
			stream := NewMockStream(nil)
			headers := &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}}},
			}}
			if err := router.processResponseHeaders(stream, headers, ctx); err != nil {
				t.Fatal(err)
			}
			body := []byte(`{"id":"fixture","object":"chat.completion","model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"fixture response"},"finish_reason":"stop"}]}`)
			if tc.code == 502 {
				body = []byte("invalid provider JSON")
			}
			if err := router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: tc.eos},
			}, ctx); err != nil {
				t.Fatal(err)
			}
			if got := stream.Responses[len(stream.Responses)-1].GetImmediateResponse().GetStatus().GetCode(); int(got) != tc.code {
				t.Fatalf("actual local response = %d, want %d", got, tc.code)
			}
			if ctx.RequestSpan != nil || ctx.UpstreamSpan != nil {
				t.Fatal("local terminal response left a trace open")
			}
			finishRequestTrace(ctx, context.Canceled)
			for _, span := range recorder.Ended() {
				attrs := traceAttributes(span)
				switch span.Name() {
				case tracing.SpanRequest:
					if attrs["http.status_code"].AsInt64() != int64(tc.code) || span.Status().Code != codes.Error {
						t.Fatalf("wrong client status: %+v, %+v", attrs, span.Status())
					}
					if _, ok := attrs[tracing.AttrSelectedModel]; ok {
						t.Fatal("blocked client response claimed a successful final model")
					}
				case tracing.SpanUpstreamRequest:
					if attrs["http.status_code"].AsInt64() != 200 || span.Status().Code == codes.Error {
						t.Fatalf("local guard rewrote provider status: %+v, %+v", attrs, span.Status())
					}
				}
			}
			if len(recorder.Ended()) != 2 {
				t.Fatalf("unexpected span count: %d", len(recorder.Ended()))
			}
		})
	}
}

func TestRoutingTraceSkipProcessingPreservesHTTPStatus(t *testing.T) {
	recorder := routingTraceRecorder(t)
	ctx := &RequestContext{SkipProcessing: true}
	ctx.TraceContext, ctx.RequestSpan = tracing.StartSpan(t.Context(), tracing.SpanRequest)
	router := &OpenAIRouter{}
	if err := router.processResponseHeaders(NewMockStream(nil), &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{EndOfStream: true, Headers: &core.HeaderMap{
			Headers: []*core.HeaderValue{{Key: ":status", Value: "503"}},
		}},
	}, ctx); err != nil {
		t.Fatal(err)
	}
	spans := recorder.Ended()
	if len(spans) != 1 || traceAttributes(spans[0])["http.status_code"].AsInt64() != 503 || spans[0].Status().Code != codes.Error {
		t.Fatal("skip-processing lost the actual HTTP failure")
	}
}

func TestRoutingTraceSignalEvidencePreservesFinitePresence(t *testing.T) {
	recorder := routingTraceRecorder(t)
	_, span := tracing.StartSpan(t.Context(), tracing.SpanSignalEvaluation)
	ctx := &RequestContext{
		VSRSignalValues:      map[string]float64{"both": 0.5, "value": 0, "invalid": math.NaN(), "confidence": math.Inf(1)},
		VSRSignalConfidences: map[string]float64{"both": 0.8, "confidence": 0, "invalid": math.Inf(-1), "value": math.NaN()},
	}
	observeSignalEvidence(ctx, span)
	span.End()
	events := recorder.Ended()[0].Events()
	if len(events) != 3 {
		t.Fatalf("signal events = %d, want only three finite evidence keys", len(events))
	}
	for i, name := range []string{"both", "confidence", "value"} {
		attrs := make(map[string]attribute.Value)
		for _, attr := range events[i].Attributes {
			attrs[string(attr.Key)] = attr.Value
		}
		if events[i].Name != "routing.signal.evidence" || attrs["signal.key"].AsString() != name {
			t.Fatal("signal evidence keys are not sorted")
		}
		_, valueExists := attrs["signal.value"]
		_, confidenceExists := attrs["signal.confidence"]
		if valueExists != (name != "confidence") || confidenceExists != (name != "value") || attrs["signal.value_present"].AsBool() != valueExists || attrs["signal.confidence_present"].AsBool() != confidenceExists {
			t.Fatalf("missing evidence became a fabricated zero: %v", attrs)
		}
	}
	if events[1].Attributes[len(events[1].Attributes)-1].Value.AsFloat64() != 0 || events[2].Attributes[len(events[2].Attributes)-1].Value.AsFloat64() != 0 {
		t.Fatal("real zero evidence was omitted")
	}
}
