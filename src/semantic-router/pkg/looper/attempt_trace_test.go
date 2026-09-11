/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExecuteWithLatencyRecordsConfidenceAttemptTrace(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
	})

	server, _ := newConfidenceLogprobBackend(t, map[string]int{"small": 0, "large": 1})
	defer server.Close()

	cfg := config.LooperConfig{Endpoint: server.URL}
	request := confidenceLogprobRequest("skip")
	request.BaseContextTokens = 50
	request.OriginalRequest.MaxTokens = openai.Int(10)
	request.ModelParams["small"] = config.ModelParams{ParamSize: "1b", Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 2}}
	request.ModelParams["large"] = config.ModelParams{ParamSize: "10b", Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 2}}
	response, err := ExecuteWithLatency(
		context.Background(),
		NewConfidenceLooper(&cfg),
		request,
	)
	if err != nil {
		t.Fatalf("ExecuteWithLatency: %v", err)
	}
	trace := response.ExecutionTrace
	if trace.Version != ExecutionTraceVersion || trace.TraceID == "" {
		t.Fatalf("execution trace identity = version %d trace %q", trace.Version, trace.TraceID)
	}
	if len(trace.Attempts) != 2 || trace.FinalAttemptOrdinal != 2 {
		t.Fatalf("attempt trace = %+v, want two attempts and final ordinal 2", trace)
	}
	if trace.Attempts[0].Status != AttemptStatusSucceeded ||
		trace.Attempts[0].Reason != AttemptReasonUnusable ||
		!trace.Attempts[0].Discarded {
		t.Fatalf("first attempt = %+v, want discarded unusable response", trace.Attempts[0])
	}
	if !trace.Attempts[1].Selected || trace.Attempts[1].Accepted == nil || !*trace.Attempts[1].Accepted {
		t.Fatalf("second attempt = %+v, want accepted selection", trace.Attempts[1])
	}
	if trace.Attempts[1].ReservedTokens == nil || *trace.Attempts[1].ReservedTokens != 10 ||
		trace.Attempts[1].EstimatedTokens == nil || trace.Attempts[1].ActualCost == nil || trace.Attempts[1].Currency != "USD" {
		t.Fatalf("second attempt accounting = %+v", trace.Attempts[1])
	}
	if trace.Attempts[0].Usage.TotalTokens+trace.Attempts[1].Usage.TotalTokens != response.Usage.TotalTokens {
		t.Fatalf("attempt usage does not reconcile with response: %+v vs %+v", trace.Attempts, response.Usage)
	}

	encoded, err := json.Marshal(trace)
	if err != nil {
		t.Fatalf("marshal trace: %v", err)
	}
	if strings.Contains(string(encoded), "small answer") || strings.Contains(string(encoded), server.URL) {
		t.Fatalf("execution trace leaked content or endpoint: %s", encoded)
	}

	spans := exporter.GetSpans()
	if len(spans) != 3 {
		t.Fatalf("ended spans = %d, want parent plus two attempts", len(spans))
	}
	var parentSpanID oteltrace.SpanID
	children := 0
	for _, span := range spans {
		if span.Name == "looper.execute" {
			parentSpanID = span.SpanContext.SpanID()
		}
	}
	if !parentSpanID.IsValid() {
		t.Fatal("looper.execute span missing")
	}
	for _, span := range spans {
		if span.Name == "looper.attempt" {
			children++
			if span.Parent.SpanID() != parentSpanID {
				t.Fatalf("attempt parent = %s, want %s", span.Parent.SpanID(), parentSpanID)
			}
		}
	}
	if children != 2 {
		t.Fatalf("attempt spans = %d, want 2", children)
	}
}

func TestBoundExecutionTracePreservesDroppedUsage(t *testing.T) {
	trace := ExecutionTrace{Version: ExecutionTraceVersion, Algorithm: "confidence"}
	for ordinal := 1; ordinal <= maxTraceAttempts+1; ordinal++ {
		trace.Attempts = append(trace.Attempts, AttemptTrace{
			Ordinal: ordinal, Stage: "candidate",
			Status: AttemptStatusSucceeded, Usage: TokenUsage{TotalTokens: 1},
		})
	}

	bounded := boundExecutionTrace(trace)
	if !bounded.AttemptsTruncated || bounded.DroppedAttemptCount != 1 || bounded.DroppedUsage.TotalTokens != 1 {
		t.Fatalf("bounded trace = %+v, want one dropped attempt and token", bounded)
	}
}
