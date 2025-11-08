package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// logRoutingDecision logs routing decision with structured logging
func (r *OpenAIRouter) logRoutingDecision(ctx *RequestContext, reasonCode string, originalModel string, selectedModel string, categoryName string, reasoningEnabled bool, endpoint string) {
	effortForMetrics := ""
	if reasoningEnabled && categoryName != "" {
		effortForMetrics = r.getReasoningEffort(categoryName, selectedModel)
	}

	logging.LogEvent("routing_decision", map[string]interface{}{
		"reason_code":        reasonCode,
		"request_id":         ctx.RequestID,
		"original_model":     originalModel,
		"selected_model":     selectedModel,
		"category":           categoryName,
		"reasoning_enabled":  reasoningEnabled,
		"reasoning_effort":   effortForMetrics,
		"selected_endpoint":  endpoint,
		"routing_latency_ms": time.Since(ctx.ProcessingStartTime).Milliseconds(),
	})
	metrics.RecordRoutingReasonCode(reasonCode, selectedModel)
}

// recordRoutingDecision records routing decision with tracing
func (r *OpenAIRouter) recordRoutingDecision(ctx *RequestContext, categoryName string, originalModel string, matchedModel string, reasoningDecision entropy.ReasoningDecision) {
	routingCtx, routingSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanRoutingDecision)

	useReasoning := reasoningDecision.UseReasoning
	logging.Infof("Entropy-based reasoning decision for this query: %v on [%s] model (confidence: %.3f, reason: %s)",
		useReasoning, matchedModel, reasoningDecision.Confidence, reasoningDecision.DecisionReason)

	effortForMetrics := r.getReasoningEffort(categoryName, matchedModel)
	metrics.RecordReasoningDecision(categoryName, matchedModel, useReasoning, effortForMetrics)

	tracing.SetSpanAttributes(routingSpan,
		attribute.String(tracing.AttrRoutingStrategy, "auto"),
		attribute.String(tracing.AttrRoutingReason, reasoningDecision.DecisionReason),
		attribute.String(tracing.AttrOriginalModel, originalModel),
		attribute.String(tracing.AttrSelectedModel, matchedModel),
		attribute.Bool(tracing.AttrReasoningEnabled, useReasoning),
		attribute.String(tracing.AttrReasoningEffort, effortForMetrics))

	routingSpan.End()
	ctx.TraceContext = routingCtx
}

// trackVSRDecision tracks VSR decision information in context
func (r *OpenAIRouter) trackVSRDecision(ctx *RequestContext, categoryName string, matchedModel string, useReasoning bool) {
	ctx.VSRSelectedCategory = categoryName
	ctx.VSRSelectedModel = matchedModel
	if useReasoning {
		ctx.VSRReasoningMode = "on"
	} else {
		ctx.VSRReasoningMode = "off"
	}
}

// setClearRouteCache sets the ClearRouteCache flag on the response
func (r *OpenAIRouter) setClearRouteCache(response *ext_proc.ProcessingResponse) {
	if response.GetRequestBody() != nil && response.GetRequestBody().GetResponse() != nil {
		response.GetRequestBody().GetResponse().ClearRouteCache = true
		logging.Debugf("Setting ClearRouteCache=true (feature enabled)")
	}
}

// recordRoutingLatency records the routing latency metric
func (r *OpenAIRouter) recordRoutingLatency(ctx *RequestContext) {
	routingLatency := time.Since(ctx.ProcessingStartTime)
	metrics.RecordModelRoutingLatency(routingLatency.Seconds())
}
