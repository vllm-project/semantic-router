package extproc

import (
	"context"
	"errors"

	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/masking"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// maskingUnavailableMessage is all a client sees; the cause stays in server
// logs so no request content can reach the response.
const maskingUnavailableMessage = "PII masking is unavailable; the request was not sent"

// maskingDispatchError is a masking failure at the dispatch boundary.
// processBodyRoutingError answers it with 503: as a stream error, Envoy's
// failure_mode_allow would forward the original, unmasked body (D4).
type maskingDispatchError struct{ cause error }

func (e *maskingDispatchError) Error() string { return "PII masking failed: " + e.cause.Error() }

func (e *maskingDispatchError) Unwrap() error { return e.cause }

// piiDetector is the one classifier method masking needs.
type piiDetector interface {
	ClassifyPIIWithDetailsAndThreshold(ctx context.Context, text string, threshold float32) ([]classification.PIIDetection, error)
}

// maskingDetectorFor resolves the request's PII detector and the model's own
// threshold. It is a variable so tests can supply spans without a model.
var maskingDetectorFor = func(r *OpenAIRouter, ctx *RequestContext) (piiDetector, float32, bool) {
	classifier := r.classifierForRequest(ctx)
	if classifier == nil || !classifier.IsPIIEnabled() {
		return nil, 0, false
	}
	return classifier, classifier.Config.PIIModel.Threshold, true
}

// applyMaskingBeforeDispatch masks the neutral request in place when the
// selected decision enables the masking plugin.
func (r *OpenAIRouter) applyMaskingBeforeDispatch(ctx *RequestContext) error {
	if ctx == nil || ctx.SemanticRequest == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	cfg := ctx.VSRSelectedDecision.GetMaskingConfig()
	if cfg == nil || !cfg.Enabled {
		return nil
	}
	detector, modelThreshold, ok := maskingDetectorFor(r, ctx)
	if !ok {
		// Fail closed: a masking-enabled route must never dispatch content it
		// could not scan (D4).
		return &maskingDispatchError{cause: errors.New("PII classifier is unavailable")}
	}
	if ctx.Masking == nil {
		// Once per request, never per dispatch: a later dispatch continues the
		// index sequence instead of reusing _0 for a different value (D5).
		ctx.Masking = masking.NewAllocator(cfg)
	}
	scan := spanScannerFor(ctx.embeddingContext(), detector, maskingScanThreshold(cfg, modelThreshold))
	result, err := masking.Apply(ctx.SemanticRequest, ctx.Masking, scan)
	if err != nil {
		return &maskingDispatchError{cause: err}
	}
	recordMaskingObservability(ctx, result)
	if result.Changed {
		// Without this the envelope still claims the original client bytes
		// describe the request, and EncodeRequest forwards them verbatim --
		// dispatching the raw PII while reporting success (#3566).
		ctx.SemanticRequest.Generation++
	}
	return nil
}

// recordMaskingObservability records counts and entity classes only -- never
// a value or a placeholder (#3566).
func recordMaskingObservability(ctx *RequestContext, result masking.Result) {
	if result.MaskedCount == 0 {
		return
	}
	decisionName := ctx.VSRSelectedDecisionName
	for entityType, count := range result.EntityCounts {
		metrics.RecordMaskedEntities(decisionName, entityType, count)
	}
	spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, config.DecisionPluginMasking, decisionName)
	tracing.SetSpanAttributes(span,
		attribute.Int("masking.count", result.MaskedCount),
		attribute.StringSlice("masking.entity_types", result.EntityTypes),
		attribute.Int("masking.citations_dropped", result.CitationsDropped),
	)
	tracing.EndPluginSpan(span, "success", 0, "pii_masked")
	ctx.TraceContext = spanCtx
}

// maskingScanThreshold uses the plugin threshold, falling back to the PII
// model's own when the plugin leaves it at zero.
func maskingScanThreshold(cfg *config.MaskingPluginConfig, modelThreshold float32) float32 {
	if cfg.Threshold > 0 {
		return cfg.Threshold
	}
	return modelThreshold
}

// spanScannerFor adapts the classifier to masking.ScanFunc. Detection offsets
// are bytes, which is what masking.Span expects (D2).
func spanScannerFor(requestCtx context.Context, detector piiDetector, threshold float32) masking.ScanFunc {
	return func(text string) ([]masking.Span, error) {
		detections, err := detector.ClassifyPIIWithDetailsAndThreshold(requestCtx, text, threshold)
		if err != nil {
			// Includes ErrTokenSpansTruncated, whose spans cover only the prefix
			// the model saw; the unscanned rest must not be dispatched (D4).
			return nil, err
		}
		spans := make([]masking.Span, len(detections))
		for i, detection := range detections {
			spans[i] = masking.Span{
				EntityType: detection.EntityType,
				Start:      detection.Start,
				End:        detection.End,
				Text:       detection.Text,
				Confidence: detection.Confidence,
			}
		}
		return spans, nil
	}
}
