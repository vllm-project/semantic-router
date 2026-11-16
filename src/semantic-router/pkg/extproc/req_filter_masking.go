package extproc

import (
	"context"
	"errors"

	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/masking"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
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

// ruleSpans is the one signal method masking needs: what a pii rule, or a
// rule that finds text, finds in a text.
type ruleSpans interface {
	RuleSpans(ctx context.Context, signalType, rule, text string) ([]modelservice.Span, error)
}

// maskingSpansFor resolves the signal layer of the request's recipe. It is a
// variable so tests can supply spans without a model.
var maskingSpansFor = func(r *OpenAIRouter, ctx *RequestContext) (ruleSpans, bool) {
	classifier := r.classifierForRequest(ctx)
	return classifier, classifier != nil
}

// applyMaskingBeforeDispatch masks the neutral request in place when its
// recipe's plugins enable masking: every text position of the request as it
// is sent, after every other rewrite, including content added after routing
// such as retrieved documents and memory.
func (r *OpenAIRouter) applyMaskingBeforeDispatch(ctx *RequestContext) error {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil
	}
	cfg := requestMasking(ctx)
	if cfg == nil || !cfg.Enabled {
		return nil
	}
	finder, ok := maskingSpansFor(r, ctx)
	if !ok {
		// Fail closed: a masking-enabled route must never dispatch content it
		// could not scan (D4).
		return &maskingDispatchError{cause: errors.New("the recipe's signals are unavailable")}
	}
	if ctx.Masking == nil {
		// Once per request, never per dispatch: a later dispatch continues the
		// index sequence instead of reusing _0 for a different value (D5).
		ctx.Masking = masking.NewAllocator(cfg)
	}
	scan := spanScannerFor(ctx.embeddingContext(), finder, cfg)
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

// spanScannerFor adapts the pii rule and the rules masking names to
// masking.ScanFunc: their spans together, a pii span under its label and a
// rule's under the rule's name. The spans are code points into text;
// masking.Span offsets are bytes (D2).
func spanScannerFor(requestCtx context.Context, finder ruleSpans, cfg *config.MaskingPluginConfig) masking.ScanFunc {
	refs := cfg.Rules
	if cfg.Signal != "" {
		refs = append([]config.SignalRef{{Type: config.SignalTypePII, Name: cfg.Signal}}, refs...)
	}
	return func(text string) ([]masking.Span, error) {
		var detections []modelservice.Span
		for _, ref := range refs {
			found, err := finder.RuleSpans(requestCtx, ref.Type, ref.Name, text)
			if err != nil {
				// Includes a scan that read only part of the text, whose
				// spans cover only that part; the unscanned rest must not be
				// dispatched (D4).
				return nil, err
			}
			detections = append(detections, found...)
		}
		offsets := modelservice.ByteOffsets(text)
		spans := make([]masking.Span, len(detections))
		for i, detection := range detections {
			spans[i] = masking.Span{
				EntityType: detection.Label,
				Start:      offsets(detection.Start),
				End:        offsets(detection.End),
				Text:       detection.Text,
				Confidence: float32(detection.Probability),
			}
		}
		return spans, nil
	}
}
