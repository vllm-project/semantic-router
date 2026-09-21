package tracing

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
)

const (
	AttrEntrypoint         = "routing.entrypoint"
	AttrRecipe             = "routing.recipe"
	AttrAlgorithm          = "routing.algorithm"
	SpanAlgorithmSelection = "semantic_router.algorithm.selection"
)

type routingAttributesKey struct{}

// WithRoutingAttributes carries content-free, resolved routing identity into
// subsequent child spans, including plugins and looper calls. These values are
// local context only: they are never placed into outbound baggage.
func WithRoutingAttributes(ctx context.Context, attrs ...attribute.KeyValue) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	previous, _ := ctx.Value(routingAttributesKey{}).([]attribute.KeyValue)
	combined := append(append([]attribute.KeyValue(nil), previous...), attrs...)
	return context.WithValue(ctx, routingAttributesKey{}, combined)
}
