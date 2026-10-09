package routing

import (
	"context"
	"slices"
)

// ListenerModels are the request models a listener accepts: the exact
// `model` values its clients may send. A session opened with them answers
// any other model with 403 model_not_allowed before routing runs, and lists
// only them on /v1/models. Request-graph hops, which the Router makes
// itself, are not client requests and are not restricted.
type ListenerModels []string

// Allows reports whether model is one of m.
func (m ListenerModels) Allows(model string) bool {
	return slices.Contains(m, model)
}

type listenerModelsKey struct{}

// WithListenerModels returns ctx restricted to models; an empty list
// restricts nothing.
func WithListenerModels(ctx context.Context, models []string) context.Context {
	if len(models) == 0 {
		return ctx
	}
	return context.WithValue(ctx, listenerModelsKey{}, ListenerModels(slices.Clone(models)))
}

// ListenerModelsFrom returns the models ctx is restricted to.
func ListenerModelsFrom(ctx context.Context) (ListenerModels, bool) {
	if ctx == nil {
		return nil, false
	}
	models, ok := ctx.Value(listenerModelsKey{}).(ListenerModels)
	return models, ok
}
