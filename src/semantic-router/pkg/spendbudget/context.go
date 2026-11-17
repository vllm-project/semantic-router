package spendbudget

import "context"

type limitsKey struct{}

// WithLimits marks ctx with the budgets of the request it serves, so a call
// made on the request's behalf, such as a Looper hop served in process,
// spends the same budgets.
func WithLimits(ctx context.Context, limits []Limit) context.Context {
	return context.WithValue(ctx, limitsKey{}, limits)
}

// LimitsFrom returns the budgets ctx is marked with.
func LimitsFrom(ctx context.Context) ([]Limit, bool) {
	if ctx == nil {
		return nil, false
	}
	limits, ok := ctx.Value(limitsKey{}).([]Limit)
	return limits, ok
}
