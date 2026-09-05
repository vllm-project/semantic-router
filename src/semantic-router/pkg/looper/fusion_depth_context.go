package looper

import "context"

type fusionDepthContextKey struct{}

// contextWithFusionDepth attaches the Fusion recursion depth marker to a
// request context. The marker is per-request (not a mutable client field) so
// a slow panel worker that sends its HTTP request after Execute returns still
// carries the correct depth instead of racing the reset to zero.
func contextWithFusionDepth(ctx context.Context, depth int) context.Context {
	return context.WithValue(ctx, fusionDepthContextKey{}, depth)
}

func fusionDepthFromContext(ctx context.Context) int {
	if ctx == nil {
		return 0
	}
	depth, _ := ctx.Value(fusionDepthContextKey{}).(int)
	return depth
}
