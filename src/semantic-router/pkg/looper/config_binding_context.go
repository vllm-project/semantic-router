package looper

import "context"

type expectedConfigHashContextKey struct{}

// WithExpectedConfigHash carries an already validated outer request binding to
// all child calls. A reload may reject a new child, but cannot silently mix
// runtime generations within an evaluation trajectory.
func WithExpectedConfigHash(ctx context.Context, hash string) context.Context {
	if hash == "" {
		return ctx
	}
	return context.WithValue(ctx, expectedConfigHashContextKey{}, hash)
}

func expectedConfigHashFromContext(ctx context.Context) string {
	hash, _ := ctx.Value(expectedConfigHashContextKey{}).(string)
	return hash
}
