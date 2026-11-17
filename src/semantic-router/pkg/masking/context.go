package masking

import "context"

type scopeKey struct{}

// Scope pins verified identity and the root session across internal Looper hops.
// It never enters the request body or headers.
type Scope struct{ Principal, RootSession string }

func WithRequestScope(ctx context.Context, scope Scope) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, scopeKey{}, scope)
}

func RequestScope(ctx context.Context) (Scope, bool) {
	if ctx == nil {
		return Scope{}, false
	}
	s, ok := ctx.Value(scopeKey{}).(Scope)
	return s, ok
}
