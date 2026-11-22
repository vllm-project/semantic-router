package routerruntime

import "context"

// ReplayResponse is the transport-neutral result returned by the router-owned
// replay runtime to the authenticated management API.
type ReplayResponse struct {
	StatusCode int
	Body       []byte
}

// ReplayRuntime exposes the read-only Router Replay API without coupling the
// management server to Envoy ExtProc response types. ctx is the management
// request's: a client that leaves stops the storage reads it began.
type ReplayRuntime interface {
	HandleReplayRequest(ctx context.Context, method, requestTarget string) (ReplayResponse, bool)
}
