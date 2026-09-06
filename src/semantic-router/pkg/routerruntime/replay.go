package routerruntime

// ReplayResponse is the transport-neutral result returned by the router-owned
// replay runtime to the authenticated management API.
type ReplayResponse struct {
	StatusCode int
	Body       []byte
}

// ReplayRuntime exposes the read-only Router Replay API without coupling the
// management server to Envoy ExtProc response types.
type ReplayRuntime interface {
	HandleReplayRequest(method, requestTarget string) (ReplayResponse, bool)
}
