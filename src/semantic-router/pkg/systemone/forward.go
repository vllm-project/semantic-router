package systemone

import (
	"encoding/json"
	"net/http"
)

// ForwardPath is the authenticated management transport for a public native
// request. Listener authorization still runs in the active frontend generation.
const ForwardPath = "/api/v1/diagnostics/models/systemone/forward"

// BackendRequestHeader prevents a remote native model action from entering
// another routing or forwarding chain. It only restricts the receiving request;
// it never grants authorization or trusts a caller's identity.
const BackendRequestHeader = "X-VSR-SystemOne-Backend"

// ForwardRequest carries only the original native operation and client
// credentials. It never selects a deployment, runtime endpoint or saved config.
type ForwardRequest struct { //nolint:gosec // G117: credentials are forwarded for listener authentication, never logged or persisted.
	Listener       string          `json:"listener,omitempty"`
	Method         string          `json:"method"`
	Path           string          `json:"path"`
	Authorization  string          `json:"authorization,omitempty"`
	APIKey         string          `json:"api_key,omitempty"`
	BackendRequest bool            `json:"backend_request,omitempty"`
	Request        json.RawMessage `json:"request,omitempty"`
}

func (f ForwardRequest) ValidOperation() bool {
	return f.Method == http.MethodGet && f.Path == "/v1/systemone/models" ||
		f.Method == http.MethodPost && (f.Path == "/v1/systemone" || f.Path == "/v1/decisions")
}
