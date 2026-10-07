package upstream

import (
	"io"
	"net/http"
	"time"
)

// Response is an upstream response that is ready to commit to the client.
type Response struct {
	StatusCode int
	// Header has hop-by-hop fields removed.
	Header http.Header
	// Body streams the rest of the response. The caller must close it; doing
	// so ends the call.
	Body io.ReadCloser
	// Trailer holds the response trailers once Body has reached EOF.
	Trailer http.Header
	// Cluster names the cluster that served. DefaultRoute reports that the
	// request carried no route key, or one no cluster serves.
	Cluster      string
	DefaultRoute bool
	// Endpoint is the backend that served; it is zero for a local reply.
	Endpoint EndpointSpec
	// Attempts lists every attempt of the call, the serving one last.
	Attempts []Attempt
	// Local is set when no attempt produced a response and the upstream
	// layer answered itself with Envoy's local reply for the last failure.
	Local *Error
}

// Attempt records one try of a call against one endpoint.
type Attempt struct {
	Cluster string
	// Endpoint is the backend's name; Address is where it was dialed.
	Endpoint string
	Address  string
	Start    time.Time
	// Latency runs until the response was ready to commit, or until the
	// attempt failed.
	Latency time.Duration
	// StatusCode is the response status, zero when none arrived.
	StatusCode int
	// Err classifies the failure; it is nil when a response arrived.
	Err *Error
}

// statusClass is the bounded outcome label of a response status.
func statusClass(code int) string {
	switch {
	case code >= 500:
		return "5xx"
	case code >= 400:
		return "4xx"
	case code >= 300:
		return "3xx"
	case code >= 200:
		return "2xx"
	default:
		return "1xx"
	}
}
