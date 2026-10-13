package gateway

import (
	"net/http"
	"time"
)

// Probe paths the gateway answers itself, before API keys and routing.
const (
	HealthPath = "/health"
	ReadyPath  = "/ready"
)

// serveProbe answers liveness and readiness probes. /health reports that the
// process serves; /ready that the routing core can take traffic.
func (h *Handler) serveProbe(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		return false
	}
	switch r.URL.Path {
	case HealthPath:
		writeProbe(w, http.StatusOK, `{"status":"ok"}`)
	case ReadyPath:
		if h.opts.Ready != nil && !h.opts.Ready() {
			writeProbe(w, http.StatusServiceUnavailable, `{"status":"not_ready"}`)
		} else {
			writeProbe(w, http.StatusOK, `{"status":"ready"}`)
		}
	default:
		return false
	}
	return true
}

func writeProbe(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(body))
}

// AccessRecord carries the fields of the local Envoy template's access log.
type AccessRecord struct {
	Start               time.Time
	Method              string
	Path                string
	Protocol            string
	Status              int
	BytesReceived       int64
	BytesSent           int64
	Duration            time.Duration
	UpstreamServiceTime time.Duration
	ForwardedFor        string
	UserAgent           string
	RequestID           string
	Authority           string
	UpstreamHost        string
}

func newAccessRecord(r *http.Request) AccessRecord {
	return AccessRecord{
		Start:         time.Now(),
		Method:        r.Method,
		Path:          requestTarget(r),
		Protocol:      r.Proto,
		BytesReceived: max(r.ContentLength, 0),
		ForwardedFor:  r.Header.Get("X-Forwarded-For"),
		UserAgent:     r.Header.Get("User-Agent"),
		Authority:     r.Host,
	}
}

// exchange is the state one request accumulates for its access record.
type exchange struct {
	record AccessRecord
}

func (h *Handler) logAccess(x *exchange, w *countingWriter) {
	if h.opts.AccessLog == nil {
		return
	}
	x.record.Status = w.status
	x.record.BytesSent = w.written
	x.record.Duration = time.Since(x.record.Start)
	h.opts.AccessLog(x.record)
}

// countingWriter records the status and body bytes written to the client.
type countingWriter struct {
	http.ResponseWriter
	status  int
	written int64
}

func (w *countingWriter) WriteHeader(status int) {
	if w.status == 0 {
		w.status = status
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *countingWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	n, err := w.ResponseWriter.Write(b)
	w.written += int64(n)
	return n, err
}

// Unwrap lets http.ResponseController reach the connection for flushes.
func (w *countingWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
