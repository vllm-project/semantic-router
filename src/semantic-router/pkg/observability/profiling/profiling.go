// Package profiling exposes the router's optional pprof HTTP listener.
//
// The handlers are registered on a dedicated mux rather than
// http.DefaultServeMux: importing net/http/pprof registers the debug endpoints
// on the default mux as a package side effect, so any server bound to the
// default mux would start serving profiles implicitly.
package profiling

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// readHeaderTimeout bounds how long a client may take to send request headers.
// Read and write deadlines are deliberately left unset: CPU profiles and
// execution traces stream for their full requested duration.
const readHeaderTimeout = 10 * time.Second

// Server owns the listener and HTTP server backing the pprof endpoints.
type Server struct {
	listener net.Listener
	server   *http.Server
}

// NewServeMux returns a mux serving only the pprof endpoints. Index dispatches
// the runtime-registered profiles (heap, goroutine, allocs, ...) itself, so the
// remaining handlers cover the profiles that need dedicated entry points.
func NewServeMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	return mux
}

// Start binds the pprof listener described by cfg and serves it on a background
// goroutine. It returns a nil server when profiling is disabled. The bind
// address and port are used verbatim so callers can pass port 0 for an
// ephemeral port and read the effective address back from Addr.
func Start(cfg config.ProfilingConfig) (*Server, error) {
	if !cfg.Enabled {
		return nil, nil
	}
	if err := ValidateBind(cfg.Bind); err != nil {
		return nil, err
	}

	addr := net.JoinHostPort(cfg.Bind, fmt.Sprintf("%d", cfg.Port))
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("failed to bind pprof listener on %s: %w", addr, err)
	}

	srv := &http.Server{
		Handler:           NewServeMux(),
		ReadHeaderTimeout: readHeaderTimeout,
	}
	s := &Server{listener: listener, server: srv}
	go func() {
		if err := srv.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logging.ComponentErrorEvent("router", "profiling_server_serve_failed", map[string]interface{}{
				"address": addr,
				"error":   err.Error(),
			})
		}
	}()
	return s, nil
}

// Addr reports the address the listener is bound to, with the resolved port.
func (s *Server) Addr() string {
	if s == nil || s.listener == nil {
		return ""
	}
	return s.listener.Addr().String()
}

// Close shuts the pprof listener down.
func (s *Server) Close() error {
	if s == nil || s.server == nil {
		return nil
	}
	return s.server.Close()
}

// ValidatePort rejects a profiling port that would collide with a port already
// claimed by another router service, and rejects ports outside the valid range.
// Port 0 is allowed so tests and operators can request an ephemeral port.
func ValidatePort(port int, reserved ...int) error {
	if port < 0 || port > 65535 {
		return fmt.Errorf("profiling port %d is out of range", port)
	}
	if port == 0 {
		return nil
	}
	for _, other := range reserved {
		if other > 0 && port == other {
			return fmt.Errorf("profiling port %d conflicts with another Router service port", port)
		}
	}
	return nil
}

// ValidateBind rejects a profiling bind address that is neither a literal IP
// nor localhost. An empty bind is rejected too: net.JoinHostPort would turn it
// into a wildcard listener and publish profiles on every interface.
func ValidateBind(bind string) error {
	if strings.TrimSpace(bind) == "" {
		return fmt.Errorf("profiling bind must not be empty")
	}
	if ip := net.ParseIP(bind); ip != nil {
		return nil
	}
	if bind == "localhost" {
		return nil
	}
	return fmt.Errorf("profiling bind %q must be an IP address or localhost", bind)
}
