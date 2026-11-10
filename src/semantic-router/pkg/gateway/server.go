package gateway

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"golang.org/x/net/netutil"
)

// ServerOptions configure a listener the way the local Envoy template
// configures its listener.
type ServerOptions struct {
	// IdleTimeout bounds a connection with no active request, and the wait for
	// a request's headers. The template's stream idle timeout is the
	// listener's timeout (300s by default in the CLI).
	IdleTimeout time.Duration
	// MaxHeaderBytes bounds request headers; Envoy allows 60 KiB.
	MaxHeaderBytes int
	// MaxConnections bounds concurrent connections; the template allows 50,000.
	MaxConnections int
	// TLS, when set, serves the listener over TLS, with HTTP/2 or HTTP/1.1
	// negotiated by ALPN. Without it the listener speaks cleartext HTTP/1.1
	// and h2c.
	TLS *tls.Config
}

// Defaults of the local Envoy template.
const (
	DefaultIdleTimeout    = 300 * time.Second
	DefaultMaxHeaderBytes = 60 << 10
	DefaultMaxConnections = 50000
)

// NewServer returns an HTTP server for handler that speaks HTTP/1.1 and,
// like Envoy's automatic codec, HTTP/2: over cleartext (h2c), or by ALPN
// when the listener serves TLS.
func NewServer(handler http.Handler, opts ServerOptions) *http.Server {
	if opts.IdleTimeout <= 0 {
		opts.IdleTimeout = DefaultIdleTimeout
	}
	if opts.MaxHeaderBytes <= 0 {
		opts.MaxHeaderBytes = DefaultMaxHeaderBytes
	}
	server := &http.Server{
		Handler:           handler,
		ReadHeaderTimeout: opts.IdleTimeout,
		IdleTimeout:       opts.IdleTimeout,
		MaxHeaderBytes:    opts.MaxHeaderBytes,
		TLSConfig:         opts.TLS,
	}
	if opts.TLS == nil {
		server.Handler = h2c.NewHandler(handler, &http2.Server{IdleTimeout: opts.IdleTimeout})
	}
	return server
}

// Serve accepts connections on ln until the server shuts down, over TLS
// when the server was made with TLS settings.
func Serve(server *http.Server, ln net.Listener) error {
	if server.TLSConfig != nil {
		return server.ServeTLS(ln, "", "")
	}
	return server.Serve(ln)
}

// Listen opens a TCP listener that accepts at most opts.MaxConnections
// connections at once.
func Listen(address string, opts ServerOptions) (net.Listener, error) {
	if opts.MaxConnections <= 0 {
		opts.MaxConnections = DefaultMaxConnections
	}
	ln, err := net.Listen("tcp", address)
	if err != nil {
		return nil, err
	}
	return netutil.LimitListener(ln, opts.MaxConnections), nil
}
