package backenddispatch

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultReadHeaderTimeout = 5 * time.Second
	defaultIdleTimeout       = 2 * time.Minute
	defaultShutdownTimeout   = 15 * time.Second
)

type ServerOptions struct {
	BindAddress       string
	Port              int
	Handler           http.Handler
	Readiness         func(context.Context) error
	ReadHeaderTimeout time.Duration
	IdleTimeout       time.Duration
	ShutdownTimeout   time.Duration
}

// Server owns the private backend-dispatch listener. It deliberately exposes
// neither a public route registrar nor a provider-specific endpoint.
type Server struct {
	mu        sync.RWMutex
	options   ServerOptions
	server    *http.Server
	listener  net.Listener
	address   string
	started   bool
	closed    bool
	serveErr  error
	serveDone chan struct{}
	closeOnce sync.Once
	closeErr  error
}

func NewServer(options ServerOptions) (*Server, error) {
	if options.BindAddress != strings.TrimSpace(options.BindAddress) || net.ParseIP(options.BindAddress) == nil {
		return nil, fmt.Errorf("backend dispatch bind address must be an IP address")
	}
	if options.Port < 0 || options.Port > 65535 {
		return nil, fmt.Errorf("backend dispatch port must be between 0 and 65535")
	}
	if isNil(options.Handler) {
		return nil, fmt.Errorf("backend dispatch handler is required")
	}
	if options.Readiness == nil {
		return nil, fmt.Errorf("backend dispatch readiness is required")
	}
	if options.ReadHeaderTimeout == 0 {
		options.ReadHeaderTimeout = defaultReadHeaderTimeout
	}
	if options.IdleTimeout == 0 {
		options.IdleTimeout = defaultIdleTimeout
	}
	if options.ShutdownTimeout == 0 {
		options.ShutdownTimeout = defaultShutdownTimeout
	}
	if options.ReadHeaderTimeout <= 0 || options.IdleTimeout <= 0 || options.ShutdownTimeout <= 0 {
		return nil, fmt.Errorf("backend dispatch server timeouts must be positive")
	}
	return &Server{options: options, serveDone: make(chan struct{})}, nil
}

// Start binds the listener before returning. Request contexts are rooted in
// ctx; lifecycle shutdown remains explicit through Close so the owner can
// drain ExtProc before physical backend streams.
func (server *Server) Start(ctx context.Context) error {
	if server == nil {
		return errors.New("backend dispatch server is unavailable")
	}
	if ctx == nil {
		return errors.New("backend dispatch context is required")
	}
	server.mu.Lock()
	defer server.mu.Unlock()
	if server.closed {
		return errors.New("backend dispatch server is closed")
	}
	if server.started {
		return errors.New("backend dispatch server is already started")
	}
	address := net.JoinHostPort(server.options.BindAddress, strconv.Itoa(server.options.Port))
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("listen for backend dispatch: %w", err)
	}
	server.listener = listener
	server.address = listener.Addr().String()
	server.server = &http.Server{
		Handler:           server,
		ReadHeaderTimeout: server.options.ReadHeaderTimeout,
		IdleTimeout:       server.options.IdleTimeout,
		MaxHeaderBytes:    64 << 10,
		BaseContext:       func(net.Listener) context.Context { return ctx },
	}
	server.started = true
	go server.serve(listener)
	return nil
}

// ServeHTTP reserves one private readiness route for data-plane membership.
// Every other request remains behind the capability-verifying dispatch
// handler. Readiness errors are deliberately redacted from the wire.
func (server *Server) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	if request.URL.Path != "/ready" {
		server.options.Handler.ServeHTTP(writer, request)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	if request.Method != http.MethodGet && request.Method != http.MethodHead {
		writer.Header().Set("Allow", "GET, HEAD")
		writer.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	if err := server.options.Readiness(request.Context()); err != nil {
		writer.WriteHeader(http.StatusServiceUnavailable)
		if request.Method != http.MethodHead {
			_, _ = writer.Write([]byte(`{"ready":false}`))
		}
		return
	}
	writer.WriteHeader(http.StatusOK)
	if request.Method != http.MethodHead {
		_, _ = writer.Write([]byte(`{"ready":true}`))
	}
}

func (server *Server) serve(listener net.Listener) {
	err := server.server.Serve(listener)
	server.mu.Lock()
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		server.serveErr = fmt.Errorf("serve backend dispatch: %w", err)
	}
	server.mu.Unlock()
	close(server.serveDone)
}

func (server *Server) Ready() error {
	if server == nil {
		return errors.New("backend dispatch server is unavailable")
	}
	server.mu.RLock()
	defer server.mu.RUnlock()
	if server.closed {
		return errors.New("backend dispatch server is closed")
	}
	if !server.started || server.listener == nil {
		return errors.New("backend dispatch server has not started")
	}
	return server.serveErr
}

func (server *Server) Address() string {
	if server == nil {
		return ""
	}
	server.mu.RLock()
	defer server.mu.RUnlock()
	return server.address
}

func (server *Server) Close() error {
	if server == nil {
		return nil
	}
	server.closeOnce.Do(func() {
		server.mu.Lock()
		server.closed = true
		started := server.started
		httpServer := server.server
		timeout := server.options.ShutdownTimeout
		server.mu.Unlock()
		if !started || httpServer == nil {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		shutdownErr := httpServer.Shutdown(ctx)
		if shutdownErr != nil {
			_ = httpServer.Close()
		}
		<-server.serveDone
		server.mu.RLock()
		serveErr := server.serveErr
		server.mu.RUnlock()
		server.closeErr = errors.Join(shutdownErr, serveErr)
	})
	return server.closeErr
}
