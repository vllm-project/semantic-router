//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"net"
	"net/http"
	"time"
)

const managementShutdownTimeout = 10 * time.Second

// serveManagementListener keeps bind, transport selection, and shutdown
// ownership in one narrow boundary. A prevalidated TLS context enables TLS;
// otherwise the operational or file-authoritative listener stays plaintext.
func serveManagementListener(
	ctx context.Context,
	server *http.Server,
	onListening func(),
) error {
	if server == nil {
		return errors.New("management HTTP server is required")
	}
	listener, serveManagementListenerErr := net.Listen("tcp", server.Addr)
	if serveManagementListenerErr != nil {
		return serveManagementListenerErr
	}
	defer listener.Close()
	if onListening != nil {
		onListening()
	}

	if ctx == nil {
		ctx = context.Background()
	}
	serveReturned := make(chan struct{})
	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		select {
		case <-ctx.Done():
			shutdownContext, cancel := context.WithTimeout(context.Background(), managementShutdownTimeout)
			defer cancel()
			if err := server.Shutdown(shutdownContext); err != nil {
				_ = server.Close()
			}
		case <-serveReturned:
		}
	}()

	if server.TLSConfig != nil {
		serveManagementListenerErr = server.ServeTLS(listener, "", "")
	} else {
		serveManagementListenerErr = server.Serve(listener)
	}
	close(serveReturned)
	<-shutdownDone
	if errors.Is(serveManagementListenerErr, http.ErrServerClosed) {
		return nil
	}
	return serveManagementListenerErr
}
