package main

import (
	"net/http"
	"testing"
)

func TestDashboardHTTPServerBoundsRequestAdmission(t *testing.T) {
	t.Parallel()

	handler := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	server := newDashboardHTTPServer(":9999", handler)
	if server.Handler == nil || server.Addr != ":9999" {
		t.Fatalf("server identity = (%q, %v), want configured address and handler", server.Addr, server.Handler)
	}
	if server.ReadHeaderTimeout != dashboardReadHeaderTimeout {
		t.Fatalf("ReadHeaderTimeout = %s, want %s", server.ReadHeaderTimeout, dashboardReadHeaderTimeout)
	}
	if server.ReadTimeout != dashboardReadTimeout {
		t.Fatalf("ReadTimeout = %s, want %s", server.ReadTimeout, dashboardReadTimeout)
	}
	if server.IdleTimeout != dashboardIdleTimeout {
		t.Fatalf("IdleTimeout = %s, want %s", server.IdleTimeout, dashboardIdleTimeout)
	}
	if server.MaxHeaderBytes != dashboardMaxHeaderBytes {
		t.Fatalf("MaxHeaderBytes = %d, want %d", server.MaxHeaderBytes, dashboardMaxHeaderBytes)
	}
	if server.WriteTimeout != 0 {
		t.Fatalf("WriteTimeout = %s, want 0 for WebSocket/SSE routes", server.WriteTimeout)
	}
}
