package main

import (
	"net"
	"net/http"
	"net/url"

	"github.com/mark3labs/mcp-go/server"
)

const endpointPath = "/mcp"

// newHTTPHandler serves the MCP server over streamable HTTP at /mcp.
// Stateless: the server keeps nothing per session, so a restart needs no reconnect.
func newHTTPHandler(p Provider) http.Handler {
	mcpServer := server.NewStreamableHTTPServer(newServer(p), server.WithEndpointPath(endpointPath), server.WithStateLess(true))
	return loopbackOriginsOnly(mcpServer)
}

// loopbackOriginsOnly refuses browser requests from other sites. A page whose
// hostname is rebound to 127.0.0.1 could otherwise drive searches through the
// operator's provider; the coding agents this serves send no Origin header.
func loopbackOriginsOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if origin := r.Header.Get("Origin"); origin != "" && !isLoopbackOrigin(origin) {
			http.Error(w, "origin not allowed", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func isLoopbackOrigin(origin string) bool {
	parsed, err := url.Parse(origin)
	if err != nil {
		return false
	}
	host := parsed.Hostname()
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
