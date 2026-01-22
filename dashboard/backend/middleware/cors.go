package middleware

import (
	"net/http"
)

// HandleCORSPreflight sets CORS headers and returns true if the request is an OPTIONS preflight that was handled.
func HandleCORSPreflight(w http.ResponseWriter, r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin != "" {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Vary", "Origin")
		// Only set credentials when echoing back a specific origin
		w.Header().Set("Access-Control-Allow-Credentials", "true")
	} else {
		w.Header().Set("Access-Control-Allow-Origin", "*")
	}
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length, Content-Range")

	// Add Private Network Access (PNA) headers to allow public pages to access private network resources
	// This is required when accessing the dashboard via a public domain that proxies to local services
	// Chrome's Private Network Access policy requires these headers for preflight requests
	// See: https://developer.chrome.com/blog/private-network-access-preflight/
	w.Header().Set("Access-Control-Allow-Private-Network", "true")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return true
	}
	return false
}
