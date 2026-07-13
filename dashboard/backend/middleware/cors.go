package middleware

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/browsersecurity"
)

// HandleCORSPreflight sets CORS headers and returns true if the request is an OPTIONS preflight that was handled.
func HandleCORSPreflight(w http.ResponseWriter, r *http.Request) bool {
	origin := r.Header.Get("Origin")
	originAllowed := origin != "" && browsersecurity.ValidOrigin(r)
	if originAllowed {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Vary", "Origin")
		// Only set credentials when echoing back a specific origin
		w.Header().Set("Access-Control-Allow-Credentials", "true")
	}
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length, Content-Range")

	// Add Private Network Access (PNA) headers to allow public pages to access private network resources
	// This is required when accessing the dashboard via a public domain that proxies to local services
	// Chrome's Private Network Access policy requires these headers for preflight requests
	// See: https://developer.chrome.com/blog/private-network-access-preflight/
	if originAllowed && r.Header.Get("Access-Control-Request-Private-Network") == "true" {
		w.Header().Set("Access-Control-Allow-Private-Network", "true")
	}

	if r.Method == http.MethodOptions {
		if origin != "" && !originAllowed {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return true
		}
		w.WriteHeader(http.StatusNoContent)
		return true
	}
	return false
}
