package router

import (
	"net/http"
	"strings"
)

const srBenchAPIPath = "/api/sr-bench/v1"

// withSRBenchResponsePolicy prevents authenticated benchmark evidence and
// status responses from being retained by browsers or intermediary caches. It
// intentionally wraps authentication as well as the route mux so rejected
// requests receive the same policy as successful API and event
// responses.
func withSRBenchResponsePolicy(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == srBenchAPIPath || strings.HasPrefix(r.URL.Path, srBenchAPIPath+"/") {
			w.Header().Set("Cache-Control", "private, no-store")
			w.Header().Set("Pragma", "no-cache")
		}
		next.ServeHTTP(w, r)
	})
}
