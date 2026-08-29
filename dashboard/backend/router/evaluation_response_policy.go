package router

import (
	"net/http"
	"strings"
)

const evaluationAPIPath = "/api/evaluation/v1"

// withEvaluationResponsePolicy prevents authenticated evaluation evidence and
// status responses from being retained by browsers or intermediary caches. It
// intentionally wraps authentication as well as the route mux so rejected
// requests receive the same policy as successful API, artifact, and SSE
// responses.
func withEvaluationResponsePolicy(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == evaluationAPIPath || strings.HasPrefix(r.URL.Path, evaluationAPIPath+"/") {
			w.Header().Set("Cache-Control", "private, no-store")
			w.Header().Set("Pragma", "no-cache")
		}
		next.ServeHTTP(w, r)
	})
}
