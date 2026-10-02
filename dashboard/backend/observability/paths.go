// Package observability shares embedded observability route classification
// between the proxy and its authentication boundary.
package observability

import "strings"

// IsGrafanaQueryPath identifies Grafana's POST data-source query endpoint.
func IsGrafanaQueryPath(path string) bool {
	return path == "/embedded/grafana/api/ds/query" || path == "/api/ds/query"
}

// IsJaegerAPIPath identifies the root API paths forwarded to Jaeger. Match a
// complete path segment so similarly named Dashboard routes cannot be captured.
func IsJaegerAPIPath(path string) bool {
	for _, prefix := range []string{"/api/services", "/api/traces", "/api/operations", "/api/dependencies"} {
		if path == prefix || strings.HasPrefix(path, prefix+"/") {
			return true
		}
	}
	return false
}
