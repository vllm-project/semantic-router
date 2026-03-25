package handlers

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// RouterClassifierProxyHandler forwards taxonomy-classifier management traffic
// to the router apiserver while keeping dashboard readonly enforcement intact.
func RouterClassifierProxyHandler(routerAPIURL string, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if routerAPIURL == "" {
			http.Error(w, "Router API URL is not configured", http.StatusBadGateway)
			return
		}
		if readonlyMode && r.Method != http.MethodGet {
			http.Error(w, "Dashboard is in read-only mode. Configuration editing is disabled.", http.StatusForbidden)
			return
		}

		targetURL := strings.TrimSuffix(routerAPIURL, "/") + strings.TrimPrefix(r.URL.Path, "/api/router")
		if r.URL.RawQuery != "" {
			targetURL += "?" + r.URL.RawQuery
		}

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read request body: %v", err), http.StatusBadRequest)
			return
		}

		proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(bodyBytes))
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to build router API request: %v", err), http.StatusInternalServerError)
			return
		}
		copyProxyHeaders(proxyReq.Header, r.Header)

		resp, err := http.DefaultClient.Do(proxyReq)
		if err != nil {
			http.Error(w, fmt.Sprintf("Router API request failed: %v", err), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		copyProxyHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	}
}

func copyProxyHeaders(dst, src http.Header) {
	for key, values := range src {
		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}
