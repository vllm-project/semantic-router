package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

// TrainingErrorEnvelope also gives shared auth, CSRF, method and body-limit
// rejections the v2 APIError shape. Successful downloads stream directly.
func TrainingErrorEnvelope(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != c.APIBasePath && !strings.HasPrefix(r.URL.Path, c.APIBasePath+"/") {
			next.ServeHTTP(w, r)
			return
		}
		response := &trainingErrorWriter{ResponseWriter: w}
		next.ServeHTTP(response, r)
		if response.status < 400 {
			return
		}
		var value c.APIError
		if err := json.Unmarshal(response.body.Bytes(), &value); err != nil || value.Code == "" {
			codes := map[int]string{400: "invalid_request", 401: "unauthenticated", 403: "forbidden", 404: "not_found", 405: "method_not_allowed", 409: "conflict", 413: "payload_too_large", 416: "invalid_range", 500: "internal_error", 503: "unavailable"}
			code := codes[response.status]
			if code == "" {
				code = "request_failed"
			}
			value = c.APIError{Code: code, Message: strings.TrimSpace(response.body.String())}
		}
		w.Header().Del("Content-Length")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.status)
		_ = json.NewEncoder(w).Encode(value)
	})
}

type trainingErrorWriter struct {
	http.ResponseWriter
	status int
	body   bytes.Buffer
}

func (w *trainingErrorWriter) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	if status < 400 {
		w.ResponseWriter.WriteHeader(status)
	}
}

func (w *trainingErrorWriter) Write(body []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(200)
	}
	if w.status >= 400 {
		return w.body.Write(body)
	}
	return w.ResponseWriter.Write(body)
}
func (w *trainingErrorWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
