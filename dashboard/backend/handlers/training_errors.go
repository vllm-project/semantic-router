package handlers

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"strings"

	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

// Middleware and net/http errors are not public diagnostics. Keep their client
// messages fixed; handlers explicitly construct APIError for safe validation details.
var trainingFallbackErrors = map[int]c.APIError{
	http.StatusBadRequest:                   {Code: "invalid_request", Message: "invalid request"},
	http.StatusUnauthorized:                 {Code: "unauthenticated", Message: "authentication required"},
	http.StatusForbidden:                    {Code: "forbidden", Message: "permission denied"},
	http.StatusNotFound:                     {Code: "not_found", Message: "training resource not found"},
	http.StatusMethodNotAllowed:             {Code: "method_not_allowed", Message: "method not allowed"},
	http.StatusConflict:                     {Code: "conflict", Message: "training conflict"},
	http.StatusRequestEntityTooLarge:        {Code: "payload_too_large", Message: "request body too large"},
	http.StatusRequestedRangeNotSatisfiable: {Code: "invalid_range", Message: "invalid byte range"},
	http.StatusInternalServerError:          {Code: "internal_error", Message: "training operation failed"},
	http.StatusServiceUnavailable:           {Code: "unavailable", Message: "training service unavailable"},
}

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
			var ok bool
			value, ok = trainingFallbackErrors[response.status]
			if !ok {
				value = c.APIError{Code: "request_failed", Message: "training request failed"}
			}
			log.Printf("training API middleware: status=%d code=%s detail=%q", response.status, value.Code, response.body.String())
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
