package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

const decisionModelRoutesPath = "/api/v1/diagnostics/routes/systemone"

// DecisionModelRoutesHandler delegates operator diagnostics to the active
// frontend. The Router owns route admission, lifecycle and execution budgets;
// no public listener credential or browser-selected endpoint is forwarded.
func DecisionModelRoutesHandler(upstream string, providers ...routerauth.CredentialProvider) http.HandlerFunc {
	transport := decisionModelTransport{
		upstream: strings.TrimRight(upstream, "/"),
		client:   &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
	}
	if len(providers) > 0 {
		transport.provider = providers[0]
	}
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			decisionModelError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Use GET to discover routes or POST to run a native route")
			return
		}
		var body []byte
		ctx := r.Context()
		if r.Method == http.MethodGet {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
		} else {
			var input struct {
				Model   string          `json:"model"`
				Request json.RawMessage `json:"request"`
			}
			decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, decisionModelRequestLimit))
			decoder.DisallowUnknownFields()
			err := decoder.Decode(&input)
			var limit *http.MaxBytesError
			if errors.As(err, &limit) {
				decisionModelError(w, http.StatusRequestEntityTooLarge, "request_too_large", "The request exceeds 2 MiB")
				return
			}
			var request map[string]json.RawMessage
			if err != nil || strings.TrimSpace(input.Model) == "" || json.Unmarshal(input.Request, &request) != nil || request == nil || decoder.Decode(new(any)) != io.EOF {
				decisionModelError(w, http.StatusBadRequest, "invalid_request", "Provide one model and a native request object")
				return
			}
			body, err = json.Marshal(input)
			if err != nil {
				decisionModelError(w, http.StatusBadRequest, "invalid_request", "Unable to encode the native request")
				return
			}
		}
		response, data, err := transport.request(ctx, r.Method, decisionModelRoutesPath, body)
		if err != nil {
			decisionModelTransportError(w, err)
			return
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 && response.StatusCode < 400 || !json.Valid(data) {
			decisionModelError(w, http.StatusBadGateway, "invalid_response", "The frontend returned an invalid native route response")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.StatusCode)
		_, _ = w.Write(data)
	}
}
