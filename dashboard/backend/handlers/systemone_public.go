package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

// PublicSystemOneHandler keeps the native surface stable across instance modes.
// The active frontend owns listener authorization and model lifetime together.
func PublicSystemOneHandler(upstream string, providers ...routerauth.CredentialProvider) http.HandlerFunc {
	transport := decisionModelTransport{upstream: strings.TrimRight(upstream, "/"), client: &http.Client{Timeout: 35 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	if len(providers) > 0 {
		transport.provider = providers[0]
	}
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		forwarded := systemone.ForwardRequest{Listener: os.Getenv("VLLM_SR_SYSTEMONE_LISTENER"), Method: r.Method, Path: r.URL.Path, Authorization: r.Header.Get("Authorization"), APIKey: r.Header.Get("Api-Key")}
		if !forwarded.ValidOperation() {
			decisionModelError(w, 405, "method_not_allowed", "Unsupported native operation")
			return
		}
		if r.Method == http.MethodPost {
			body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, decisionModelRequestLimit))
			if err != nil || !json.Valid(body) {
				decisionModelError(w, 400, "invalid_request", "Unable to read bounded native request")
				return
			}
			forwarded.Request = body
		}
		var payload bytes.Buffer
		encoder := json.NewEncoder(&payload)
		encoder.SetEscapeHTML(false)
		// #nosec G117 -- Forward client credentials only over the authenticated management transport for active listener authorization; never log or persist them.
		if err := encoder.Encode(forwarded); err != nil {
			decisionModelError(w, 400, "invalid_request", "Unable to encode native request")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 35*time.Second)
		defer cancel()
		response, data, err := transport.request(ctx, http.MethodPost, systemone.ForwardPath, payload.Bytes())
		if err != nil || response.StatusCode < 200 || response.StatusCode >= 300 && response.StatusCode < 400 || !json.Valid(data) {
			decisionModelError(w, 503, "systemone_unavailable", "The native frontend is unavailable")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if challenge := response.Header.Get("WWW-Authenticate"); challenge != "" {
			w.Header().Set("WWW-Authenticate", challenge)
		}
		w.WriteHeader(response.StatusCode)
		_, _ = w.Write(data)
	}
}
