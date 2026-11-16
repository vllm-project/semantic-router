package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

// vsrRulesTryLimit bounds one try request: the listener's own rules limit,
// with room for JSON escapes.
const vsrRulesTryLimit = 1 << 20

// VSRRulesTryHandler asks a listener's rules about a sample text for the vSR
// panel's Try it, as the vSR plugins ask them (POST /v1/systemone/rules). The
// person never holds the listener's key: it is read here from the config and
// sent to the Router through the management transport, which checks it against
// the active listener. Using a stored key is what config write grants (the
// route requires it), and only a listener that binds rules is reached.
func VSRRulesTryHandler(configPath, upstream string, providers ...routerauth.CredentialProvider) http.HandlerFunc {
	transport := decisionModelTransport{upstream: strings.TrimRight(upstream, "/"), client: &http.Client{Timeout: 35 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	if len(providers) > 0 {
		transport.provider = providers[0]
	}
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		var request struct {
			Listener string          `json:"listener"`
			Request  json.RawMessage `json:"request"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, vsrRulesTryLimit))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil || request.Listener == "" || !json.Valid(request.Request) {
			decisionModelError(w, http.StatusBadRequest, "invalid_request", "Send {listener, request} with the rules request the plugins send")
			return
		}
		cfg, err := readCanonicalConfigFile(configPath)
		if err != nil {
			decisionModelError(w, http.StatusInternalServerError, "config_unavailable", "Unable to read the config")
			return
		}
		key, found := "", false
		for _, listener := range cfg.Listeners {
			if api, ok := listener.APIs["systemone"]; listener.Name == request.Listener && ok && api.VSR != nil && api.VSR.Rules != nil {
				found = true
				if len(listener.APIKeys) > 0 {
					key = listener.APIKeys[0]
				}
			}
		}
		if !found {
			decisionModelError(w, http.StatusNotFound, "listener_not_found", "The config has no System One listener of that name that binds vSR rules")
			return
		}
		forwarded := systemone.ForwardRequest{Listener: request.Listener, Method: http.MethodPost, Path: systemone.RulesPath, Request: request.Request}
		if key != "" {
			forwarded.Authorization = "Bearer " + key
		}
		var payload bytes.Buffer
		encoder := json.NewEncoder(&payload)
		encoder.SetEscapeHTML(false)
		// #nosec G117 -- The listener key goes only over the authenticated management transport; it is never logged or returned.
		if encodeErr := encoder.Encode(forwarded); encodeErr != nil {
			decisionModelError(w, http.StatusBadRequest, "invalid_request", "Unable to encode the rules request")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 35*time.Second)
		defer cancel()
		response, data, err := transport.request(ctx, http.MethodPost, systemone.ForwardPath, payload.Bytes())
		if err != nil {
			decisionModelTransportError(w, err)
			return
		}
		if !json.Valid(data) {
			decisionModelError(w, http.StatusBadGateway, "runtime_unavailable", "The Router answered with something other than JSON")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.StatusCode)
		_, _ = w.Write(data)
	}
}
