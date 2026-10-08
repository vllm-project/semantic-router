// Package systemone publishes the native inference contract under one listener grant.
package systemone

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Invoke is a retained concrete deployment call, supplied by the serving owner.
type Invoke func(context.Context, string, json.RawMessage) (int, []byte, error)

// Handler serves only the selected listener's native model grant. Authentication
// is independent of Dashboard sessions and of the listener's Chat model grant.
func Handler(config *routerconfig.RouterConfig, listener *routerconfig.Listener, invoke Invoke) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if r.Method != http.MethodPost && !(r.Method == http.MethodGet && r.URL.Path == "/v1/systemone/models") {
			publicSystemOneError(w, 405, "method_not_allowed")
			return
		}
		if listener == nil || listener.SystemOne == nil || len(listener.SystemOne.Models) == 0 {
			publicSystemOneError(w, 404, "systemone_not_published")
			return
		}

		if !systemOneAuthorized(r, listener.APIKeys) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="vllm-semantic-router"`)
			publicSystemOneError(w, 401, "invalid_api_key")
			return
		}
		if r.Method == http.MethodGet {
			models := make([]map[string]any, 0, len(listener.SystemOne.Models))
			for _, id := range listener.SystemOne.Models {
				if _, _, err := config.ResolveSystemOneDeployment(id); err == nil {
					models = append(models, map[string]any{"id": id, "object": "model", "api": "systemone", "routing": false})
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": models})
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 2<<20))
		var request map[string]json.RawMessage
		if err != nil || json.Unmarshal(body, &request) != nil || request == nil {
			publicSystemOneError(w, 400, "invalid_request")
			return
		}
		var model string
		if json.Unmarshal(request["model"], &model) != nil || model == "" {
			publicSystemOneError(w, 400, "model_required")
			return
		}
		// API identity is explicit. Future routed SystemOne can use this same
		// resolver without treating a string that looks like 'auto' as a route.
		if _, routed := config.ResolveEntrypoint(routerconfig.SystemOneAPI, model); routed {
			publicSystemOneError(w, 400, "systemone_routing_not_supported")
			return
		}
		if !slices.Contains(listener.SystemOne.Models, model) {
			publicSystemOneError(w, 403, "model_not_allowed")
			return
		}
		deployment, _, err := config.ResolveSystemOneDeployment(model)
		if err != nil {
			publicSystemOneError(w, 404, "model_not_found")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()
		code, response, err := invoke(ctx, deployment, body)
		if err != nil && (code < 400 || code > 599) {
			publicSystemOneError(w, 503, "systemone_unavailable")
			return
		}
		if code < 200 || code >= 300 {
			// Runtime failures may contain local model paths or deployment keys.
			publicSystemOneError(w, code, "systemone_request_failed")
			return
		}
		var result map[string]json.RawMessage
		if len(response) > 4<<20 || json.Unmarshal(response, &result) != nil || result == nil {
			publicSystemOneError(w, 502, "invalid_runtime_response")
			return
		}
		result["model"], _ = json.Marshal(model)
		_ = json.NewEncoder(w).Encode(result)
	}
}

// SelectListener never unions grants from multiple listeners.
func SelectListener(listeners []routerconfig.Listener, name string) (*routerconfig.Listener, error) {
	var selected *routerconfig.Listener
	for index := range listeners {
		listener := &listeners[index]
		if listener.SystemOne == nil || len(listener.SystemOne.Models) == 0 {
			continue
		}
		if name != "" && listener.Name != name {
			continue
		}
		if selected != nil {
			return nil, errors.New("select an explicit SystemOne listener")
		}
		selected = listener
	}
	if selected == nil {
		return nil, errors.New("SystemOne is not published")
	}
	return selected, nil
}

func systemOneAuthorized(r *http.Request, keys []string) bool {
	if len(keys) == 0 {
		return true
	} // Same explicit open-listener semantics as Chat.
	value := r.Header.Get("Authorization")
	var bearer string
	if len(value) > 6 && (value[0] == 'B' || value[0] == 'b') && value[1:6] == "earer" {
		bearer = strings.TrimLeft(value[6:], " \t\n\v\f\r")
		if bearer == value[6:] {
			bearer = ""
		}
	}
	valid := 0
	for _, key := range keys {
		valid |= subtle.ConstantTimeCompare([]byte(key), []byte(bearer))
		valid |= subtle.ConstantTimeCompare([]byte(key), []byte(r.Header.Get("Api-Key")))
	}
	return valid == 1
}

func publicSystemOneError(w http.ResponseWriter, status int, code string) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": code, "type": "systemone_error", "message": strings.ReplaceAll(code, "_", " ")}})
}
