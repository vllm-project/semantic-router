// Package systemone publishes the native inference contract under one listener grant.
package systemone

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// RequestLimit bounds one native request body. Requests may carry images and
// videos as base64 data URLs (one video of up to 32,000,000 bytes fits); the
// managed runtime accepts larger bodies than this.
const RequestLimit = 48 << 20

// A request the public API sends to its model directly waits publicWait for
// the answer; its options.deadline_ms extends that, up to the listener's
// systemone.max_deadline_ms, for requests that take longer, such as
// long-context ones.
const publicWait = 30 * time.Second

// Invoke is a retained inference call supplied by the serving owner.
type Invoke func(context.Context, string, json.RawMessage) (int, []byte, error)

// Handler serves only the selected listener's native model grant. Authentication
// is independent of Dashboard sessions and of the listener's Chat model grant.
func Handler(config *routerconfig.RouterConfig, listener *routerconfig.Listener, invoke Invoke) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		listing := r.Method == http.MethodGet && (r.URL.Path == "/v1/systemone/models" || r.URL.Path == "/v1/models")
		if r.Method != http.MethodPost && !listing {
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
			models := publishedModels(config, listener)
			list := map[string]any{"object": "list", "data": models}
			if r.URL.Path == "/v1/models" {
				// System One SDKs list models as {models: [{name, description, release_date}]}.
				listed := make([]map[string]string, 0, len(models))
				for _, model := range models {
					listed = append(listed, map[string]string{"name": model["id"].(string), "description": "System One decision model", "release_date": ""})
				}
				list["models"] = listed
			}
			_ = json.NewEncoder(w).Encode(list)
			return
		}
		encoding, problem := requestEncoding(r.Header)
		if problem != nil {
			writeWireError(w, problem)
			return
		}
		body, err := readRequestBody(w, r)
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			publicSystemOneError(w, http.StatusRequestEntityTooLarge, "request_too_large")
			return
		}
		if err != nil {
			publicSystemOneError(w, 400, "invalid_request")
			return
		}
		if encoding != "" {
			if body, problem = decodeBody(r.Context(), encoding, body); problem != nil {
				writeWireError(w, problem)
				return
			}
		}
		boundary, problem := multipartBoundary(r.Header.Get("Content-Type"))
		if problem == nil && boundary != "" {
			body, problem = canonicalMultipart(body, boundary)
		}
		if problem != nil {
			writeWireError(w, problem)
			return
		}
		request, valid := scanRequest(body)
		if !valid {
			publicSystemOneError(w, 400, "invalid_request")
			return
		}
		model := request.model
		if model == "" {
			publicSystemOneError(w, 400, "model_required")
			return
		}
		if !slices.Contains(listener.SystemOne.Models, model) {
			publicSystemOneErrorMessage(w, 403, "model_not_allowed", modelNotAllowed(model, listener.SystemOne.Models))
			return
		}
		if request.repeatedOptions {
			publicSystemOneErrorMessage(w, 400, "invalid_request", "options may appear only once")
			return
		}
		limit := listener.SystemOne.MaxDeadline()
		wait := requestWait(nil, limit)
		if request.options != nil {
			options, problem := publicOptions(request.options)
			if problem != "" {
				publicSystemOneErrorMessage(w, 400, "invalid_request", problem)
				return
			}
			body = request.withOptions(body, options)
			wait = requestWait(options, limit)
		}
		_, routed := config.ResolveEntrypoint(routerconfig.SystemOneAPI, model)
		if routed && !config.RoutingEnabled() {
			publicSystemOneError(w, http.StatusNotFound, "systemone_routing_disabled")
			return
		}
		if r.Header.Get(BackendRequestHeader) != "" {
			remoteAlias := config.IsSystemOneBackend(model) && config.ModelConfig[model].Deployment == ""
			if routed || remoteAlias {
				publicSystemOneError(w, http.StatusConflict, "systemone_nested_routing")
				return
			}
		}
		target := model
		if !routed && !config.IsSystemOneBackend(model) {
			target, _, err = config.ResolveSystemOneDeployment(model)
			if err != nil {
				publicSystemOneError(w, 404, "model_not_found")
				return
			}
		}
		ctx := r.Context()
		if !routed {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, wait)
			defer cancel()
		}
		code, response, err := invoke(ctx, target, body)
		if routed && errors.Is(err, ErrUnresolved) {
			publicSystemOneError(w, http.StatusServiceUnavailable, "systemone_unresolved")
			return
		}
		if errors.Is(err, context.DeadlineExceeded) {
			if routed {
				publicSystemOneError(w, http.StatusGatewayTimeout, "systemone_deadline_exceeded")
			} else {
				publicSystemOneErrorMessage(w, http.StatusGatewayTimeout, "systemone_deadline_exceeded", deadlineExceeded(wait, limit))
			}
			return
		}
		if err != nil && (code < 400 || code > 599) {
			publicSystemOneError(w, 503, "systemone_unavailable")
			return
		}
		systemOne := r.URL.Path == "/v1/systemone"
		if code < 200 || code >= 300 {
			if failure, failed := refusedQuestion(code, response, systemOne); failed {
				writeQuestionFailure(w, failure)
				return
			}
			status, errorCode, message, reason, limit := runtimeFailure(code, response)
			fields := map[string]any{"code": errorCode, "type": "systemone_error", "reason": reason, "message": message}
			if limit != nil {
				fields["limit"] = *limit
			}
			writeErrorBody(w, status, fields)
			return
		}
		var result map[string]json.RawMessage
		if len(response) > 4<<20 || json.Unmarshal(response, &result) != nil || result == nil {
			publicSystemOneError(w, 502, "invalid_runtime_response")
			return
		}
		if systemOne {
			if failure, failed := failedQuestion(result); failed {
				writeQuestionFailure(w, failure)
				return
			}
		}
		if aliasErr := api.AliasResponseModel(result, model); aliasErr != nil {
			publicSystemOneError(w, 502, "invalid_runtime_response")
			return
		}
		_ = json.NewEncoder(w).Encode(result)
	}
}

// publishedModels are the listener's System One models that can be served now.
func publishedModels(config *routerconfig.RouterConfig, listener *routerconfig.Listener) []map[string]any {
	models := make([]map[string]any, 0, len(listener.SystemOne.Models))
	for _, id := range listener.SystemOne.Models {
		_, routed := config.ResolveEntrypoint(routerconfig.SystemOneAPI, id)
		if routed && !config.RoutingEnabled() {
			continue
		}
		_, _, err := config.ResolveSystemOneDeployment(id)
		if routed || config.IsSystemOneBackend(id) || err == nil {
			models = append(models, map[string]any{"id": id, "object": "model", "api": "systemone", "routing": routed})
		}
	}
	return models
}

// modelNotAllowed names the models a client may use, so a client sending a
// model this API does not serve learns which to set instead.
func modelNotAllowed(model string, allowed []string) string {
	shown := model
	if len(shown) > 64 {
		shown = shown[:64]
	}
	return fmt.Sprintf("model %q is not served here; this API serves %s", shown, strings.Join(allowed, ", "))
}

// readRequestBody reads a body of at most RequestLimit bytes into one buffer
// sized by its declared length, so a large media request is copied once.
func readRequestBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	reader := http.MaxBytesReader(w, r.Body, RequestLimit)
	if r.ContentLength <= 0 || r.ContentLength > RequestLimit {
		return io.ReadAll(reader)
	}
	buffer := bytes.NewBuffer(make([]byte, 0, r.ContentLength+bytes.MinRead))
	_, err := buffer.ReadFrom(reader)
	return buffer.Bytes(), err
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
	apiKey := r.Header.Get("Api-Key")
	valid := 0
	for _, key := range keys {
		if bearer != "" {
			valid |= subtle.ConstantTimeCompare([]byte(key), []byte(bearer))
		}
		if apiKey != "" {
			valid |= subtle.ConstantTimeCompare([]byte(key), []byte(apiKey))
		}
	}
	return valid == 1
}

func publicSystemOneError(w http.ResponseWriter, status int, code string) {
	publicSystemOneErrorMessage(w, status, code, strings.ReplaceAll(code, "_", " "))
}

// retryAfterSeconds tells a client when a temporarily refused request may succeed:
// a full model at once, a model that is starting or restarting a little later.
var retryAfterSeconds = map[string]string{
	string(api.Overloaded):  "1",
	string(api.NotReady):    "5",
	"systemone_unavailable": "5",
}

func writeWireError(w http.ResponseWriter, problem *wireError) {
	writeErrorBody(w, problem.status, map[string]any{"code": problem.code, "type": "systemone_error", "reason": problem.reason, "message": problem.message})
}

func writeErrorBody(w http.ResponseWriter, status int, fields map[string]any) {
	for key, value := range fields {
		if value == "" {
			delete(fields, key)
		}
	}
	if code, _ := fields["code"].(string); retryAfterSeconds[code] != "" {
		w.Header().Set("Retry-After", retryAfterSeconds[code])
	}
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": fields})
}

func publicSystemOneErrorMessage(w http.ResponseWriter, status int, code, message string) {
	if seconds, ok := retryAfterSeconds[code]; ok {
		w.Header().Set("Retry-After", seconds)
	}
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": code, "type": "systemone_error", "message": message}})
}
