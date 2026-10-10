package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// ServingInvoker applies API identity before dispatching direct native models
// or an auto recipe. Listener authorization stays in Handler.
func ServingInvoker(cfg *config.RouterConfig, listener string, router Router, local Invoke, remote *upstream.Set) Invoke {
	backend := BackendInvoker(cfg, listener, local, remote)
	return func(ctx context.Context, target string, body json.RawMessage) (int, []byte, error) {
		if _, routed := cfg.ResolveEntrypoint(config.SystemOneAPI, target); routed {
			if router == nil || !cfg.RoutingEnabled() {
				return 0, nil, errors.New("native routing unavailable")
			}
			return router.RouteSystemOne(ctx, target, body, backend)
		}
		if cfg.IsSystemOneBackend(target) {
			return backend(ctx, target, body)
		}
		if local == nil {
			return 0, nil, errors.New("native runtime unavailable")
		}
		return local(ctx, target, body)
	}
}

// BackendInvoker binds one generation's provider aliases to its retained local
// runtimes or upstream set. No unknown alias may fall through to a Chat default.
func BackendInvoker(cfg *config.RouterConfig, listener string, local Invoke, remote *upstream.Set) Invoke {
	return func(ctx context.Context, alias string, body json.RawMessage) (int, []byte, error) {
		model, exists := cfg.ModelConfig[alias]
		if !exists {
			return 0, nil, errors.New("unknown inference backend")
		}
		if model.Deployment != "" {
			if local == nil {
				return 0, nil, errors.New("local inference runtime unavailable")
			}
			return local(ctx, model.Deployment, body)
		}
		if remote == nil {
			return 0, nil, errors.New("upstream inference runtime unavailable")
		}
		_, endpointName, found, err := cfg.ResolvePrimaryBackendForModel(alias)
		if err != nil || !found {
			return 0, nil, errors.New("inference backend binding unavailable")
		}
		profile, err := cfg.GetProviderProfileForEndpoint(endpointName)
		if err != nil {
			return 0, nil, err
		}
		protocol := "vllm-sr/systemone@1"
		if model.APIFormat != config.APIFormatSystemOne {
			if model.APIFormat != "openai" && model.APIFormat != "" {
				return 0, nil, errors.New("judge requires OpenAI Chat Completions")
			}
			protocol = "openai/chat-completions@1"
		}
		path, err := profile.ResolveCreatePath(protocol)
		if err != nil {
			return 0, nil, err
		}
		var request map[string]json.RawMessage
		if json.Unmarshal(body, &request) != nil || request == nil {
			return 0, nil, errors.New("invalid inference body")
		}
		request["model"], _ = json.Marshal(cfg.ResolveExternalModelID(alias, endpointName))
		encoded, err := json.Marshal(request)
		if err != nil {
			return 0, nil, err
		}
		header := http.Header{"Content-Type": {"application/json"}}
		if model.APIFormat == config.APIFormatSystemOne {
			header.Set(BackendRequestHeader, "1")
		}
		endpoint, _ := cfg.GetEndpointByName(endpointName)
		if endpoint.APIKey != "" {
			name, prefix, authErr := profile.ResolveAuthHeader()
			if authErr != nil {
				return 0, nil, authErr
			}
			value := endpoint.APIKey
			if prefix != "" {
				value = prefix + " " + value
			}
			header.Set(name, value)
		}
		response, err := remote.Do(ctx, &upstream.Request{
			Method: http.MethodPost, Path: path, Header: header, Body: encoded,
			RouteKey: alias, ExactRoute: true, Listener: listener,
		})
		if err != nil {
			return 0, nil, err
		}
		defer response.Body.Close()
		data, err := io.ReadAll(io.LimitReader(response.Body, (4<<20)+1))
		if err != nil || len(data) > 4<<20 {
			return 0, nil, errors.New("invalid inference response size")
		}
		return response.StatusCode, data, nil
	}
}
