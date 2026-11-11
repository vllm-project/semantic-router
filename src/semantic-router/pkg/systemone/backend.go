package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
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
			if config.CanonicalAPIFormat(model.APIFormat) != config.APIFormatChatCompletions && model.APIFormat != "" {
				return 0, nil, errors.New("judge requires OpenAI Chat Completions")
			}
			protocol = "openai/chat-completions@1"
		}
		path, err := profile.ResolveCreatePath(protocol)
		if err != nil {
			return 0, nil, err
		}
		header := http.Header{"Content-Type": {"application/json"}}
		if model.APIFormat == config.APIFormatSystemOne {
			header.Set(headers.VSRSystemOneBackend, "1")
		}
		endpoint, _ := cfg.GetEndpointByName(endpointName)
		key := endpoint.APIKey
		if profile.Delegated() {
			// A call made for a client request presents its delegated
			// credential; one without it, or a Router-own call, is never sent.
			credential, delegatedErr := principal.Delegated(ctx)
			if errors.Is(delegatedErr, principal.ErrCredentialMissing) {
				return http.StatusUnauthorized, nil, delegatedErr
			}
			if delegatedErr != nil {
				return http.StatusForbidden, nil, delegatedErr
			}
			key = credential
		}
		if key != "" {
			name, prefix, authErr := profile.ResolveAuthHeader()
			if authErr != nil {
				return 0, nil, authErr
			}
			value := key
			if prefix != "" {
				value = prefix + " " + value
			}
			header.Set(name, value)
		}
		send := func(ctx context.Context, callPath string, callHeader http.Header, encoded []byte) (int, http.Header, []byte, error) {
			sent := header.Clone()
			for name, values := range callHeader {
				sent[name] = values
			}
			response, callErr := remote.Do(ctx, &upstream.Request{
				Method: http.MethodPost, Path: path + callPath, Header: sent, Body: encoded,
				RouteKey: alias, ExactRoute: true, Listener: listener,
			})
			if callErr != nil {
				return 0, nil, nil, callErr
			}
			defer response.Body.Close()
			data, readErr := io.ReadAll(io.LimitReader(response.Body, (4<<20)+1))
			if readErr != nil || len(data) > 4<<20 {
				return 0, nil, nil, errors.New("invalid inference response size")
			}
			return response.StatusCode, response.Header, data, nil
		}
		external := cfg.ResolveExternalModelID(alias, endpointName)
		if model.APIFormat == config.APIFormatSystemOne {
			// A vllm-sr/4.0 backend gets the internal request as it is,
			// unless roles it cannot carry must come out first.
			schema := systemonecodec.SchemaID(cfg.UpstreamSchema(alias))
			if schema != systemonecodec.SchemaVLLMSR40 || hasRoles(body) {
				return forwardThrough(ctx, schema, external, alias, body, send)
			}
		}
		var request map[string]json.RawMessage
		if json.Unmarshal(body, &request) != nil || request == nil {
			return 0, nil, errors.New("invalid inference body")
		}
		request["model"], _ = json.Marshal(external)
		encoded, err := json.Marshal(request)
		if err != nil {
			return 0, nil, err
		}
		status, _, data, err := send(ctx, "", nil, encoded)
		return status, data, err
	}
}

// forwardThrough asks an upstream System One backend in its schema: the
// request, in the internal form, read into the IR and asked through the
// schema's codec (one batch call, or one call per request), and its answers
// written back in the internal form.
func forwardThrough(ctx context.Context, schema systemonecodec.SchemaID, model, alias string, body []byte,
	send func(context.Context, string, http.Header, []byte) (int, http.Header, []byte, error),
) (int, []byte, error) {
	codec, _, err := systemonecodec.Builtin().Resolve(schema, nil)
	if err != nil {
		return 0, nil, err
	}
	internal := systemonecodec.InternalCodec{}
	batch, err := internal.DecodeRequest(body)
	if err != nil {
		status, encoded := systemonecodec.NativeErrorBody(&systemoneprotocol.Problem{Status: http.StatusBadRequest, Kind: systemoneprotocol.ProblemRejected, Detail: err.Error()}, "")
		return status, encoded, nil
	}
	form := batch.Form
	batch.Model = model
	names := make([]string, len(batch.Items))
	for index := range batch.Items {
		batch.Items[index].Request.Model = ""
		names[index] = batch.Items[index].Name
	}
	exchange := systemonecodec.Exchange{
		Codec: codec, Target: systemonecodec.Target{Deployment: alias, Model: model},
		Policy:         systemonecodec.Policy{StripUnsupported: true, Presets: modelservice.CanonicalPreset},
		MaxConcurrency: systemonecodec.DefaultMaxConcurrency,
		Send: func(ctx context.Context, call systemonecodec.WireCall, encoded []byte) (systemonecodec.WireReply, error) {
			status, header, data, err := send(ctx, call.Path, call.Header, encoded)
			if err != nil {
				return systemonecodec.WireReply{}, err
			}
			return systemonecodec.WireReply{Status: status, Header: header, Body: data}, nil
		},
	}
	results := exchange.Ask(ctx, batch, nil)
	if systemonecodec.AllTransport(results) {
		return 0, nil, results[0].Err
	}
	return internal.EncodeResults(results, names, form, model)
}

// hasRoles reports whether an internal request names its parts' roles, or
// none.
func hasRoles(body []byte) bool {
	batch, err := systemonecodec.InternalCodec{}.DecodeRequest(body)
	if err != nil {
		return false
	}
	for _, item := range batch.Items {
		if item.Request.Roles.Mode != systemoneprotocol.RolesImplicit {
			return true
		}
	}
	return false
}
