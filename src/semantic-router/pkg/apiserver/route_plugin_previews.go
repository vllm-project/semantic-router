//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

type pluginBindingRef = pluginruntime.Binding

type systemPromptPreviewRequest struct {
	Configuration *config.SystemPromptPluginConfig `json:"configuration,omitempty"`
	Binding       *pluginBindingRef                `json:"binding,omitempty"`
	Format        llmprotocol.WireFormat           `json:"format,omitempty"`
	RequestBody   json.RawMessage                  `json:"request_body"`
}

type requestParamsPreviewRequest struct {
	Configuration *config.RequestParamsPluginConfig `json:"configuration,omitempty"`
	Binding       *pluginBindingRef                 `json:"binding,omitempty"`
	Format        llmprotocol.WireFormat            `json:"format,omitempty"`
	RequestBody   json.RawMessage                   `json:"request_body"`
}

type headerMutationPreviewRequest struct {
	Configuration *config.HeaderMutationPluginConfig `json:"configuration,omitempty"`
	Binding       *pluginBindingRef                  `json:"binding,omitempty"`
}

type fastResponsePreviewRequest struct {
	Configuration *config.FastResponsePluginConfig `json:"configuration,omitempty"`
	Binding       *pluginBindingRef                `json:"binding,omitempty"`
}

type pluginRequestPreviewResponse struct {
	pluginruntime.Guarantees
	Changed     bool                               `json:"changed"`
	Format      llmprotocol.WireFormat             `json:"format"`
	RequestBody json.RawMessage                    `json:"request_body"`
	Parameters  *pluginruntime.RequestParamsResult `json:"parameters,omitempty"`
	Diagnostics llmprotocol.Diagnostics            `json:"diagnostics,omitempty"`
}

type headerMutationPreviewResponse struct {
	pluginruntime.Guarantees
	Mutations      []pluginruntime.HeaderMutation `json:"mutations"`
	ValuesRedacted bool                           `json:"values_redacted"`
}

type fastResponsePreviewResponse struct {
	pluginruntime.Guarantees
	Message string `json:"message"`
}

func apiPluginPreviewRoutes() []apiRoute {
	policy := routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView}
	return []apiRoute{
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/system_prompt/preview", Method: "POST", Description: "Preview system instruction changes using the dispatch protocol codec; no persistence or backend calls"}, policy, (*ClassificationAPIServer).handleSystemPromptPreview, pluginOperationFor("system_prompt", "preview"), strictJSONBodyFor[systemPromptPreviewRequest](), jsonResponse[pluginRequestPreviewResponse](http.StatusOK, "Transformed request"), errorResponses(http.StatusBadRequest, http.StatusRequestEntityTooLarge)),
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/request_params/preview", Method: "POST", Description: "Preview the dispatch parameter policy, including blocked fields, defaults, and caps; no persistence or backend calls"}, policy, (*ClassificationAPIServer).handleRequestParamsPreview, pluginOperationFor("request_params", "preview"), strictJSONBodyFor[requestParamsPreviewRequest](), jsonResponse[pluginRequestPreviewResponse](http.StatusOK, "Transformed request and policy effects"), errorResponses(http.StatusBadRequest, http.StatusRequestEntityTooLarge)),
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/header_mutation/preview", Method: "POST", Description: "Preview ordered Envoy header operations; values require secret_view; no persistence or backend calls"}, policy, (*ClassificationAPIServer).handleHeaderMutationPreview, pluginOperationFor("header_mutation", "preview"), strictJSONBodyFor[headerMutationPreviewRequest](), jsonResponse[headerMutationPreviewResponse](http.StatusOK, "Ordered header mutation plan"), errorResponses(http.StatusBadRequest, http.StatusRequestEntityTooLarge)),
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/fast_response/preview", Method: "POST", Description: "Preview fixed assistant text before transport encoding; no persistence or backend calls"}, policy, (*ClassificationAPIServer).handleFastResponsePreview, pluginOperationFor("fast_response", "preview"), strictJSONBodyFor[fastResponsePreviewRequest](), jsonResponse[fastResponsePreviewResponse](http.StatusOK, "Fixed assistant text"), errorResponses(http.StatusBadRequest, http.StatusRequestEntityTooLarge)),
	}
}

func previewPluginConfig[T any](s *ClassificationAPIServer, pluginType string, candidate *T, binding *pluginBindingRef) (*T, error) {
	if (candidate == nil) == (binding == nil) {
		return nil, fmt.Errorf("provide exactly one of configuration or binding")
	}
	var plugin config.DecisionPlugin
	if candidate != nil {
		payload, err := config.NewStructuredPayload(candidate)
		if err != nil {
			return nil, err
		}
		plugin = config.DecisionPlugin{Type: pluginType, Configuration: payload}
	} else {
		if binding.Recipe == "" || binding.Decision == "" {
			return nil, fmt.Errorf("binding requires recipe and decision")
		}
		inventory, release, ok := s.pluginInventory()
		defer release()
		if !ok || inventory.Config == nil {
			return nil, fmt.Errorf("active router configuration is unavailable")
		}
		for _, ref := range inventory.Config.RoutingDecisionRefs() {
			if ref.Recipe != binding.Recipe || ref.Decision.Name != binding.Decision {
				continue
			}
			if configured := ref.Decision.GetPlugin(pluginType); configured != nil {
				plugin = *configured
			}
			break
		}
		if plugin.Configuration == nil {
			return nil, fmt.Errorf("plugin binding was not found in the specified recipe and decision")
		}
	}
	decoded, err := config.DecodeDecisionPlugin(plugin)
	if err != nil {
		return nil, err
	}
	result, ok := decoded.(*T)
	if !ok {
		return nil, fmt.Errorf("plugin configuration type does not match preview operation")
	}
	return result, nil
}

func (s *ClassificationAPIServer) writePluginPreviewError(w http.ResponseWriter, err error) {
	s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_PLUGIN_PREVIEW", scrubSecretsInErrorMessage(err.Error()))
}

func (s *ClassificationAPIServer) handleSystemPromptPreview(w http.ResponseWriter, r *http.Request) {
	var request systemPromptPreviewRequest
	if err := s.parseStrictJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	policy, err := previewPluginConfig(s, config.DecisionPluginSystemPrompt, request.Configuration, request.Binding)
	if err != nil {
		s.writePluginPreviewError(w, err)
		return
	}
	s.writeRequestPluginPreview(w, request.Format, request.RequestBody, func(semantic *llmprotocol.Request) (bool, *pluginruntime.RequestParamsResult, error) {
		if policy.SystemPrompt == "" || !config.DecisionPluginEnabled(policy) {
			return false, nil, nil
		}
		return llmprotocol.SetSystemInstruction(semantic, policy.SystemPrompt, policy.EffectiveMode()), nil, nil
	})
}

func (s *ClassificationAPIServer) handleRequestParamsPreview(w http.ResponseWriter, r *http.Request) {
	var request requestParamsPreviewRequest
	if err := s.parseStrictJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	policy, err := previewPluginConfig(s, config.DecisionPluginRequestParams, request.Configuration, request.Binding)
	if err != nil {
		s.writePluginPreviewError(w, err)
		return
	}
	s.writeRequestPluginPreview(w, request.Format, request.RequestBody, func(semantic *llmprotocol.Request) (bool, *pluginruntime.RequestParamsResult, error) {
		result, err := pluginruntime.ApplyRequestParams(semantic, policy)
		return result.Changed, &result, err
	})
}

func (s *ClassificationAPIServer) writeRequestPluginPreview(w http.ResponseWriter, format llmprotocol.WireFormat, body json.RawMessage, apply func(*llmprotocol.Request) (bool, *pluginruntime.RequestParamsResult, error)) {
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, diagnostics, err := engine.DecodeRequestForMutation(format, body)
	if err != nil {
		s.writePluginPreviewError(w, err)
		return
	}
	changed, params, err := apply(&request)
	if err != nil {
		s.writePluginPreviewError(w, err)
		return
	}
	if changed {
		request.Generation++
	}
	encoded, err := engine.EncodeRequest(format, request, envelope)
	if err != nil {
		s.writePluginPreviewError(w, err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, pluginRequestPreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: "preview"}, Changed: changed, Format: format, RequestBody: encoded.Body, Parameters: params, Diagnostics: append(diagnostics, encoded.Diagnostics...)})
}

func (s *ClassificationAPIServer) handleHeaderMutationPreview(w http.ResponseWriter, r *http.Request) {
	var request headerMutationPreviewRequest
	if err := s.parseStrictJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	policy, err := previewPluginConfig(s, config.DecisionPluginHeaderMutation, request.Configuration, request.Binding)
	if err != nil {
		s.writePluginPreviewError(w, err)
		return
	}
	mutations := pluginruntime.HeaderMutations(policy)
	redacted := false
	if !s.canViewSecrets(r) {
		for i := range mutations {
			if mutations[i].Operation != "remove" {
				mutations[i].Value = redactedConfigValue
				redacted = true
			}
		}
	}
	s.writeJSONResponse(w, http.StatusOK, headerMutationPreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: "preview"}, Mutations: mutations, ValuesRedacted: redacted})
}

func (s *ClassificationAPIServer) handleFastResponsePreview(w http.ResponseWriter, r *http.Request) {
	var request fastResponsePreviewRequest
	if err := s.parseStrictJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	policy, err := previewPluginConfig(s, config.DecisionPluginFastResponse, request.Configuration, request.Binding)
	if err != nil {
		s.writePluginPreviewError(w, err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, fastResponsePreviewResponse{Guarantees: pluginruntime.Guarantees{Mode: "preview"}, Message: policy.Message})
}
