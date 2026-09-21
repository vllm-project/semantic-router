package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

const (
	modelDiscoveryPath            = "/api/models/discover"
	maxModelDiscoveryRequestBytes = 16 << 10
	maxModelDiscoveryResponseSize = 4 << 20
)

type ModelDiscoveryRequest struct {
	BaseURL  string `json:"baseUrl"`
	APIKey   string `json:"apiKey"`
	Provider string `json:"provider"`
}

type ModelDiscoveryResponse struct {
	Models []string `json:"models"`
}

// ModelDiscoveryHandler is a Dashboard authoring helper. It queries a provider's
// read-only model inventory and returns identifiers that can be compiled into
// providers.models. Neither credentials nor provider state are retained here.
func ModelDiscoveryHandler(client *http.Client) http.HandlerFunc {
	discoveryClient := secureModelDiscoveryClient(client)
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if r.URL.Path != modelDiscoveryPath {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "Only POST is supported.", http.StatusMethodNotAllowed)
			return
		}

		var input ModelDiscoveryRequest
		decoder := json.NewDecoder(io.LimitReader(r.Body, maxModelDiscoveryRequestBytes))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			writeModelDiscoveryError(w, http.StatusBadRequest, "Check the connection details.")
			return
		}
		registry, err := modelcatalog.BuiltIn()
		if err != nil {
			writeModelDiscoveryError(w, http.StatusInternalServerError, "The built-in provider catalog is unavailable.")
			return
		}
		provider, ok := registry.Provider(strings.TrimSpace(input.Provider))
		if !ok {
			writeModelDiscoveryError(w, http.StatusBadRequest, "Choose a supported provider.")
			return
		}
		target, err := modelInventoryTarget(input.BaseURL, registry, provider)
		if err != nil {
			writeModelDiscoveryError(w, http.StatusBadRequest, err.Error())
			return
		}

		requestContext := withModelDiscoveryNetworkPolicy(r.Context(), target.policy)
		request, err := http.NewRequestWithContext(requestContext, http.MethodGet, target.url, nil)
		if err != nil {
			writeModelDiscoveryError(w, http.StatusBadRequest, "The provider URL is invalid.")
			return
		}
		request.Header.Set("Accept", "application/json")
		applyModelDiscoveryHeaders(request, provider, strings.TrimSpace(input.APIKey))

		response, err := discoveryClient.Do(request)
		if err != nil {
			writeModelDiscoveryError(w, http.StatusBadGateway, "The provider could not be reached.")
			return
		}
		defer response.Body.Close()
		body, err := io.ReadAll(io.LimitReader(response.Body, maxModelDiscoveryResponseSize+1))
		if err != nil || len(body) > maxModelDiscoveryResponseSize {
			writeModelDiscoveryError(w, http.StatusBadGateway, "The provider returned an unreadable model list.")
			return
		}
		if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
			writeModelDiscoveryError(w, http.StatusBadGateway, fmt.Sprintf("The provider rejected the connection (HTTP %d).", response.StatusCode))
			return
		}

		models, err := decodeProviderModelIDs(body)
		if err != nil {
			writeModelDiscoveryError(w, http.StatusBadGateway, err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ModelDiscoveryResponse{Models: models})
	}
}

func modelInventoryTarget(raw string, registry *modelcatalog.Registry, provider modelcatalog.ProviderDefinition) (modelDiscoveryTarget, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return modelDiscoveryTarget{}, errors.New("enter a complete HTTP or HTTPS base URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return modelDiscoveryTarget{}, errors.New("the base URL cannot contain credentials, query parameters, or a fragment")
	}
	operationPath, err := registry.ResolveOperationPath(provider.ID, provider.DefaultProtocol, "list_models", parsed.Path)
	if err != nil {
		return modelDiscoveryTarget{}, errors.New("this provider does not declare model discovery support")
	}
	policy, err := modelDiscoveryNetworkPolicyForProvider(parsed, provider)
	if err != nil {
		return modelDiscoveryTarget{}, err
	}
	parsed.Path = path.Clean(operationPath)
	parsed.RawPath = ""
	return modelDiscoveryTarget{url: parsed.String(), policy: policy}, nil
}

func applyModelDiscoveryHeaders(request *http.Request, provider modelcatalog.ProviderDefinition, apiKey string) {
	for header, value := range provider.DefaultHeaders {
		request.Header.Set(header, value)
	}
	if apiKey == "" || provider.Auth.Strategy == "none" {
		return
	}
	value := apiKey
	if prefix := strings.TrimSpace(provider.Auth.Prefix); prefix != "" {
		value = prefix + " " + apiKey
	}
	request.Header.Set(provider.Auth.Header, value)
}

func decodeProviderModelIDs(body []byte) ([]string, error) {
	type modelListItem struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	var payload struct {
		Data   []modelListItem `json:"data"`
		Models []modelListItem `json:"models"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, errors.New("the provider returned an invalid model list")
	}
	items := payload.Data
	if len(items) == 0 {
		items = payload.Models
	}
	unique := make(map[string]struct{}, len(items))
	for _, item := range items {
		value := strings.TrimSpace(item.ID)
		if value == "" {
			value = strings.TrimSpace(item.Name)
		}
		if value != "" {
			unique[strings.TrimPrefix(value, "models/")] = struct{}{}
		}
	}
	if len(unique) == 0 {
		return nil, errors.New("no chat models were returned by this provider")
	}
	models := make([]string, 0, len(unique))
	for model := range unique {
		models = append(models, model)
	}
	sort.Strings(models)
	return models, nil
}

func writeModelDiscoveryError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}
