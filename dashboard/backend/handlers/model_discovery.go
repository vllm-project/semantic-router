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
	"time"
)

const (
	modelDiscoveryPath            = "/api/models/discover"
	maxModelDiscoveryRequestBytes = 16 << 10
	maxModelDiscoveryResponseSize = 4 << 20
)

type ModelDiscoveryRequest struct {
	BaseURL  string `json:"baseUrl"`
	APIKey   string `json:"apiKey"`
	AuthMode string `json:"authMode"`
}

type ModelDiscoveryResponse struct {
	Models []string `json:"models"`
}

// ModelDiscoveryHandler is a Dashboard authoring helper. It queries a provider's
// read-only model inventory and returns identifiers that can be compiled into
// providers.models. Neither credentials nor provider state are retained here.
func ModelDiscoveryHandler(client *http.Client) http.HandlerFunc {
	if client == nil {
		client = &http.Client{Timeout: 12 * time.Second}
	}
	discoveryClient := *client
	discoveryClient.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
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
		endpoint, err := modelInventoryURL(input.BaseURL)
		if err != nil {
			writeModelDiscoveryError(w, http.StatusBadRequest, err.Error())
			return
		}

		request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, endpoint, nil)
		if err != nil {
			writeModelDiscoveryError(w, http.StatusBadRequest, "The provider URL is invalid.")
			return
		}
		request.Header.Set("Accept", "application/json")
		applyModelDiscoveryAuth(request, strings.TrimSpace(input.AuthMode), strings.TrimSpace(input.APIKey))

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

func modelInventoryURL(raw string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", errors.New("enter a complete HTTP or HTTPS base URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("the base URL cannot contain credentials, query parameters, or a fragment")
	}
	cleanPath := strings.TrimSuffix(parsed.Path, "/")
	if !strings.HasSuffix(cleanPath, "/models") {
		parsed.Path = path.Join(cleanPath, "models")
	}
	return parsed.String(), nil
}

func applyModelDiscoveryAuth(request *http.Request, mode, apiKey string) {
	if apiKey == "" {
		return
	}
	switch mode {
	case "anthropic":
		request.Header.Set("x-api-key", apiKey)
		request.Header.Set("anthropic-version", "2023-06-01")
	default:
		request.Header.Set("Authorization", "Bearer "+apiKey)
	}
}

func decodeProviderModelIDs(body []byte) ([]string, error) {
	var payload struct {
		Data   []map[string]any `json:"data"`
		Models []map[string]any `json:"models"`
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
		for _, field := range []string{"id", "name"} {
			value, ok := item[field].(string)
			value = strings.TrimSpace(value)
			if ok && value != "" {
				unique[strings.TrimPrefix(value, "models/")] = struct{}{}
				break
			}
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
