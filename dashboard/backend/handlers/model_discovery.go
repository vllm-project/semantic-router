package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const (
	modelDiscoveryPath             = "/api/models/discover"
	maxModelDiscoveryRequestBytes  = 16 << 10
	maxModelDiscoveryResponseBytes = 2 << 20
	modelDiscoveryTimeout          = 8 * time.Second
)

type ModelDiscoveryRequest struct {
	BaseURL      string            `json:"baseUrl"`
	APIKey       string            `json:"apiKey,omitempty"`
	AuthHeader   string            `json:"authHeader,omitempty"`
	AuthPrefix   string            `json:"authPrefix,omitempty"`
	ExtraHeaders map[string]string `json:"extraHeaders,omitempty"`
}

type DiscoveredModel struct {
	ID      string `json:"id"`
	OwnedBy string `json:"ownedBy,omitempty"`
}

type ModelDiscoveryResponse struct {
	Models []DiscoveredModel `json:"models"`
}

type modelDiscoveryClient interface {
	Do(*http.Request) (*http.Response, error)
}

// ModelDiscoveryHandler lists models from an administrator-provided,
// OpenAI-compatible provider connection. It never stores or returns credentials.
func ModelDiscoveryHandler() http.HandlerFunc {
	client := &http.Client{
		Timeout: modelDiscoveryTimeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	return newModelDiscoveryHandler(client)
}

func newModelDiscoveryHandler(client modelDiscoveryClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if r.URL.Path != modelDiscoveryPath {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeModelDiscoveryError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is supported.")
			return
		}

		request, err := decodeModelDiscoveryRequest(w, r)
		if err != nil {
			writeModelDiscoveryError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), modelDiscoveryTimeout)
		defer cancel()
		models, err := discoverProviderModels(ctx, client, request)
		if err != nil {
			var discoveryErr *modelDiscoveryError
			if !errors.As(err, &discoveryErr) {
				discoveryErr = &modelDiscoveryError{status: http.StatusBadGateway, code: "discovery_failed", message: "Models could not be loaded from this connection."}
			}
			writeModelDiscoveryError(w, discoveryErr.status, discoveryErr.code, discoveryErr.message)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ModelDiscoveryResponse{Models: models})
	}
}

func decodeModelDiscoveryRequest(w http.ResponseWriter, r *http.Request) (ModelDiscoveryRequest, error) {
	var request ModelDiscoveryRequest
	r.Body = http.MaxBytesReader(w, r.Body, maxModelDiscoveryRequestBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		return request, errors.New("enter a valid connection")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return request, errors.New("enter a valid connection")
	}

	request.BaseURL = strings.TrimSpace(request.BaseURL)
	request.APIKey = strings.TrimSpace(request.APIKey)
	request.AuthHeader = strings.TrimSpace(request.AuthHeader)
	request.AuthPrefix = strings.TrimSpace(request.AuthPrefix)
	if len(request.ExtraHeaders) > 16 {
		return request, errors.New("use no more than 16 extra headers")
	}
	for key, value := range request.ExtraHeaders {
		trimmedKey := strings.TrimSpace(key)
		trimmedValue := strings.TrimSpace(value)
		if trimmedKey == "" || len(trimmedKey) > 128 || len(trimmedValue) > 4096 || !modelVerificationHeaderAllowed(trimmedKey) {
			return request, errors.New("an extra header is not allowed")
		}
		delete(request.ExtraHeaders, key)
		request.ExtraHeaders[trimmedKey] = trimmedValue
	}
	if request.BaseURL == "" {
		return request, errors.New("base URL is required")
	}
	if request.AuthHeader == "" {
		request.AuthHeader = "Authorization"
	}
	if request.AuthPrefix == "" && strings.EqualFold(request.AuthHeader, "Authorization") {
		request.AuthPrefix = "Bearer"
	}
	if !modelVerificationHeaderAllowed(request.AuthHeader) {
		return request, errors.New("auth header is not allowed")
	}
	return request, nil
}

func discoverProviderModels(ctx context.Context, client modelDiscoveryClient, input ModelDiscoveryRequest) ([]DiscoveredModel, error) {
	candidates, err := modelDiscoveryURLs(input.BaseURL)
	if err != nil {
		return nil, &modelDiscoveryError{status: http.StatusBadRequest, code: "invalid_url", message: "Enter a valid HTTP or HTTPS base URL."}
	}
	for index, candidate := range candidates {
		models, status, callErr := callModelDiscovery(ctx, client, candidate, input)
		if callErr == nil {
			return models, nil
		}
		if status != http.StatusNotFound || index == len(candidates)-1 {
			return nil, callErr
		}
	}
	return nil, &modelDiscoveryError{status: http.StatusBadGateway, code: "provider_unreachable", message: "This connection did not return a model list."}
}

func modelDiscoveryURLs(raw string) ([]string, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errors.New("invalid base URL")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	if strings.HasSuffix(parsed.Path, "/models") {
		return []string{parsed.String()}, nil
	}
	basePath := parsed.Path
	parsed.Path = basePath + "/models"
	candidates := []string{parsed.String()}
	if basePath == "" {
		parsed.Path = "/v1/models"
		candidates = append(candidates, parsed.String())
	}
	return candidates, nil
}

func callModelDiscovery(ctx context.Context, client modelDiscoveryClient, endpoint string, input ModelDiscoveryRequest) ([]DiscoveredModel, int, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, 0, &modelDiscoveryError{status: http.StatusBadRequest, code: "invalid_url", message: "Enter a valid HTTP or HTTPS base URL."}
	}
	request.Header.Set("Accept", "application/json")
	for key, value := range input.ExtraHeaders {
		request.Header.Set(key, value)
	}
	if input.APIKey != "" {
		value := input.APIKey
		if input.AuthPrefix != "" {
			value = input.AuthPrefix + " " + value
		}
		request.Header.Set(input.AuthHeader, value)
	}
	response, err := client.Do(request)
	if err != nil {
		message := "This connection could not be reached."
		code := "provider_unreachable"
		status := http.StatusBadGateway
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			message, code, status = "This connection took too long to respond.", "discovery_timeout", http.StatusGatewayTimeout
		}
		return nil, 0, &modelDiscoveryError{status: status, code: code, message: message}
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, response.StatusCode, &modelDiscoveryError{status: http.StatusBadGateway, code: "provider_rejected", message: fmt.Sprintf("This connection returned HTTP %d.", response.StatusCode)}
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxModelDiscoveryResponseBytes+1))
	if err != nil || len(body) > maxModelDiscoveryResponseBytes {
		return nil, response.StatusCode, &modelDiscoveryError{status: http.StatusBadGateway, code: "invalid_response", message: "This connection returned an invalid model list."}
	}
	models, err := decodeDiscoveredModels(body)
	if err != nil {
		return nil, response.StatusCode, &modelDiscoveryError{status: http.StatusBadGateway, code: "invalid_response", message: "This connection returned an invalid model list."}
	}
	return models, response.StatusCode, nil
}

func decodeDiscoveredModels(body []byte) ([]DiscoveredModel, error) {
	var payload struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || payload.Data == nil {
		return nil, errors.New("invalid model list")
	}
	byID := make(map[string]DiscoveredModel, len(payload.Data))
	for _, model := range payload.Data {
		id := strings.TrimSpace(model.ID)
		if id == "" || len(id) > 512 {
			continue
		}
		byID[id] = DiscoveredModel{ID: id, OwnedBy: strings.TrimSpace(model.OwnedBy)}
	}
	models := make([]DiscoveredModel, 0, len(byID))
	for _, model := range byID {
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

type modelDiscoveryError struct {
	status  int
	code    string
	message string
}

func (err *modelDiscoveryError) Error() string { return err.message }

func writeModelDiscoveryError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": code, "message": message})
}
