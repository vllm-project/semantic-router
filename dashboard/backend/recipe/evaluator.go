package recipe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

// HTTPRouterEvaluator is the production seam for strict validation. It only
// calls the configured Router's real Eval API; it has no simulation fallback.
type HTTPRouterEvaluator struct {
	BaseURL            string
	Client             *http.Client
	CredentialProvider routerauth.CredentialProvider
}

type routerModelsResponse struct {
	Data []struct {
		ID      string `json:"id"`
		Routing struct {
			Resolution string `json:"resolution"`
			Selectable bool   `json:"selectable"`
			Recipe     string `json:"recipe"`
		} `json:"routing"`
	} `json:"data"`
}

func (e *HTTPRouterEvaluator) ResolveRequestModel(ctx context.Context, recipeName string) (string, error) {
	if strings.TrimSpace(e.BaseURL) == "" {
		return "", errors.New("router API URL is not configured")
	}
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		strings.TrimRight(e.BaseURL, "/")+"/v1/models",
		nil,
	)
	if err != nil {
		return "", errors.New("router models URL is invalid")
	}
	if authErr := routerauth.RewriteAuthorization(request, e.CredentialProvider); authErr != nil {
		return "", errors.New("router management credential is unavailable")
	}
	response, err := e.Client.Do(request)
	if err != nil {
		return "", errors.New("router models are unavailable")
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
		return "", fmt.Errorf("router models returned status %d", response.StatusCode)
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxEvalResponseBytes+1))
	if err != nil {
		return "", errors.New("read router models response")
	}
	if len(payload) > maxEvalResponseBytes {
		return "", fmt.Errorf("router models response exceeds %d byte limit", maxEvalResponseBytes)
	}
	var catalog routerModelsResponse
	if err := json.Unmarshal(payload, &catalog); err != nil {
		return "", errors.New("router models response is invalid")
	}
	recipeName = strings.TrimSpace(recipeName)
	for _, model := range catalog.Data {
		if model.Routing.Resolution == "virtual" && model.Routing.Selectable &&
			strings.TrimSpace(model.Routing.Recipe) == recipeName && strings.TrimSpace(model.ID) != "" {
			return strings.TrimSpace(model.ID), nil
		}
	}
	return "", fmt.Errorf("active Router does not advertise a model for Recipe %q", recipeName)
}

func (e *HTTPRouterEvaluator) Evaluate(ctx context.Context, request EvalRequest) (json.RawMessage, error) {
	if strings.TrimSpace(e.BaseURL) == "" {
		return nil, errors.New("router API URL is not configured")
	}
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	endpoint := strings.TrimRight(e.BaseURL, "/") + "/api/v1/eval?trace=true"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, errors.New("router eval URL is invalid")
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	if authErr := routerauth.RewriteAuthorization(httpRequest, e.CredentialProvider); authErr != nil {
		return nil, errors.New("router management credential is unavailable")
	}
	response, err := e.Client.Do(httpRequest)
	if err != nil {
		return nil, errors.New("router eval is unavailable")
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
		return nil, fmt.Errorf("router eval returned status %d", response.StatusCode)
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxEvalResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read router eval response: %w", err)
	}
	if len(payload) > maxEvalResponseBytes {
		return nil, fmt.Errorf("router eval response exceeds %d byte limit", maxEvalResponseBytes)
	}
	return payload, nil
}
