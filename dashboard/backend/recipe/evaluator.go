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
