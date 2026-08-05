package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	maxErrorBodyBytes     = 4096
	defaultTimeoutSeconds = 10
	baseRetryDelay        = 50 * time.Millisecond
	maxRetryDelay         = 500 * time.Millisecond
)

type OpenAICompatibleConfig struct {
	BaseURL           string
	Model             string
	APIKeyEnv         string
	TimeoutSeconds    int
	MaxRetries        int
	Dimensions        int
	ExpectedDimension int
	HTTPClient        *http.Client
}

type OpenAICompatibleProvider struct {
	endpoint          string
	model             string
	apiKeyEnv         string
	timeout           time.Duration
	maxRetries        int
	dimensions        int
	expectedDimension int
	client            *http.Client
}

type embeddingsRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type embeddingsResponse struct {
	Data  []embeddingDatum `json:"data"`
	Error *providerError   `json:"error,omitempty"`
}

type embeddingDatum struct {
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type providerError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
}

func NewOpenAICompatibleProvider(cfg OpenAICompatibleConfig) (*OpenAICompatibleProvider, error) {
	endpoint, err := embeddingsEndpoint(cfg.BaseURL)
	if err != nil {
		return nil, err
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		return nil, fmt.Errorf("embedding endpoint model is required for backend %q", config.EmbeddingBackendOpenAICompatible)
	}
	if cfg.ExpectedDimension > 0 && cfg.Dimensions > 0 && cfg.ExpectedDimension != cfg.Dimensions {
		return nil, fmt.Errorf("embedding endpoint dimensions (%d) must match target_dimension (%d)", cfg.Dimensions, cfg.ExpectedDimension)
	}
	if cfg.TimeoutSeconds < 0 {
		return nil, fmt.Errorf("embedding endpoint timeout_seconds must be non-negative")
	}

	timeoutSeconds := cfg.TimeoutSeconds
	if timeoutSeconds == 0 {
		timeoutSeconds = defaultTimeoutSeconds
	}
	timeout := time.Duration(timeoutSeconds) * time.Second
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: timeout}
	} else if client.Timeout == 0 && timeout > 0 {
		copyClient := *client
		copyClient.Timeout = timeout
		client = &copyClient
	}

	return &OpenAICompatibleProvider{
		endpoint:          endpoint,
		model:             model,
		apiKeyEnv:         strings.TrimSpace(cfg.APIKeyEnv),
		timeout:           timeout,
		maxRetries:        max(0, cfg.MaxRetries),
		dimensions:        cfg.Dimensions,
		expectedDimension: cfg.ExpectedDimension,
		client:            client,
	}, nil
}

func (p *OpenAICompatibleProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := p.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

func (p *OpenAICompatibleProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	apiKey, err := p.resolveAPIKey()
	if err != nil {
		return nil, err
	}

	var lastErr error
	attempts := p.maxRetries + 1
	for attempt := 1; attempt <= attempts; attempt++ {
		embeddings, err := p.embedBatchOnce(ctx, texts, apiKey)
		if err == nil {
			return embeddings, nil
		}
		lastErr = err
		if !shouldRetryEmbeddingError(err) || attempt == attempts {
			break
		}
		if err := waitBeforeRetry(ctx, attempt); err != nil {
			lastErr = err
			break
		}
	}
	if isTimeoutError(lastErr) {
		return nil, fmt.Errorf("embedding provider request timed out after %d attempt(s): %w", attempts, lastErr)
	}
	return nil, fmt.Errorf("embedding provider request failed after %d attempt(s): %w", attempts, lastErr)
}

func waitBeforeRetry(ctx context.Context, attempt int) error {
	delay := baseRetryDelay << max(0, attempt-1)
	if delay > maxRetryDelay {
		delay = maxRetryDelay
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *OpenAICompatibleProvider) Dimension() int {
	if p.expectedDimension > 0 {
		return p.expectedDimension
	}
	return p.dimensions
}

func (p *OpenAICompatibleProvider) Backend() string {
	return config.EmbeddingBackendOpenAICompatible
}

func (p *OpenAICompatibleProvider) embedBatchOnce(ctx context.Context, texts []string, apiKey string) ([][]float32, error) {
	body, err := json.Marshal(embeddingsRequest{Model: p.model, Input: texts, Dimensions: p.dimensions})
	if err != nil {
		return nil, err
	}

	attemptCtx := ctx
	var cancel context.CancelFunc
	if p.timeout > 0 {
		attemptCtx, cancel = context.WithTimeout(ctx, p.timeout)
	}
	if cancel != nil {
		defer cancel()
	}

	req, err := http.NewRequestWithContext(attemptCtx, http.MethodPost, p.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, responseError(resp)
	}

	var decoded embeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode embedding response: %w", err)
	}
	if decoded.Error != nil && decoded.Error.Message != "" {
		return nil, fmt.Errorf("embedding provider error: %s", decoded.Error.Message)
	}
	return p.parseEmbeddings(decoded.Data, len(texts))
}

func (p *OpenAICompatibleProvider) parseEmbeddings(data []embeddingDatum, expectedCount int) ([][]float32, error) {
	if len(data) != expectedCount {
		return nil, fmt.Errorf("embedding provider returned %d embedding(s), expected %d", len(data), expectedCount)
	}

	result := make([][]float32, expectedCount)
	for sequentialIndex, item := range data {
		index, err := resolveEmbeddingIndex(item, sequentialIndex, expectedCount)
		if err != nil {
			return nil, err
		}
		if result[index] != nil {
			return nil, fmt.Errorf("embedding provider returned duplicate embedding index %d", index)
		}
		converted, err := p.convertEmbedding(item.Embedding)
		if err != nil {
			return nil, fmt.Errorf("embedding index %d: %w", index, err)
		}
		result[index] = converted
	}
	if err := ensureCompleteEmbeddings(result); err != nil {
		return nil, err
	}
	return result, nil
}

func resolveEmbeddingIndex(item embeddingDatum, sequentialIndex int, expectedCount int) (int, error) {
	index := item.Index
	if expectedCount == 1 && index == 0 {
		return 0, nil
	}
	if index == 0 && sequentialIndex > 0 {
		index = sequentialIndex
	}
	if index < 0 || index >= expectedCount {
		return 0, fmt.Errorf("embedding provider returned invalid embedding index %d", item.Index)
	}
	return index, nil
}

func ensureCompleteEmbeddings(result [][]float32) error {
	for i, embedding := range result {
		if embedding == nil {
			return fmt.Errorf("embedding provider omitted embedding index %d", i)
		}
	}
	return nil
}

func (p *OpenAICompatibleProvider) convertEmbedding(values []float64) ([]float32, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("empty embedding vector")
	}
	if p.expectedDimension > 0 && len(values) != p.expectedDimension {
		return nil, fmt.Errorf("embedding dimension mismatch: got %d, want %d", len(values), p.expectedDimension)
	}
	embedding := make([]float32, len(values))
	for i, value := range values {
		embedding[i] = float32(value)
	}
	return embedding, nil
}

func (p *OpenAICompatibleProvider) resolveAPIKey() (string, error) {
	if p.apiKeyEnv == "" {
		return "", nil
	}
	value := os.Getenv(p.apiKeyEnv)
	if value == "" {
		return "", fmt.Errorf("embedding API key env %q is not set", p.apiKeyEnv)
	}
	return value, nil
}

func embeddingsEndpoint(baseURL string) (string, error) {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		return "", fmt.Errorf("embedding endpoint base_url is required for backend %q", config.EmbeddingBackendOpenAICompatible)
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("invalid embedding endpoint base_url %q: %w", baseURL, err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("embedding endpoint base_url %q must include scheme and host", baseURL)
	}
	path := strings.TrimRight(parsed.Path, "/")
	if !strings.HasSuffix(path, "/embeddings") {
		path += "/embeddings"
	}
	parsed.Path = path
	parsed.RawQuery = ""
	return parsed.String(), nil
}

type embeddingHTTPError struct {
	statusCode int
	message    string
	retryable  bool
}

func (e *embeddingHTTPError) Error() string {
	return e.message
}

func responseError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
	bodyText := strings.TrimSpace(string(body))
	if bodyText == "" {
		bodyText = http.StatusText(resp.StatusCode)
	}
	message := fmt.Sprintf("embedding provider returned status %d: %s", resp.StatusCode, bodyText)
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		message = fmt.Sprintf("embedding provider authentication failed with status %d: %s", resp.StatusCode, bodyText)
	}
	return &embeddingHTTPError{
		statusCode: resp.StatusCode,
		message:    message,
		retryable:  resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= http.StatusInternalServerError,
	}
}

func shouldRetryEmbeddingError(err error) bool {
	var httpErr *embeddingHTTPError
	if errors.As(err, &httpErr) {
		return httpErr.retryable
	}
	return isTimeoutError(err) || isTemporaryNetworkError(err)
}

func isTimeoutError(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}

func isTemporaryNetworkError(err error) bool {
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Temporary()
}
