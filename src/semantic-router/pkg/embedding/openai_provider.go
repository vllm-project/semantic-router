package embedding

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	maxErrorBodyDrainBytes             = 64 << 10
	maxEmbeddingResponseBytes          = 32 << 20
	embeddingResponseBaseBytes         = 16 << 10
	embeddingResponsePerVectorBytes    = 256
	embeddingResponsePerValueBytes     = 32
	unknownEmbeddingDimensionForBudget = 8192
	defaultTimeoutSeconds              = 10
	baseRetryDelay                     = 50 * time.Millisecond
	maxRetryDelay                      = 500 * time.Millisecond
)

type embeddingResponseErrorKind string

const (
	embeddingResponseTooLarge     embeddingResponseErrorKind = "too_large"
	embeddingResponseReadFailure  embeddingResponseErrorKind = "read_failure"
	embeddingResponseInvalidJSON  embeddingResponseErrorKind = "invalid_json"
	embeddingResponseTrailingData embeddingResponseErrorKind = "trailing_data"
	embeddingResponseInvalidData  embeddingResponseErrorKind = "invalid_data"
)

type embeddingResponseError struct {
	kind    embeddingResponseErrorKind
	message string
	cause   error
}

func (e *embeddingResponseError) Error() string {
	return e.message
}

func (e *embeddingResponseError) Unwrap() error {
	return e.cause
}

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
	Index     *int      `json:"index,omitempty"`
	Embedding []float64 `json:"embedding"`
}

type providerError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
}

func NewOpenAICompatibleProvider(cfg OpenAICompatibleConfig) (*OpenAICompatibleProvider, error) {
	endpoint, model, err := validateOpenAICompatibleConfig(cfg)
	if err != nil {
		return nil, err
	}

	timeoutSeconds := cfg.TimeoutSeconds
	if timeoutSeconds == 0 {
		timeoutSeconds = defaultTimeoutSeconds
	}
	timeout := time.Duration(timeoutSeconds) * time.Second
	client := &http.Client{Transport: newNoProxyEmbeddingTransport()}
	if cfg.HTTPClient != nil {
		*client = *cfg.HTTPClient
		if client.Transport == nil {
			client.Transport = newNoProxyEmbeddingTransport()
		}
	}
	if client.Timeout == 0 && timeout > 0 {
		client.Timeout = timeout
	}
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
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
	delay := retryDelay(attempt)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func retryDelay(attempt int) time.Duration {
	delay := baseRetryDelay
	if attempt <= 1 {
		return delay
	}
	// The loop is bounded by the fixed base/max ratio, not by attempt. This
	// keeps even an extreme attempt value cheap and avoids overflowing shifts.
	for remaining := attempt - 1; remaining > 0 && delay < maxRetryDelay; remaining-- {
		if delay > maxRetryDelay/2 {
			return maxRetryDelay
		}
		delay *= 2
	}
	return min(delay, maxRetryDelay)
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
	req, cancel, err := p.newEmbeddingBatchRequest(ctx, texts, apiKey)
	if err != nil {
		return nil, err
	}
	defer cancel()

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, redactEmbeddingTransportError(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, responseError(resp)
	}

	decoded, err := decodeEmbeddingResponse(resp, p.responseByteLimit(len(texts)))
	if err != nil {
		return nil, err
	}
	if decoded.Error != nil && decoded.Error.Message != "" {
		return nil, newEmbeddingResponseError(
			embeddingResponseInvalidData,
			"embedding provider returned an error response",
			nil,
		)
	}
	embeddings, err := p.parseEmbeddings(decoded.Data, len(texts))
	if err != nil {
		return nil, newEmbeddingResponseError(embeddingResponseInvalidData, err.Error(), err)
	}
	return embeddings, nil
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
	index := sequentialIndex
	if item.Index != nil {
		index = *item.Index
	}
	if index < 0 || index >= expectedCount {
		return 0, fmt.Errorf("embedding provider returned invalid embedding index %d", index)
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
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("embedding value at position %d is not finite", i)
		}
		if math.Abs(value) > math.MaxFloat32 {
			return nil, fmt.Errorf("embedding value at position %d is outside the float32 range", i)
		}
		converted := float32(value)
		if math.IsNaN(float64(converted)) || math.IsInf(float64(converted), 0) {
			return nil, fmt.Errorf("embedding value at position %d is not finite after float32 conversion", i)
		}
		embedding[i] = converted
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
