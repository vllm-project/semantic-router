package modelruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Transport carries one HTTP exchange to a runtime.
type Transport interface {
	Do(ctx context.Context, method, path string, body []byte) (status int, payload []byte, err error)
}

// Client calls the runtime surfaces over a Transport.
type Client struct {
	transport Transport
}

// NewClient wraps a transport.
func NewClient(transport Transport) *Client {
	return &Client{transport: transport}
}

// NewHTTPClient reaches a runtime at baseURL, for example a port-forward.
func NewHTTPClient(baseURL string, timeout time.Duration) *Client {
	return NewClient(httpTransport{baseURL: strings.TrimRight(baseURL, "/"), client: &http.Client{Timeout: timeout}})
}

type httpTransport struct {
	baseURL string
	client  *http.Client
}

func (t httpTransport) Do(ctx context.Context, method, path string, body []byte) (int, []byte, error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, method, t.baseURL+path, reader)
	if err != nil {
		return 0, nil, err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := t.client.Do(request)
	if err != nil {
		return 0, nil, err
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(response.Body)
	return response.StatusCode, payload, err
}

// StatusError is a response with an unexpected HTTP status.
type StatusError struct {
	Path   string
	Status int
	Body   string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("%s returned %d: %s", e.Path, e.Status, e.Body)
}

// Health returns the runtime's health and its HTTP status (200 only when every
// model is ready).
func (c *Client) Health(ctx context.Context) (Health, int, error) {
	var health Health
	status, payload, err := c.transport.Do(ctx, http.MethodGet, "/health", nil)
	if err != nil {
		return health, status, err
	}
	if err := json.Unmarshal(payload, &health); err != nil {
		return health, status, fmt.Errorf("/health: %w", err)
	}
	return health, status, nil
}

// Liveness returns the runtime's liveness, which it answers while the process
// serves HTTP, whatever its models' readiness.
func (c *Client) Liveness(ctx context.Context) (Liveness, error) {
	var live Liveness
	return live, c.get(ctx, "/health/live", &live)
}

// Models lists the served models.
func (c *Client) Models(ctx context.Context) (ModelList, error) {
	var models ModelList
	return models, c.get(ctx, "/v1/models", &models)
}

// Model returns the card of one served model.
func (c *Client) Model(ctx context.Context, id string) (ModelCard, error) {
	models, err := c.Models(ctx)
	if err != nil {
		return ModelCard{}, err
	}
	for _, card := range models.Data {
		if card.ID == id {
			return card, nil
		}
	}
	ids := make([]string, 0, len(models.Data))
	for _, card := range models.Data {
		ids = append(ids, card.ID)
	}
	return ModelCard{}, fmt.Errorf("model %q is not served; the runtime serves %v", id, ids)
}

// Classify calls POST /v1/classify.
func (c *Client) Classify(ctx context.Context, request ClassifyRequest) (ClassifyResponse, error) {
	var response ClassifyResponse
	return response, c.post(ctx, "/v1/classify", request, &response)
}

// Embed calls POST /v1/embeddings.
func (c *Client) Embed(ctx context.Context, request EmbeddingsRequest) (EmbeddingsResponse, error) {
	var response EmbeddingsResponse
	return response, c.post(ctx, "/v1/embeddings", request, &response)
}

// Rerank calls POST /v1/rerank.
func (c *Client) Rerank(ctx context.Context, request RerankRequest) (RerankResponse, error) {
	var response RerankResponse
	return response, c.post(ctx, "/v1/rerank", request, &response)
}

// Decide calls POST /v1/decisions and requires an answer to every question.
func (c *Client) Decide(ctx context.Context, request DecisionsRequest) (DecisionsResponse, error) {
	var response DecisionsResponse
	if err := c.post(ctx, "/v1/decisions", request, &response); err != nil {
		return response, err
	}
	for id, question := range request.Questions {
		if answer := response.Answers[id]; answer.Error != "" || !response.answers(id, question.Type) {
			return response, fmt.Errorf("question %q has no answer: %+v", id, answer)
		}
	}
	return response, nil
}

// Bundle calls POST /v1/bundle.
func (c *Client) Bundle(ctx context.Context, request BundleRequest) (BundleResponse, error) {
	var response BundleResponse
	return response, c.post(ctx, "/v1/bundle", request, &response)
}

// Metrics scrapes GET /metrics.
func (c *Client) Metrics(ctx context.Context) (Metrics, error) {
	status, payload, err := c.transport.Do(ctx, http.MethodGet, "/metrics", nil)
	if err != nil {
		return Metrics{}, err
	}
	if status != http.StatusOK {
		return Metrics{}, &StatusError{Path: "/metrics", Status: status, Body: string(payload)}
	}
	return ParseMetrics(string(payload))
}

// WaitReady polls /health until every model is ready.
func (c *Client) WaitReady(ctx context.Context, timeout time.Duration) error {
	return Eventually(ctx, timeout, func(ctx context.Context) error {
		health, status, err := c.Health(ctx)
		if err != nil {
			return err
		}
		if status != http.StatusOK {
			return fmt.Errorf("runtime health %d: %+v", status, health)
		}
		return nil
	})
}

func (c *Client) get(ctx context.Context, path string, into interface{}) error {
	return c.exchange(ctx, http.MethodGet, path, nil, into)
}

func (c *Client) post(ctx context.Context, path string, request, into interface{}) error {
	body, err := json.Marshal(request)
	if err != nil {
		return err
	}
	return c.exchange(ctx, http.MethodPost, path, body, into)
}

func (c *Client) exchange(ctx context.Context, method, path string, body []byte, into interface{}) error {
	status, payload, err := c.transport.Do(ctx, method, path, body)
	if err != nil {
		return fmt.Errorf("%s %s: %w", method, path, err)
	}
	if status != http.StatusOK {
		return &StatusError{Path: path, Status: status, Body: string(payload)}
	}
	if err := json.Unmarshal(payload, into); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
