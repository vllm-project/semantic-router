package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const remoteEmbeddingExpectedDimension = 4

func init() {
	pkgtestcases.Register("remote-embedding-routing", pkgtestcases.TestCase{
		Description: "Verify remote embedding startup, reordered batch vectors, and deterministic embedding-signal routing",
		Tags:        []string{"embedding", "remote-provider", "routing", "openai-compatible"},
		Fn:          testRemoteEmbeddingRouting,
	})
}

type remoteEmbeddingStartupStatus struct {
	Ready             bool                           `json:"ready"`
	EmbeddingProvider *remoteEmbeddingProviderStatus `json:"embedding_provider"`
}

type remoteEmbeddingProviderStatus struct {
	Mode           string `json:"mode"`
	Backend        string `json:"backend"`
	Model          string `json:"model"`
	Dimension      int    `json:"dimension"`
	APIKeyEnv      string `json:"api_key_env"`
	APIKeyEnvSet   *bool  `json:"api_key_env_set"`
	Healthy        *bool  `json:"healthy"`
	LastProbeError string `json:"last_probe_error"`
	LastCheckedAt  string `json:"last_checked_at"`
}

type remoteEmbeddingRouteCase struct {
	Name             string
	Prompt           string
	ExpectedDecision string
}

func testRemoteEmbeddingRouting(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	providerStatus, err := fetchRemoteEmbeddingStartupStatus(ctx, client, opts)
	if err != nil {
		return err
	}

	gatewaySession, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer gatewaySession.Close()
	if err := checkRemoteEmbeddingBatch(ctx, client, opts, gatewaySession); err != nil {
		return fmt.Errorf("reordered remote embedding batch: %w", err)
	}

	cases := []remoteEmbeddingRouteCase{
		{
			Name:             "billing prompt matches remote embedding signal",
			Prompt:           "Please explain the invoice charge on my subscription and issue a refund.",
			ExpectedDecision: "billing-route",
		},
		{
			Name:             "unrelated prompt uses fallback decision",
			Prompt:           "Explain how photosynthesis works in a short paragraph.",
			ExpectedDecision: "default-route",
		},
	}

	results := make(map[string]string, len(cases))
	for _, testCase := range cases {
		decision, err := requestRemoteEmbeddingDecision(ctx, gatewaySession, testCase.Prompt)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.Name, err)
		}
		results[testCase.Name] = decision
		if decision != testCase.ExpectedDecision {
			return fmt.Errorf(
				"%s: expected x-vsr-selected-decision=%q, got %q",
				testCase.Name,
				testCase.ExpectedDecision,
				decision,
			)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"provider_backend":      providerStatus.Backend,
			"provider_model":        providerStatus.Model,
			"provider_healthy":      true,
			"dimension":             providerStatus.Dimension,
			"batch_vectors_checked": 2,
			"routing_cases":         len(cases),
			"routing_passed":        len(cases),
			"decisions":             results,
		})
	}

	return nil
}

func checkRemoteEmbeddingBatch(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	gateway *fixtures.ServiceSession,
) error {
	marker := fmt.Sprintf("batch-%d", time.Now().UnixNano())
	texts := []string{
		marker + " " + strings.Repeat("Billing invoice payment refund. ", 20),
		marker + " " + strings.Repeat("Photosynthesis converts sunlight into energy. ", 20),
	}
	// The diagnostics API embeds texts individually. Compression first fills the
	// runtime vector cache with a real batch containing both distinct history texts.
	response, err := sendProtocolMatrixRaw(ctx, gateway, "/v1/chat/completions", map[string]interface{}{
		"model":      "vllm-sr/auto",
		"max_tokens": 16,
		"messages": []map[string]string{
			{"role": "user", "content": texts[0]},
			{"role": "assistant", "content": texts[1]},
			{"role": "user", "content": "Continue."},
			{"role": "assistant", "content": "Acknowledged."},
			{"role": "user", "content": "Please explain my billing invoice."},
		},
	}, false, nil)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK || response.Headers.Get("x-vsr-selected-decision") != "billing-route" {
		return fmt.Errorf("expected billing-route status 200, got %d, decision %q: %s", response.StatusCode, response.Headers.Get("x-vsr-selected-decision"), response.Body)
	}
	mock, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "mock-embedding", "8000")
	if err != nil {
		return err
	}
	defer mock.Close()
	batchResponse, err := getJSON(ctx, mock.HTTPClient(30*time.Second), mock.URL("/last-batch"))
	if err != nil {
		return err
	}
	if batchResponse.StatusCode != http.StatusOK {
		return fmt.Errorf("expected /last-batch status 200, got %d: %s", batchResponse.StatusCode, batchResponse.Body)
	}
	var batch struct {
		Inputs []string `json:"inputs"`
	}
	if err := json.Unmarshal(batchResponse.Body, &batch); err != nil {
		return err
	}
	if !slices.Contains(batch.Inputs, texts[0]) || !slices.Contains(batch.Inputs, texts[1]) {
		return fmt.Errorf("expected both history texts in one provider batch, got %v", batch.Inputs)
	}
	api, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer api.Close()
	payload, err := json.Marshal(map[string]interface{}{"recipe": "default", "texts": texts})
	if err != nil {
		return err
	}
	vectorsResponse, err := postJSON(ctx, api.HTTPClient(30*time.Second), http.MethodPost, api.URL("/api/v1/diagnostics/embeddings"), payload)
	if err != nil {
		return err
	}
	if vectorsResponse.StatusCode != http.StatusOK {
		return fmt.Errorf("expected embeddings status 200, got %d: %s", vectorsResponse.StatusCode, vectorsResponse.Body)
	}
	var vectors struct {
		Embeddings []struct {
			Text      string    `json:"text"`
			Embedding []float32 `json:"embedding"`
		} `json:"embeddings"`
	}
	if err := json.Unmarshal(vectorsResponse.Body, &vectors); err != nil {
		return err
	}
	want := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	if len(vectors.Embeddings) != len(texts) {
		return fmt.Errorf("expected %d embeddings, got %d", len(texts), len(vectors.Embeddings))
	}
	for i, result := range vectors.Embeddings {
		if result.Text != texts[i] || !slices.Equal(result.Embedding, want[i]) {
			return fmt.Errorf("batch input %d: got text %q, vector %v; want text %q, vector %v", i, result.Text, result.Embedding, texts[i], want[i])
		}
	}
	return nil
}

func fetchRemoteEmbeddingStartupStatus(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*remoteEmbeddingProviderStatus, error) {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	response, err := getJSON(ctx, session.HTTPClient(30*time.Second), session.URL("/startup-status"))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected /startup-status status 200, got %d: %s", response.StatusCode, response.Body)
	}

	var status remoteEmbeddingStartupStatus
	if err := json.Unmarshal(response.Body, &status); err != nil {
		return nil, fmt.Errorf("decode /startup-status response: %w", err)
	}
	if !status.Ready || status.EmbeddingProvider == nil {
		return nil, fmt.Errorf("expected ready router with embedding provider status, got %+v", status)
	}
	if err := validateRemoteEmbeddingProviderStatus(status.EmbeddingProvider); err != nil {
		return nil, err
	}
	return status.EmbeddingProvider, nil
}

func validateRemoteEmbeddingProviderStatus(provider *remoteEmbeddingProviderStatus) error {
	if provider.Mode != "remote" || provider.Backend != "openai_compatible" {
		return fmt.Errorf("expected remote openai_compatible provider, got %+v", provider)
	}
	if provider.Dimension != remoteEmbeddingExpectedDimension {
		return fmt.Errorf("expected provider dimension %d, got %d", remoteEmbeddingExpectedDimension, provider.Dimension)
	}
	if provider.Model != "e2e-remote-embedding" {
		return fmt.Errorf("expected provider model e2e-remote-embedding, got %q", provider.Model)
	}
	if provider.APIKeyEnv != "REMOTE_EMBEDDING_E2E_KEY" || provider.APIKeyEnvSet == nil || !*provider.APIKeyEnvSet {
		return fmt.Errorf("expected configured provider credential env, got %+v", provider)
	}
	if provider.Healthy == nil || !*provider.Healthy {
		return fmt.Errorf("expected healthy provider, got error %q", provider.LastProbeError)
	}
	if provider.LastCheckedAt == "" {
		return fmt.Errorf("expected provider last_checked_at timestamp")
	}
	return nil
}

func requestRemoteEmbeddingDecision(
	ctx context.Context,
	session *fixtures.ServiceSession,
	prompt string,
) (string, error) {
	payload, err := json.Marshal(map[string]interface{}{
		"model": "vllm-sr/auto",
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	})
	if err != nil {
		return "", fmt.Errorf("marshal chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		session.URL("/v1/chat/completions"),
		bytes.NewReader(payload),
	)
	if err != nil {
		return "", fmt.Errorf("create chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := session.HTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return "", fmt.Errorf("send chat request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read chat response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("expected chat status 200, got %d: %s", resp.StatusCode, body)
	}

	decision := resp.Header.Get("x-vsr-selected-decision")
	if decision == "" {
		return "", fmt.Errorf("chat response omitted x-vsr-selected-decision: %s", body)
	}
	return decision, nil
}
