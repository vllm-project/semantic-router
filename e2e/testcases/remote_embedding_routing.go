package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const remoteEmbeddingExpectedDimension = 4

func init() {
	pkgtestcases.Register("remote-embedding-routing", pkgtestcases.TestCase{
		Description: "Verify remote embedding provider startup and deterministic embedding-signal routing",
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
			"provider_backend": providerStatus.Backend,
			"provider_model":   providerStatus.Model,
			"provider_healthy": true,
			"dimension":        providerStatus.Dimension,
			"routing_cases":    len(cases),
			"routing_passed":   len(cases),
			"decisions":        results,
		})
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
		"model": "auto",
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
