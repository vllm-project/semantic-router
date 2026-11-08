package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const remoteEmbeddingExpectedDimension = 4

func init() {
	pkgtestcases.Register("remote-embedding-routing", pkgtestcases.TestCase{
		Description: "Verify the remote embedding provider prepared for response cache recall, billing_support routing, a paraphrase hit through remote-embedding recall, and a recalled request the same_request judge rejects as a miss",
		Tags:        []string{"embedding", "remote-provider", "routing", "openai-compatible", "response-cache"},
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

// remoteEmbeddingParaphrase restates the billing case's request. Recall
// embeds it with the remote embedding deployment and finds the stored billing
// answer; the same_request judge on the System One fixture confirms it.
const remoteEmbeddingParaphrase = "Could you walk me through the invoice charge on my subscription and refund it? __SYSTEMONE_YES__"

// remoteEmbeddingRejected is a billing request recall finds the stored
// billing answers for: the mock embeds every billing text alike, so their
// similarity is 1. The fixture's reject marker answers billing_support yes
// and the same_request judge no, so the recalled answer is not served.
const remoteEmbeddingRejected = "What is the pricing of the annual subscription plan? __SYSTEMONE_REJECT__"

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
			Name:             "billing prompt matches billing_support",
			Prompt:           "Please explain the invoice charge on my subscription and issue a refund. __SYSTEMONE_YES__",
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
		response, chatErr := requestRemoteEmbeddingChat(ctx, gatewaySession, testCase.Prompt)
		if chatErr != nil {
			return fmt.Errorf("%s: %w", testCase.Name, chatErr)
		}
		if response.Get("x-vsr-cache-hit") == "true" {
			return fmt.Errorf("%s: a first request hit the response cache", testCase.Name)
		}
		decision := response.Get("x-vsr-selected-decision")
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

	time.Sleep(time.Second)
	paraphrase, err := requestRemoteEmbeddingChat(ctx, gatewaySession, remoteEmbeddingParaphrase)
	if err != nil {
		return fmt.Errorf("paraphrased billing request: %w", err)
	}
	if paraphrase.Get("x-vsr-cache-hit") != "true" || paraphrase.Get("x-vsr-selected-decision") != "billing-route" {
		return fmt.Errorf(
			"paraphrased billing request: expected a billing-route response cache hit, got x-vsr-cache-hit=%q decision=%q",
			paraphrase.Get("x-vsr-cache-hit"),
			paraphrase.Get("x-vsr-selected-decision"),
		)
	}

	// Last, because the rejected request's own answer is stored after it.
	rejected, err := requestRemoteEmbeddingChat(ctx, gatewaySession, remoteEmbeddingRejected)
	if err != nil {
		return fmt.Errorf("rejected billing request: %w", err)
	}
	if rejected.Get("x-vsr-cache-hit") == "true" || rejected.Get("x-vsr-selected-decision") != "billing-route" {
		return fmt.Errorf(
			"rejected billing request: expected a billing-route miss once same_request rejects the recalled answer, got x-vsr-cache-hit=%q decision=%q",
			rejected.Get("x-vsr-cache-hit"),
			rejected.Get("x-vsr-selected-decision"),
		)
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
			"cache_similarity": paraphrase.Get("x-vsr-cache-similarity"),
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

func requestRemoteEmbeddingChat(
	ctx context.Context,
	session *fixtures.ServiceSession,
	prompt string,
) (http.Header, error) {
	payload, err := json.Marshal(map[string]interface{}{
		"model": "vllm-sr/auto",
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		session.URL("/v1/chat/completions"),
		bytes.NewReader(payload),
	)
	if err != nil {
		return nil, fmt.Errorf("create chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-debug", "true")

	resp, err := session.HTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return nil, fmt.Errorf("send chat request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read chat response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected chat status 200, got %d: %s", resp.StatusCode, body)
	}

	if resp.Header.Get("x-vsr-selected-decision") == "" {
		return nil, fmt.Errorf("chat response omitted x-vsr-selected-decision: %s", body)
	}
	return resp.Header, nil
}
