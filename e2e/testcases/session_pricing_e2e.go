package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("session-pricing-chat-completions", pkgtestcases.TestCase{
		Description: "After a routed chat completion, Prometheus exposes llm_session_turn_cost histogram when model pricing is configured",
		Tags:        []string{"kubernetes", "observability", "metrics", "llm", "pricing"},
		Fn:          testSessionPricingChatCompletions,
	})
}

// testSessionPricingChatCompletions verifies that after a Chat Completions request the
// llm_session_turn_cost histogram is present in /metrics (pricing must be configured
// for the routed model in router-config.yaml for the observation to appear).
func testSessionPricingChatCompletions(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	traffic, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer traffic.Close()

	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer metricsSession.Close()

	chat := fixtures.NewChatCompletionsClient(traffic, 60*time.Second)

	headers := map[string]string{
		"x-authz-user-id": "e2e-pricing-chat-user",
	}
	resp, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "vllm-sr/auto",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "Say hello in one short sentence for pricing telemetry."},
		},
		User: "e2e-pricing-chat-user",
	}, headers)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("chat completion: expected 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	body, err := fetchMetrics(ctx, metricsSession)
	if err != nil {
		return err
	}

	// Token histograms from PR 1 must still be present.
	if !strings.Contains(body, "llm_session_turn_prompt_tokens") {
		return fmt.Errorf("metrics body missing llm_session_turn_prompt_tokens")
	}
	if !strings.Contains(body, "llm_session_turn_completion_tokens") {
		return fmt.Errorf("metrics body missing llm_session_turn_completion_tokens")
	}
	// Cost histogram descriptor must be registered (present even when no observations).
	if !strings.Contains(body, "llm_session_turn_cost") {
		return fmt.Errorf("metrics body missing llm_session_turn_cost")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"chat_status": resp.StatusCode,
		})
	}
	return nil
}

// testSessionPricingResponseAPI verifies that after a Response API request the
// llm_session_turn_cost histogram descriptor is exposed in /metrics.
func fetchMetrics(ctx context.Context, metricsSession *fixtures.ServiceSession) (string, error) {
	metricsHTTP := metricsSession.HTTPClient(15 * time.Second)
	metricsResp, err := fixtures.DoGETRequest(ctx, metricsHTTP, metricsSession.URL("/metrics"))
	if err != nil {
		return "", fmt.Errorf("fetch /metrics: %w", err)
	}
	if metricsResp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("/metrics: expected 200, got %d", metricsResp.StatusCode)
	}
	return string(metricsResp.Body), nil
}
