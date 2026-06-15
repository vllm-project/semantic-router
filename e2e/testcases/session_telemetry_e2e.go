package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("session-telemetry-metrics", pkgtestcases.TestCase{
		Description: "After a routed chat completion, Prometheus exposes llm_session_turn_* histograms on the router metrics port",
		Tags:        []string{"kubernetes", "observability", "metrics", "llm"},
		Fn:          testSessionTelemetryMetrics,
	})
}

func testSessionTelemetryMetrics(
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
		"x-authz-user-id": "e2e-session-telemetry-user",
	}
	resp, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "MoM",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "Say hello in one short sentence for session telemetry."},
		},
		User: "e2e-session-telemetry-user",
	}, headers)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("chat completion: expected 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	metricsHTTP := metricsSession.HTTPClient(15 * time.Second)
	metricsResp, err := fixtures.DoGETRequest(ctx, metricsHTTP, metricsSession.URL("/metrics"))
	if err != nil {
		return fmt.Errorf("fetch /metrics: %w", err)
	}
	if metricsResp.StatusCode != http.StatusOK {
		return fmt.Errorf("/metrics: expected 200, got %d", metricsResp.StatusCode)
	}
	body := string(metricsResp.Body)
	if !strings.Contains(body, "llm_session_turn_prompt_tokens") {
		return fmt.Errorf("metrics body missing llm_session_turn_prompt_tokens")
	}
	if !strings.Contains(body, "llm_session_turn_completion_tokens") {
		return fmt.Errorf("metrics body missing llm_session_turn_completion_tokens")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"chat_status": resp.StatusCode,
		})
	}
	return nil
}
