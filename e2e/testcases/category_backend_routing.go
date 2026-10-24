package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("category-backend-routing", pkgtestcases.TestCase{
		Description: "Verify category backend distribution drives domain routing",
		Tags:        []string{"classifier", "http-classify", "routing"},
		Fn:          testCategoryBackendRouting,
	})
}

func testCategoryBackendRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()
	resp, err := sendLocalChatCompletion(ctx, localPort, "vllm-sr/auto", "__CATEGORY_BACKEND_MATH__ route this request", 30*time.Second)
	if err != nil {
		return fmt.Errorf("remote category request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("remote category request: %s", formatUnexpectedChatCompletionStatus(resp))
	}
	if got := resp.Headers.Get("x-vsr-selected-decision"); got != "category_backend_math" {
		return fmt.Errorf("x-vsr-selected-decision=%q, want category_backend_math", got)
	}
	// Trip on two exhausted calls, then send content the healthy mock would
	// classify as math. The cooldown must preserve fallback routing.
	for i, prompt := range []string{
		"__CATEGORY_BACKEND_UNAVAILABLE__ first",
		"__CATEGORY_BACKEND_UNAVAILABLE__ second",
		"__CATEGORY_BACKEND_MATH__ during cooldown",
	} {
		response, err := sendLocalChatCompletion(ctx, localPort, "vllm-sr/auto", prompt, 30*time.Second)
		if err != nil {
			return fmt.Errorf("breaker request %d: %w", i, err)
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("breaker request %d: %s", i, formatUnexpectedChatCompletionStatus(response))
		}
		if got := response.Headers.Get("x-vsr-selected-decision"); got != "default-route" {
			return fmt.Errorf("breaker request %d selected %q, want default-route", i, got)
		}
	}
	return nil
}
