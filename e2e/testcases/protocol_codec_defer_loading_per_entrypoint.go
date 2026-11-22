package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-defer-loading-per-entrypoint", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "One gateway accepts Claude Code's deferred tools on its entrypoint and refuses them on an entrypoint that accepts no loss",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecDeferLoadingPerEntrypoint,
	})
}

// testProtocolCodecDeferLoadingPerEntrypoint sends the same deferred tool to
// both entrypoints of the opt-in Router: vllm-sr/claude-code takes the global
// losses, which accept tools.defer_loading, and vllm-sr/codex sets an empty
// list, so it refuses the tool before routing.
func testProtocolCodecDeferLoadingPerEntrypoint(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	routerOpts := opts
	routerOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace: ToolResultOptInNamespace, Name: ToolResultOptInRelease, ServicePort: ToolResultOptInPort,
	}
	session, err := fixtures.OpenServiceSession(ctx, client, routerOpts)
	if err != nil {
		return err
	}
	defer session.Close()
	for _, check := range []struct {
		model  string
		status int
	}{
		{"vllm-sr/claude-code", http.StatusOK},
		{"vllm-sr/codex", http.StatusBadRequest},
	} {
		body := map[string]any{
			"model": check.model, "max_tokens": 64,
			"messages": []any{map[string]any{"role": "user", "content": "Fetch the page"}},
			"tools": []any{
				map[string]any{"name": "WebFetch", "input_schema": map[string]any{"type": "object"}, "defer_loading": true},
			},
		}
		result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, false,
			map[string]string{"x-vsr-test-session-id": "defer-loading-per-entrypoint-" + uuid.NewString()})
		if err != nil {
			return fmt.Errorf("%s: %w", check.model, err)
		}
		if result.StatusCode != check.status {
			return fmt.Errorf("%s: status=%d, want %d: body=%s",
				check.model, result.StatusCode, check.status, truncateString(string(result.Body), 500))
		}
		if check.status == http.StatusBadRequest && !strings.Contains(string(result.Body), "tools.defer_loading") {
			return fmt.Errorf("%s: refusal does not name tools.defer_loading: %s", check.model, truncateString(string(result.Body), 500))
		}
	}
	return nil
}
