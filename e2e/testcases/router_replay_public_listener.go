package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("router-replay-public-listener-denied", pkgtestcases.TestCase{
		Description: "Public inference listeners fail closed for the Router Replay management prefix",
		Tags:        []string{"router-replay", "security", "listener-contract"},
		Fn:          testRouterReplayPublicListenerDenied,
	})
}

func testRouterReplayPublicListenerDenied(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open public listener session: %w", err)
	}
	defer session.Close()

	raw, err := fixtures.DoGETRequest(
		ctx,
		session.HTTPClient(30*time.Second),
		session.BaseURL()+"/v1/router_replay?limit=1",
	)
	if err != nil {
		return fmt.Errorf("GET public Router Replay path: %w", err)
	}
	if raw.StatusCode != http.StatusNotFound {
		return fmt.Errorf(
			"public Router Replay status %d, want 404: %s",
			raw.StatusCode,
			string(raw.Body),
		)
	}
	return nil
}
