package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("standalone-debug-headers-default-off", pkgtestcases.TestCase{
		Description: "A client's x-vsr-debug is ignored unless global.router.headers.debug allows it",
		Tags:        []string{"standalone", "gateway", "security"},
		Fn:          testStandaloneDebugHeadersDefaultOff,
	})
}

// testStandaloneDebugHeadersDefaultOff asks for the debug headers on a request
// that matches the profile's keyword rule. The profile leaves
// global.router.headers.debug unset, so the response keeps the default surface
// and names no matched rule.
func testStandaloneDebugHeadersDefaultOff(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	response, err := fixtures.NewChatCompletionsClient(session, 60*time.Second).Create(ctx, fixtures.ChatCompletionsRequest{
		Model:    "vllm-sr/auto",
		Messages: []fixtures.ChatMessage{{Role: "user", Content: standaloneFailoverQuery}},
	}, map[string]string{"x-vsr-debug": "true"})
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("chat completion status %d: %s", response.StatusCode, truncateString(string(response.Body), 400))
	}
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != standaloneFailoverRoute {
		return fmt.Errorf("decision %q, want %q on the default surface", decision, standaloneFailoverRoute)
	}
	for _, name := range []string{"x-vsr-matched-keywords", "x-vsr-selected-reasoning", "x-vsr-context-token-count", "x-vsr-client-protocol"} {
		if value := response.Headers.Get(name); value != "" {
			return fmt.Errorf("%s = %q: a client turned on the debug headers without global.router.headers.debug", name, value)
		}
	}
	return nil
}
