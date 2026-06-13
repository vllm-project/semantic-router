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
	pkgtestcases.Register("router-replay-large-request-list-ok", pkgtestcases.TestCase{
		Description: "GET /v1/router_replay returns 200 (not 413) after a large request is recorded",
		Tags:        []string{"router-replay", "functional", "router-replay-api"},
		Fn:          testRouterReplayLargeRequestListOK,
	})
}

// testRouterReplayLargeRequestListOK is the end-to-end regression for the
// history-list 413: a large chat request is recorded, then the replay list
// endpoint is queried. Before the fix a single multi-MB request produced a
// replay record whose structured fields pushed the serialized page past the
// ext-proc message-size cap, so GET /v1/router_replay returned 413 and the
// whole history view broke. The list endpoint must now stay 200 and remain
// listable regardless of how large the recorded request was.
func testRouterReplayLargeRequestListOK(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Router replay: large request must not 413 the history list")
	}
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()

	sess := fmt.Sprintf("e2e_large_%d", time.Now().UnixNano())
	user := "e2e-replay-large-user"

	// ~5 MB of user content — the size class that triggered the reported 413.
	largeContent := "router replay large-request regression " + sess + " " +
		strings.Repeat("x", 5*1024*1024)

	if err := postChatCompletionsWithReplayHeaders(ctx, session, postChatOpts{
		sessionID: sess,
		userID:    user,
		content:   largeContent,
	}); err != nil {
		return fmt.Errorf("post large chat completion: %w", err)
	}
	time.Sleep(2 * time.Second)

	// The list endpoint must return 200, not 413, and the page must decode.
	q := "limit=25&offset=0"
	httpClient := session.HTTPClient(30 * time.Second)
	raw, err := fixtures.DoGETRequest(ctx, httpClient, session.BaseURL()+"/v1/router_replay?"+q)
	if err != nil {
		return fmt.Errorf("GET router_replay list: %w", err)
	}
	if raw.StatusCode == http.StatusRequestEntityTooLarge {
		return fmt.Errorf("router_replay list returned 413 after a large request was recorded (the bug this guards against)")
	}
	if raw.StatusCode != http.StatusOK {
		return fmt.Errorf("GET router_replay list status %d: %s", raw.StatusCode, truncateE2EBody(raw.Body))
	}

	var listResp replayListResponse
	if err := raw.DecodeJSON(&listResp); err != nil {
		return fmt.Errorf("decode list after large request: %w", err)
	}
	if listResp.Count == 0 {
		return fmt.Errorf("expected at least one replay row after recording a large request, got 0")
	}

	if opts.Verbose {
		fmt.Printf("[Test] large-request replay list ok: status=200 count=%d\n", listResp.Count)
	}
	return nil
}

// truncateE2EBody keeps error output readable when the body is large.
func truncateE2EBody(body []byte) string {
	const max = 200
	if len(body) <= max {
		return string(body)
	}
	return string(body[:max]) + "...(truncated)"
}
