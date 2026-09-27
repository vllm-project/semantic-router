package testcases

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("public-listener-empty-body-rejected", pkgtestcases.TestCase{
		Description: "An empty POST to an inference path gets the Router's own 400 and never reaches the backend (issue #4292)",
		Tags:        []string{"security", "listener-contract", "protocol"},
		Fn:          testPublicListenerEmptyBodyRejected,
	})
}

func testPublicListenerEmptyBodyRejected(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open public listener session: %w", err)
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	for _, path := range []string{"/v1/chat/completions", "/v1/messages"} {
		if err := assertEmptyBodyRejected(ctx, httpClient, session.BaseURL()+path); err != nil {
			return fmt.Errorf("POST %s: %w", path, err)
		}
	}
	return nil
}

func assertEmptyBodyRejected(ctx context.Context, httpClient *http.Client, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, http.NoBody)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("status %d, want 400: %s", resp.StatusCode, body)
	}
	if got := resp.Header.Get("x-vsr-response-path"); got != "error" {
		return fmt.Errorf("x-vsr-response-path %q, want the Router error path", got)
	}
	if !strings.Contains(string(body), "request body is empty") {
		return fmt.Errorf("body is not the Router empty-body error: %s", body)
	}
	return nil
}
