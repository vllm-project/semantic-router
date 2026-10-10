package testcases

import (
	"context"
	"encoding/json"
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
		Description: "An empty POST to any inference path gets the Router's own 400 and never reaches the backend (issue #4292)",
		Tags:        []string{"security", "listener-contract", "protocol"},
		Fn:          testPublicListenerEmptyBodyRejected,
	})
}

// emptyBodyRoutes are the #4292 routes; the Responses paths need response_api.enabled, which provider-protocols sets.
var emptyBodyRoutes = []struct {
	path      string
	anthropic bool
}{
	{path: "/v1/chat/completions"},
	{path: "/v1/messages", anthropic: true},
	{path: "/v1/responses"},
	{path: "/openai/v1/chat/completions"},
	{path: "/openai/responses"},
	{path: "/openai/v1/responses"},
	{path: "/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"},
}

// emptyBodyLeakHeaders are upstream headers #4292 saw relayed on the bodyless request.
var emptyBodyLeakHeaders = []string{"Set-Cookie", "Access-Control-Allow-Origin"}

type emptyBodyErrorEnvelope struct {
	Type  string `json:"type"`
	Error *struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
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
	for _, route := range emptyBodyRoutes {
		if err := assertEmptyBodyRejected(ctx, httpClient, session.BaseURL()+route.path, route.anthropic); err != nil {
			return fmt.Errorf("POST %s: %w", route.path, err)
		}
	}
	return nil
}

func assertEmptyBodyRejected(ctx context.Context, httpClient *http.Client, url string, anthropic bool) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, http.NoBody)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Origin", "https://e2e.example")
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
	for _, name := range emptyBodyLeakHeaders {
		if values := resp.Header.Values(name); len(values) > 0 {
			return fmt.Errorf("upstream header %s relayed: %q", name, values)
		}
	}
	return checkEmptyBodyEnvelope(body, anthropic)
}

func checkEmptyBodyEnvelope(body []byte, anthropic bool) error {
	var envelope emptyBodyErrorEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Error == nil {
		return fmt.Errorf("body is not an error envelope: %s", body)
	}
	if !strings.Contains(envelope.Error.Message, "request body is empty") {
		return fmt.Errorf("body is not the Router empty-body error: %s", body)
	}
	if envelope.Error.Type != "invalid_request_error" {
		return fmt.Errorf("error.type %q, want invalid_request_error: %s", envelope.Error.Type, body)
	}
	if anthropic {
		if envelope.Type != "error" {
			return fmt.Errorf("type %q, want the Anthropic error envelope: %s", envelope.Type, body)
		}
		return nil
	}
	if envelope.Error.Code != "body_limit" {
		return fmt.Errorf("error.code %q, want body_limit in the OpenAI envelope: %s", envelope.Error.Code, body)
	}
	return nil
}
