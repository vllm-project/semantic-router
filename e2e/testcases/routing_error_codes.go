package testcases

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("routing-error-codes", pkgtestcases.TestCase{
		Description: "A request the Router cannot route gets the Router's own 400 with the documented reason code in error.code (issue #4653)",
		Tags:        []string{"routing", "errors", "gateway"},
		Fn:          testRoutingErrorCodes,
	})
}

// routingErrorProbe is a request no profile running this case can route: no
// provider model has the first name, and the declared no-route recipe has no
// decision that can select a backend.
type routingErrorProbe struct {
	name    string
	model   string
	code    string
	message string
}

var routingErrorProbes = []routingErrorProbe{
	{name: "unknown model", model: "e2e-no-such-model", code: "model_not_found", message: "the requested model is not available"},
	{name: "declared recipe without a route", model: "e2e-no-route", code: "no_route", message: "no route matched the request"},
	{name: "undeclared legacy alias", model: "MoM", code: "model_not_found", message: "the requested model is not available"},
	{name: "undeclared algorithm alias", model: "vllm-sr/flow", code: "model_not_found", message: "the requested model is not available"},
}

func testRoutingErrorCodes(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	httpClient := session.HTTPClient(60 * time.Second)
	var failures []error
	for _, probe := range routingErrorProbes {
		if err := checkRoutingErrorProbe(ctx, httpClient, session.URL("/v1/chat/completions"), probe); err != nil {
			failures = append(failures, fmt.Errorf("%s: %w", probe.name, err))
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"probes": len(routingErrorProbes), "failed": len(failures)})
	}
	return errors.Join(failures...)
}

func checkRoutingErrorProbe(ctx context.Context, httpClient *http.Client, url string, probe routingErrorProbe) error {
	response, err := fixtures.DoPOSTRequest(ctx, httpClient, url, fixtures.ChatCompletionsRequest{
		Model:    probe.model,
		Messages: []fixtures.ChatMessage{{Role: "user", Content: "Say hello."}},
	})
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("status %d, want 400: %s", response.StatusCode, truncateString(string(response.Body), 400))
	}
	// A backend's 400 would not carry the Router's error response path.
	if path := response.Headers.Get("x-vsr-response-path"); path != "error" {
		return fmt.Errorf("x-vsr-response-path %q, want the Router's own error", path)
	}
	var body struct {
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := response.DecodeJSON(&body); err != nil {
		return fmt.Errorf("%w: %s", err, truncateString(string(response.Body), 400))
	}
	if body.Error.Type != "invalid_request_error" || body.Error.Code != probe.code || body.Error.Message != probe.message {
		return fmt.Errorf("error %+v, want an invalid_request_error with code %q and message %q", body.Error, probe.code, probe.message)
	}
	return nil
}
