package testcases

import (
	"context"
	"fmt"
	"io"
	"net/http"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-messages-protocol-headers", pkgtestcases.TestCase{
		Description: "Verify x-vsr-inbound-protocol/x-vsr-outbound-protocol markers and absence of lossiness warnings for clean /v1/messages requests",
		Tags:        []string{"anthropic", "headers", "observability"},
		Fn:          testAnthropicMessagesProtocolHeaders,
	})
}

// testAnthropicMessagesProtocolHeaders asserts that the response of a clean
// /v1/messages request carries the protocol-marker headers the router emits
// for every translated request, and that no lossiness warnings are reported
// when the request body uses only fields the router knows how to translate.
// These headers are the operator-facing contract introduced by the warnings
// PR; their presence is what makes a translation cell observable from the
// outside without scraping internal logs.
func testAnthropicMessagesProtocolHeaders(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Verifying protocol marker headers on /v1/messages response")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// Use a deliberately plain request body: only fields the inbound parser
	// translates without warnings (model, max_tokens, single text message,
	// optional system string). Anything richer would risk warnings that
	// later PRs in the series may add.
	resp, err := sendAnthropicMessagesRequest(ctx, anthropicMessagesRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		System:    "You are a helpful assistant.",
		Messages: []anthropicMessage{
			{Role: "user", Content: "Hello."},
		},
	}, localPort)
	if err != nil {
		return fmt.Errorf("anthropic messages request failed: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d", resp.StatusCode)
	}

	// Header names are duplicated as string literals here because the
	// e2e module (e2e/go.mod) is intentionally decoupled from the
	// router module (src/semantic-router/go.mod) and does not import
	// pkg/headers. If a header name changes upstream, this test must
	// be updated explicitly.
	inbound := resp.Header.Get("x-vsr-inbound-protocol")
	outbound := resp.Header.Get("x-vsr-outbound-protocol")
	lossiness := resp.Header.Get("x-vsr-lossiness-warnings")

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":       resp.StatusCode,
			"inbound_protocol":  inbound,
			"outbound_protocol": outbound,
			"lossiness":         lossiness,
		})
	}

	if inbound != "anthropic" {
		return fmt.Errorf("expected x-vsr-inbound-protocol=anthropic, got %q", inbound)
	}
	// Outbound protocol depends on the backend the router selected. In
	// the kubernetes e2e profile the backend is OpenAI-shaped, so
	// outbound must be openai. If a future profile points the backend
	// at an Anthropic-native endpoint this assertion would tighten
	// there.
	if outbound != "openai" {
		return fmt.Errorf("expected x-vsr-outbound-protocol=openai for kubernetes profile, got %q", outbound)
	}
	if lossiness != "" {
		return fmt.Errorf("expected no x-vsr-lossiness-warnings for clean request, got %q", lossiness)
	}

	return nil
}
