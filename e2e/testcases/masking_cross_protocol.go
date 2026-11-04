package testcases

import (
	"bytes"
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

// This profile has no PII model to host, so it drives PII detection through
// the same remote token_spans.v1 stub server ("mock-pii-spans") the
// pii-remote-backend profile uses, and the same "provider-mocker" echo backend used
// throughout this suite as the chat backend. Its default completion
// is a deterministic JSON echo of what it received
// (provider-mocker chat_request.py, build_chat_content):
// {"user":["<raw text of every user message>"], ...}.
// That is the only way this harness can assert on the exact provider-bound
// bytes, since Router Replay is deliberately blind on a masking route (D3) --
// the request body it would otherwise expose is exactly what this feature
// suppresses.
const (
	maskingSpanMarker  = "__PII_SPAN__"
	maskingErrorMarker = "__PII_ERROR__"
	maskingRawEmail    = "alice@corp.example"
	// An explicit entrypoint, not a concrete model: a concrete model name
	// bypasses every recipe-local decision and plugin, so masking would never
	// run and these assertions would pass vacuously.
	maskingEntrypoint = "vllm-sr/masking"
	// The decision inside the masking recipe, asserted on every case so a
	// bypassed decision fails loudly instead of passing vacuously.
	maskingDecisionName = "mask_pii"
)

// maskingEcho is the subset of provider-mocker's deterministic echo this
// profile reads. Every user message the backend received lands in User,
// verbatim.
type maskingEcho struct {
	User []string `json:"user"`
}

func init() {
	pkgtestcases.Register("masking-cross-protocol", pkgtestcases.TestCase{
		Description: "An equivalent PII-bearing request through Chat, Responses and Anthropic all reach the mock backend with a placeholder and no raw value",
		Tags:        []string{"masking", "pii", "cross-protocol", "functional"},
		Fn:          testMaskingCrossProtocol,
	})
	pkgtestcases.Register("masking-classifier-unavailable-fails-closed", pkgtestcases.TestCase{
		Description: "A classifier error on a masking-enabled decision returns 503 and dispatches nothing",
		Tags:        []string{"masking", "pii", "fail-closed"},
		Fn:          testMaskingClassifierUnavailableFailsClosed,
	})
}

type maskingWireCase struct {
	name string
	path string
	body map[string]interface{}
}

// testMaskingCrossProtocol is this feature's headline claim: because every
// wire format decodes into one neutral request before plugins run, the same
// masking happens regardless of which client protocol carried the PII. It is
// the only place that claim is exercised through the real codec stack rather
// than a stub scanner (unlike the pkg/masking unit tests).
func testMaskingCrossProtocol(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	// The stub reports exactly one span, over the first occurrence of the
	// marker's argument, so the value appears once: a second occurrence would
	// stay raw and the assertion below would blame masking for the stub.
	prompt := fmt.Sprintf("%s EMAIL_ADDRESS %s please forward the invoice", maskingSpanMarker, maskingRawEmail)
	cases := []maskingWireCase{
		{
			name: "chat completions",
			path: "/v1/chat/completions",
			body: map[string]interface{}{
				"model":    maskingEntrypoint,
				"messages": []map[string]interface{}{{"role": "user", "content": prompt}},
			},
		},
		{
			name: "responses",
			path: "/v1/responses",
			body: map[string]interface{}{
				"model": maskingEntrypoint,
				"store": false,
				"input": []map[string]interface{}{{
					"role":    "user",
					"content": []map[string]interface{}{{"type": "input_text", "text": prompt}},
				}},
			},
		},
		{
			name: "anthropic messages",
			path: "/v1/messages",
			body: map[string]interface{}{
				"model":      maskingEntrypoint,
				"max_tokens": 64,
				"messages":   []map[string]interface{}{{"role": "user", "content": prompt}},
			},
		},
	}

	echoedUsers := make(map[string]string, len(cases))
	for _, tc := range cases {
		echoed, err := runMaskingWireCase(ctx, session, tc)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if strings.Contains(echoed, maskingRawEmail) {
			return fmt.Errorf("%s: raw value reached the backend: %q", tc.name, echoed)
		}
		if !strings.Contains(echoed, "[EMAIL_ADDRESS_0]") {
			return fmt.Errorf("%s: backend did not receive a placeholder: %q", tc.name, echoed)
		}
		echoedUsers[tc.name] = echoed
	}

	// Byte-for-byte identical, not just "both masked": the neutral IR
	// converges to one splice regardless of which protocol decoded it.
	want := echoedUsers[cases[0].name]
	for _, tc := range cases[1:] {
		if echoedUsers[tc.name] != want {
			return fmt.Errorf(
				"masked content diverged across protocols:\n%s: %q\n%s: %q",
				cases[0].name, want, tc.name, echoedUsers[tc.name],
			)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"masked_content": want,
			"protocols":      "chat,responses,anthropic",
		})
	}
	return nil
}

// testMaskingClassifierUnavailableFailsClosed drives the remote classifier
// into its declared-error path (mock-pii-spans returns a 200 carrying an
// error member) on a masking-enabled decision, and asserts the request is
// refused rather than dispatched unmasked (D4).
func testMaskingClassifierUnavailableFailsClosed(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	prompt := fmt.Sprintf("%s contact %s please", maskingErrorMarker, maskingRawEmail)
	resp, err := runMaskingRawRequest(ctx, session, "/v1/chat/completions", map[string]interface{}{
		"model":    maskingEntrypoint,
		"messages": []map[string]interface{}{{"role": "user", "content": prompt}},
	})
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		return fmt.Errorf("status = %d, want 503: %s", resp.StatusCode, truncateString(string(resp.Body), 300))
	}
	if strings.Contains(string(resp.Body), maskingRawEmail) {
		return fmt.Errorf("the refusal response echoes request content: %s", string(resp.Body))
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"status": resp.StatusCode})
	}
	return nil
}

type maskingRawResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

func runMaskingRawRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	path string,
	payload map[string]interface{},
) (*maskingRawResponse, error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, session.BaseURL()+path, bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	if path == "/v1/messages" {
		request.Header.Set("anthropic-version", "2023-06-01")
	}
	// v0.4 demotes the selected-decision header behind x-vsr-debug (#2205);
	// opt in so the assertions can prove the masking decision was selected.
	request.Header.Set("x-vsr-debug", "true")
	response, err := session.HTTPClient(45 * time.Second).Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	return &maskingRawResponse{
		StatusCode: response.StatusCode,
		Headers:    response.Header,
		Body:       body,
	}, nil
}

// runMaskingWireCase sends one case's request and returns the exact user-turn
// text provider-mocker reports having received, decoded from the protocol-specific
// completion shape.
func runMaskingWireCase(ctx context.Context, session *fixtures.ServiceSession, tc maskingWireCase) (string, error) {
	resp, err := runMaskingRawRequest(ctx, session, tc.path, tc.body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%s returned HTTP %d: %s", tc.path, resp.StatusCode, truncateString(string(resp.Body), 500))
	}
	// Without this the whole case can pass vacuously: a request that bypasses
	// the decision never runs the plugin, so "no raw value" would prove nothing.
	if decision := resp.Headers.Get("x-vsr-selected-decision"); decision != maskingDecisionName {
		return "", fmt.Errorf(
			"selected decision = %q, want %q; the request bypassed the masking decision so nothing was masked",
			decision, maskingDecisionName,
		)
	}
	content, err := extractAssistantText(tc.path, resp.Body)
	if err != nil {
		return "", fmt.Errorf("extract assistant text: %w (body: %s)", err, truncateString(string(resp.Body), 300))
	}
	var echo maskingEcho
	if err := json.Unmarshal([]byte(content), &echo); err != nil {
		return "", fmt.Errorf("completion is not the provider-mocker echo JSON: %w (%q)", err, content)
	}
	if len(echo.User) != 1 {
		return "", fmt.Errorf("provider-mocker echo carries %d user messages, want 1: %q", len(echo.User), content)
	}
	return echo.User[0], nil
}

// extractAssistantText pulls the assistant's text out of a Chat, Responses,
// or Anthropic Messages completion body.
func extractAssistantText(path string, body []byte) (string, error) {
	switch path {
	case "/v1/chat/completions":
		var decoded struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &decoded); err != nil {
			return "", err
		}
		if len(decoded.Choices) == 0 {
			return "", fmt.Errorf("no choices in chat completion")
		}
		return decoded.Choices[0].Message.Content, nil
	case "/v1/responses":
		var decoded struct {
			OutputText string `json:"output_text"`
		}
		if err := json.Unmarshal(body, &decoded); err != nil {
			return "", err
		}
		if decoded.OutputText == "" {
			return "", fmt.Errorf("empty output_text in responses completion")
		}
		return decoded.OutputText, nil
	case "/v1/messages":
		var decoded struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(body, &decoded); err != nil {
			return "", err
		}
		for _, block := range decoded.Content {
			if block.Type == "text" {
				return block.Text, nil
			}
		}
		return "", fmt.Errorf("no text block in anthropic completion")
	default:
		return "", fmt.Errorf("unsupported path %q", path)
	}
}
