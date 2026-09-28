package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	jsonPromptMarkerDecision   = "json_prompt_keyword"
	jsonPromptFallbackDecision = "route_to_anthropic_provider"
	jsonPromptMarker           = "signalonlyjsonmarker"
)

func init() {
	pkgtestcases.Register("prompt-compression-json-routing", pkgtestcases.TestCase{
		Description: "Dense JSON and code trigger signal-only compression while the Anthropic backend receives the original request",
		Tags:        []string{"prompt-compression", "routing", "anthropic", "regression"},
		Fn:          testPromptCompressionJSONRouting,
	})
}

// The oversized JSON is one sentence. Once #4168's dense-text counter detects
// that it exceeds the configured budget, the compressor cannot retain it.
// Prior to the fix, the whitespace counter priced the whole JSON as one token
// and the marker route was selected. The provider payload stays uncompressed.
func testPromptCompressionJSONRouting(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	router, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer router.Close()

	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "anthropic.messages.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	shortPrompt := "Route this short " + jsonPromptMarker + " request."
	if assertionErr := assertJSONPromptCompressionRoute(ctx, router, provider, shortPrompt, jsonPromptMarkerDecision); assertionErr != nil {
		return fmt.Errorf("short marker control: %w", assertionErr)
	}

	record := `{"id":"req_0001","tag":"` + jsonPromptMarker + `","status":"ok","latency_ms":157},`
	longPrompt := "Inspect the catalog. " + strings.Repeat(record, 80) + " Give a count."
	if len(longPrompt) <= 4096 {
		return fmt.Errorf("json fixture is only %d bytes; it must cross the configured min_length", len(longPrompt))
	}
	if assertionErr := assertJSONPromptCompressionRoute(ctx, router, provider, longPrompt, jsonPromptFallbackDecision); assertionErr != nil {
		return fmt.Errorf("dense JSON prompt: %w", assertionErr)
	}
	// Keep the minified expression in one sentence. Semicolons split it into
	// small chunks that the compressor can legitimately retain.
	codeTerm := `item["tag"]==="` + jsonPromptMarker + `"&&handle(item),`
	codePrompt := "Inspect the code. const flags=[" + strings.Repeat(codeTerm, 150) + "] Give a count."
	if len(codePrompt) <= 4096 {
		return fmt.Errorf("code fixture is only %d bytes; it must cross the configured min_length", len(codePrompt))
	}
	if assertionErr := assertJSONPromptCompressionRoute(ctx, router, provider, codePrompt, jsonPromptFallbackDecision); assertionErr != nil {
		return fmt.Errorf("dense code prompt: %w", assertionErr)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"short_prompt_decision": jsonPromptMarkerDecision,
			"json_prompt_decision":  jsonPromptFallbackDecision,
			"json_prompt_bytes":     len(longPrompt),
			"code_prompt_bytes":     len(codePrompt),
			"provider_input_intact": true,
		})
	}
	return nil
}

func assertJSONPromptCompressionRoute(
	ctx context.Context,
	router, provider *fixtures.ServiceSession,
	prompt, wantDecision string,
) error {
	sessionID := fmt.Sprintf("json-compression-%d", time.Now().UnixNano())
	response, err := sendProtocolMatrixRaw(ctx, router, "/v1/chat/completions", map[string]any{
		"model":      "auto",
		"max_tokens": 32,
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
	}, false, map[string]string{
		"x-vsr-debug":           "true",
		"x-vsr-test-session-id": sessionID,
	})
	if err != nil {
		return fmt.Errorf("send routed request: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("routed request returned HTTP %d: %s", response.StatusCode, truncateString(string(response.Body), 500))
	}
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != wantDecision {
		return fmt.Errorf("selected decision %q, want %q (matched keywords: %q)",
			decision, wantDecision, response.Headers.Get("x-vsr-matched-keywords"))
	}
	matched := response.Headers.Get("x-vsr-matched-keywords")
	if hasMarker := strings.Contains(matched, "json_prompt_marker"); hasMarker != (wantDecision == jsonPromptMarkerDecision) {
		return fmt.Errorf("matched keyword header %q disagrees with decision %q", matched, wantDecision)
	}

	observed, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("inspect Anthropic provider request: %w", err)
	}
	var recorded struct {
		Body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			} `json:"messages"`
		} `json:"body"`
	}
	if decodeErr := json.Unmarshal(observed, &recorded); decodeErr != nil {
		return fmt.Errorf("decode Anthropic provider request: %w", decodeErr)
	}
	if len(recorded.Body.Messages) != 1 || recorded.Body.Messages[0].Role != "user" ||
		len(recorded.Body.Messages[0].Content) != 1 ||
		recorded.Body.Messages[0].Content[0].Type != "text" ||
		recorded.Body.Messages[0].Content[0].Text != prompt {
		return fmt.Errorf("provider input changed after signal compression (want %d bytes): %s",
			len(prompt), truncateString(string(observed), 500))
	}
	return nil
}
