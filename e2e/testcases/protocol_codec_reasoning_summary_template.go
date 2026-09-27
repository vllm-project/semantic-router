package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	templateResponsesBackendModel = "nemotron-summary-responses"
	templateResponsesDecision     = "reasoning_summary_template_decision"
)

func init() {
	pkgtestcases.Register("protocol-codec-reasoning-summary-template-responses-backend", pkgtestcases.TestCase{
		Description: "Responses summary is dropped with a warning when provider reasoning uses chat_template_kwargs",
		Tags:        []string{"protocol-codec", "response-api", "agents", "reasoning"},
		Fn:          testProtocolCodecReasoningSummaryTemplateResponsesBackend,
	})
}

func testProtocolCodecReasoningSummaryTemplateResponsesBackend(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.responses.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	const marker = "reasoning_summary_template_probe"
	const sessionID = "reasoning-summary-template-responses"
	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/responses", map[string]any{
		"model": "auto", "input": marker, "store": false,
		"reasoning": map[string]string{"summary": "auto"},
	}, false, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return fmt.Errorf("template Responses summary request: %w", err)
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("template Responses summary returned HTTP %d: %s", result.StatusCode, truncateString(string(result.Body), 500))
	}
	if decision := result.Headers.Get("x-vsr-selected-decision"); decision != templateResponsesDecision {
		return fmt.Errorf("template Responses selected decision = %q, want %q", decision, templateResponsesDecision)
	}
	if model := result.Headers.Get("x-vsr-selected-model"); model != templateResponsesBackendModel {
		return fmt.Errorf("template Responses selected model = %q, want %q", model, templateResponsesBackendModel)
	}
	if bodyErr := assertResponsesBody(result.Body, protocolCodecResponsesReply); bodyErr != nil {
		return fmt.Errorf("template Responses summary response: %w", bodyErr)
	}
	if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !hasProtocolFieldDiagnostic(warnings, "dropped", "reasoning.summary") {
		return fmt.Errorf("template Responses did not report dropped reasoning.summary: %q", warnings)
	}

	raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("template Responses provider observation: %w", err)
	}
	var observed struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &observed); err != nil {
		return fmt.Errorf("decode template Responses provider observation: %w", err)
	}
	if !strings.Contains(string(raw), marker) || len(observed.Body["input"]) == 0 {
		return fmt.Errorf("template Responses provider observation does not match request: %s", truncateString(string(raw), 500))
	}
	if string(observed.Body["model"]) != `"nemotron-3-nano-omni"` {
		return fmt.Errorf("template Responses provider model = %s, want nemotron-3-nano-omni", observed.Body["model"])
	}
	if _, found := observed.Body["reasoning"]; found {
		return fmt.Errorf("template Responses forwarded reasoning object: %s", observed.Body["reasoning"])
	}
	var kwargs map[string]json.RawMessage
	if err := json.Unmarshal(observed.Body["chat_template_kwargs"], &kwargs); err != nil {
		return fmt.Errorf("decode template Responses chat_template_kwargs: %w", err)
	}
	if string(kwargs["enable_thinking"]) != "true" {
		return fmt.Errorf("template Responses provider lost enable_thinking control: %s", observed.Body["chat_template_kwargs"])
	}
	return nil
}
