package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const vllmReasoningBudgetMarker = "__mock_vllm_reasoning_budget__"

func init() {
	pkgtestcases.Register("protocol-codec-vllm-reasoning-budget", pkgtestcases.TestCase{
		Description: "A neutral reasoning budget reaches the vLLM Chat provider as thinking_token_budget",
		Tags:        []string{"protocol-codec", "vllm", "reasoning"},
		Fn:          testProtocolCodecVLLMReasoningBudget,
	})
}

func testProtocolCodecVLLMReasoningBudget(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	sessionID := "vllm-reasoning-budget-projection-" + uuid.NewString()
	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", map[string]any{
		"model":                   chatBackendModel,
		"messages":                []map[string]string{{"role": "user", "content": vllmReasoningBudgetMarker}},
		"reasoning_effort":        "medium",
		"reasoning_budget_tokens": 1024,
	}, false, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return fmt.Errorf("vLLM reasoning budget request: %w", err)
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("vLLM reasoning budget request returned HTTP %d: %s",
			result.StatusCode, truncateString(string(result.Body), 700))
	}
	if decision := result.Headers.Get("x-vsr-selected-decision"); decision != "reasoning_budget_projection_decision" {
		return fmt.Errorf("reasoning budget selected decision = %q, want reasoning_budget_projection_decision", decision)
	}

	raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("vLLM reasoning budget provider observation: %w", err)
	}
	var observed struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &observed); err != nil {
		return fmt.Errorf("decode vLLM reasoning budget provider request: %w", err)
	}
	if string(observed.Body["thinking_token_budget"]) != "1024" {
		return fmt.Errorf("vLLM thinking_token_budget = %s, want 1024: %s",
			observed.Body["thinking_token_budget"], truncateString(string(raw), 700))
	}
	if _, found := observed.Body["reasoning_budget_tokens"]; found {
		return fmt.Errorf("vLLM provider request retained the neutral budget: %s", truncateString(string(raw), 700))
	}
	effort := observed.Body["reasoning_effort"]
	if len(effort) == 0 {
		var kwargs map[string]json.RawMessage
		if err := json.Unmarshal(observed.Body["chat_template_kwargs"], &kwargs); err != nil {
			return fmt.Errorf("decode vLLM chat_template_kwargs: %w", err)
		}
		effort = kwargs["reasoning_effort"]
	}
	if string(effort) != `"medium"` {
		return fmt.Errorf("vLLM reasoning_effort = %s, want medium: %s",
			effort, truncateString(string(raw), 700))
	}
	if !strings.Contains(string(raw), vllmReasoningBudgetMarker) {
		return fmt.Errorf("vLLM provider request lost the probe marker: %s", truncateString(string(raw), 700))
	}
	return nil
}
