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

const nullReasoningBudgetOptInModel = "mock/opt-in-ollama-chat"

func init() {
	pkgtestcases.Register("protocol-codec-null-reasoning-budget-opt-in", pkgtestcases.TestCase{
		Description: "A null Chat reasoning budget is omitted for an opted-in non-vLLM provider, while vLLM-native budget fields remain private",
		Tags:        []string{"protocol-codec", "ollama", "reasoning"},
		Fn:          testProtocolCodecNullReasoningBudgetOptIn,
	})
}

func testProtocolCodecNullReasoningBudgetOptIn(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	routerOpts := opts
	routerOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace: ToolResultOptInNamespace, Name: ToolResultOptInRelease, ServicePort: ToolResultOptInPort,
	}
	session, err := fixtures.OpenServiceSession(ctx, client, routerOpts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	sessionID := "null-reasoning-budget-opt-in-" + uuid.NewString()
	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", map[string]any{
		"model":                   nullReasoningBudgetOptInModel,
		"messages":                []map[string]string{{"role": "user", "content": "null reasoning budget probe"}},
		"reasoning_budget_tokens": nil,
	}, false, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return fmt.Errorf("null reasoning budget request: %w", err)
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("null reasoning budget request returned HTTP %d: %s",
			result.StatusCode, truncateString(string(result.Body), 700))
	}

	raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("null reasoning budget provider observation: %w", err)
	}
	var observed struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if decodeErr := json.Unmarshal(raw, &observed); decodeErr != nil {
		return fmt.Errorf("decode null reasoning budget provider request: %w", decodeErr)
	}
	for _, field := range []string{"reasoning_budget_tokens", "thinking_token_budget"} {
		if _, found := observed.Body[field]; found {
			return fmt.Errorf("provider request retained %s: %s", field, truncateString(string(raw), 700))
		}
	}

	rejectedID := "public-thinking-budget-rejection-" + uuid.NewString()
	rejected, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", map[string]any{
		"model":                 nullReasoningBudgetOptInModel,
		"messages":              []map[string]string{{"role": "user", "content": "provider-native budget rejection probe"}},
		"thinking_token_budget": 512,
	}, false, map[string]string{"x-vsr-test-session-id": rejectedID})
	if err != nil {
		return fmt.Errorf("provider-native thinking budget request: %w", err)
	}
	if rejected.StatusCode != http.StatusBadRequest || !strings.Contains(string(rejected.Body), "thinking_token_budget") {
		return fmt.Errorf("public Chat accepted or misreported thinking_token_budget with HTTP %d: %s",
			rejected.StatusCode, truncateString(string(rejected.Body), 700))
	}
	return nil
}
