package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-zero-penalty-anthropic", pkgtestcases.TestCase{
		Description: "Explicit zero Chat penalties reach Anthropic Messages without unsupported capability errors",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecZeroPenaltyAnthropic,
	})
}

func testProtocolCodecZeroPenaltyAnthropic(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "anthropic.messages.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	sessionID := "anthropic-zero-penalty-" + uuid.NewString()
	result, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", map[string]any{
		"model": "MoM", "max_tokens": 32, "frequency_penalty": 0, "presence_penalty": 0,
		"messages": []map[string]string{{"role": "user", "content": "Anthropic zero penalty probe"}},
	}, false, map[string]string{"x-vsr-test-session-id": sessionID})
	if requestErr != nil {
		return fmt.Errorf("anthropic zero penalty request: %w", requestErr)
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("anthropic zero penalty returned HTTP %d: %s", result.StatusCode,
			truncateString(string(result.Body), 500))
	}
	upstream, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if observationErr != nil {
		return fmt.Errorf("anthropic zero penalty provider observation: %w", observationErr)
	}
	var observation struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if decodeErr := json.Unmarshal(upstream, &observation); decodeErr != nil {
		return fmt.Errorf("decode Anthropic zero penalty observation: %w", decodeErr)
	}
	if len(observation.Body["messages"]) == 0 {
		return fmt.Errorf("anthropic backend lost zero-penalty request: %s", truncateString(string(upstream), 500))
	}
	for _, field := range []string{"frequency_penalty", "presence_penalty"} {
		if _, leaked := observation.Body[field]; leaked {
			return fmt.Errorf("anthropic backend received unsupported %s: %s", field, truncateString(string(upstream), 500))
		}
	}
	return nil
}
