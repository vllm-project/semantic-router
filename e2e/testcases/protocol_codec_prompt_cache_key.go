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
	pkgtestcases.Register("protocol-codec-prompt-cache-key-anthropic", pkgtestcases.TestCase{
		Description: "Chat and Responses prompt_cache_key is omitted with a warning on buffered and streaming Messages dispatch",
		Tags:        []string{"protocol-codec", "anthropic", "response-api", "agents", "streaming"},
		Fn:          testProtocolCodecPromptCacheKeyAnthropic,
	})
}

func testProtocolCodecPromptCacheKeyAnthropic(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	for _, api := range protocolCodecE2EClients {
		if api.path == "/v1/messages" {
			continue
		}
		for _, stream := range []bool{false, true} {
			marker := protocolCodecAnthropicProbe + " prompt_cache_key " + api.name
			sessionID := "prompt-cache-key-" + uuid.NewString()
			body := api.request("MoM", marker, stream)
			body["prompt_cache_key"] = sessionID
			result, requestErr := sendProtocolMatrixRaw(ctx, session, api.path, body, stream,
				map[string]string{"x-vsr-test-session-id": sessionID})
			if requestErr != nil {
				return fmt.Errorf("%s stream=%t request: %w", api.name, stream, requestErr)
			}
			if result.StatusCode != http.StatusOK {
				return fmt.Errorf("%s stream=%t HTTP %d: %s", api.name, stream, result.StatusCode, truncateString(string(result.Body), 500))
			}
			if stream {
				err = api.validateStream(result.Body, protocolCodecAnthropicReply)
			} else {
				err = api.validateBuffered(result.Body, protocolCodecAnthropicReply)
			}
			if err != nil {
				return fmt.Errorf("%s stream=%t response: %w", api.name, stream, err)
			}
			if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !hasProtocolFieldDiagnostic(warnings, "dropped", "prompt_cache_key") {
				return fmt.Errorf("%s stream=%t omitted prompt_cache_key without a dropped warning: %q", api.name, stream, warnings)
			}
			if err := verifyProviderSimulatorRequest(ctx, provider, sessionID, "anthropic.messages.v1", marker); err != nil {
				return fmt.Errorf("%s stream=%t provider request: %w", api.name, stream, err)
			}
			raw, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
			if observationErr != nil {
				return observationErr
			}
			var receipt struct {
				Body map[string]json.RawMessage `json:"body"`
			}
			if err := json.Unmarshal(raw, &receipt); err != nil {
				return err
			}
			if _, forwarded := receipt.Body["prompt_cache_key"]; forwarded {
				return fmt.Errorf("%s stream=%t leaked prompt_cache_key to Messages: %s", api.name, stream, truncateString(string(raw), 500))
			}
		}
	}
	return nil
}
