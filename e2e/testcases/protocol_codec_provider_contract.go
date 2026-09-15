package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func openProtocolCodecProviderSession(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	backendFormat string,
) (*fixtures.ServiceSession, error) {
	backendOpts := opts
	switch backendFormat {
	case "openai.chat.v1", "openai.responses.v1":
		backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
			Namespace: "default", Name: "mock-vllm", ServicePort: "8000",
		}
	case "anthropic.messages.v1":
		backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
			Namespace: "anthropic-backend-system", Name: "anthropic-backend-qwen", ServicePort: "8080",
		}
	default:
		return nil, fmt.Errorf("provider simulator is not registered for backend format %q", backendFormat)
	}
	return fixtures.OpenServiceSession(ctx, client, backendOpts)
}

func verifyProviderSimulatorRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	sessionID string,
	backendFormat string,
	prompt string,
) error {
	body, err := lastProviderSimulatorRequest(ctx, session, sessionID)
	if err != nil {
		return err
	}
	var debug struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(body, &debug); err != nil {
		return fmt.Errorf("decode provider simulator request: %w", err)
	}
	if len(debug.Body) == 0 {
		return fmt.Errorf("provider simulator recorded an empty request")
	}
	if !strings.Contains(string(body), prompt) {
		return fmt.Errorf("provider request lost the matrix marker: %s", truncateString(string(body), 600))
	}

	required, forbidden := providerContractFields(backendFormat)
	for _, field := range required {
		if _, found := debug.Body[field]; !found {
			return fmt.Errorf("%s provider request is missing %q: %s", backendFormat, field, truncateString(string(body), 600))
		}
	}
	for _, field := range forbidden {
		if _, found := debug.Body[field]; found {
			return fmt.Errorf("%s provider request leaked %q: %s", backendFormat, field, truncateString(string(body), 600))
		}
	}
	return nil
}

func providerContractFields(backendFormat string) (required, forbidden []string) {
	switch backendFormat {
	case "openai.chat.v1":
		return []string{"messages", "model"}, []string{"input"}
	case "openai.responses.v1":
		return []string{"input", "model"}, []string{"messages", "max_tokens"}
	case "anthropic.messages.v1":
		return []string{"max_tokens", "messages", "model"}, []string{"input"}
	default:
		return nil, nil
	}
}

func protocolMatrixSessionID(backendFormat, clientName, mode string) string {
	backend := strings.NewReplacer(".", "-", "_", "-").Replace(backendFormat)
	return fmt.Sprintf("protocol-matrix-%s-%s-%s", backend, clientName, mode)
}
