package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-later-instructions", pkgtestcases.TestCase{
		Description: "A later system or developer message from a Chat or Responses client reaches Chat and Responses backends where the client put it",
		Tags:        []string{"protocol-codec", "response-api", "agents"},
		Fn:          testProtocolCodecLaterInstructions,
	})
	pkgtestcases.Register("protocol-codec-later-instructions-opt-in", pkgtestcases.TestCase{
		Description: "A later system or developer message from a Chat or Responses client reaches Chat and Responses backends as user text with a protocol warning when the operator accepts that loss",
		Tags:        []string{"protocol-codec", "response-api", "agents"},
		Fn:          testProtocolCodecLaterInstructionsOptIn,
	})
}

type laterInstructionBackend struct {
	model, format, reply string
	userText             bool
}

const laterInstructionText = "Answer in French from now on."

// laterInstructionTurns opens with a system message, which leads the request,
// and changes the instruction after the first exchange.
func laterInstructionTurns(role string) []map[string]any {
	return []map[string]any{
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "First turn"},
		{"role": "assistant", "content": "First reply"},
		{"role": role, "content": laterInstructionText},
		{"role": "user", "content": "Later instruction probe __mock_protocol_matrix__"},
	}
}

var laterInstructionClients = []struct {
	path   string
	body   func(model, role string) map[string]any
	assert func(body []byte, reply string) error
}{
	{"/v1/chat/completions", func(model, role string) map[string]any {
		return map[string]any{"model": model, "max_tokens": 64, "messages": laterInstructionTurns(role)}
	}, assertChatCompletionBody},
	{"/v1/responses", func(model, role string) map[string]any {
		return map[string]any{"model": model, "max_output_tokens": 64, "store": false, "input": laterInstructionTurns(role)}
	}, assertResponsesBody},
}

func testProtocolCodecLaterInstructions(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runProtocolCodecLaterInstructions(ctx, client, opts, opts, []laterInstructionBackend{
		{model: chatBackendModel, format: "openai.chat.v1", reply: protocolCodecChatReply},
		{model: nativeResponsesBackendModel, format: "openai.responses.v1", reply: protocolCodecResponsesReply},
	})
}

func testProtocolCodecLaterInstructionsOptIn(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	routerOpts := opts
	routerOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace: ToolResultOptInNamespace, Name: ToolResultOptInRelease, ServicePort: ToolResultOptInPort,
	}
	return runProtocolCodecLaterInstructions(ctx, client, opts, routerOpts, []laterInstructionBackend{
		{model: "mock/opt-in-chat", format: "openai.chat.v1", reply: protocolCodecChatReply, userText: true},
		{model: "mock/opt-in-responses", format: "openai.responses.v1", reply: protocolCodecResponsesReply, userText: true},
	})
}

func runProtocolCodecLaterInstructions(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts, routerOpts pkgtestcases.TestCaseOptions,
	backends []laterInstructionBackend,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, routerOpts)
	if err != nil {
		return err
	}
	defer session.Close()
	for _, clientProtocol := range laterInstructionClients {
		for _, check := range backends {
			for _, role := range []string{"system", "developer"} {
				if checkErr := checkLaterInstruction(ctx, client, opts, session, clientProtocol.path,
					clientProtocol.body(check.model, role), clientProtocol.assert, check, role); checkErr != nil {
					return fmt.Errorf("%s to %s (%s): %w", clientProtocol.path, check.format, role, checkErr)
				}
			}
		}
	}
	return nil
}

func checkLaterInstruction(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	session *fixtures.ServiceSession,
	path string,
	body map[string]any,
	assertResponse func([]byte, string) error,
	check laterInstructionBackend,
	role string,
) error {
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, check.format)
	if err != nil {
		return err
	}
	defer provider.Close()
	sessionID := "later-instructions-" + uuid.NewString()
	result, err := sendProtocolMatrixRaw(ctx, session, path, body, false, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return err
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", result.StatusCode, truncateString(string(result.Body), 500))
	}
	if responseErr := assertResponse(result.Body, check.reply); responseErr != nil {
		return fmt.Errorf("client response: %w", responseErr)
	}
	upstream, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("provider observation: %w", err)
	}
	if placementErr := assertLaterInstructionPlacement(upstream, check.format, role, check.userText); placementErr != nil {
		return placementErr
	}
	if moved := hasProtocolFieldDiagnostic(result.Headers.Get(warningsHeader), "approximated", "messages[].role"); moved != check.userText {
		return fmt.Errorf("later instruction as user text warning = %t", moved)
	}
	return nil
}

// assertLaterInstructionPlacement checks that the leading system message still
// leads and the later instruction stays after the first exchange with its
// original role or, when the Router sends it as user text, as a user turn.
func assertLaterInstructionPlacement(raw []byte, format, expectedRole string, userText bool) error {
	var observation struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return err
	}
	key := "messages"
	if format == "openai.responses.v1" {
		key = "input"
	}
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(observation.Body[key], &items); err != nil {
		return fmt.Errorf("decode provider %s: %w", key, err)
	}
	roles := make([]string, 0, len(items))
	for _, item := range items {
		var role string
		_ = json.Unmarshal(item["role"], &role)
		roles = append(roles, role)
	}
	want := []string{"system", "user", "assistant", expectedRole, "user"}
	if userText {
		want = []string{"system", "user", "assistant", "user", "user"}
	}
	if !reflect.DeepEqual(roles, want) || !bytes.Contains(items[3]["content"], []byte(laterInstructionText)) {
		return fmt.Errorf("backend roles %v, want %v: %s", roles, want, truncateString(string(raw), 800))
	}
	return nil
}
