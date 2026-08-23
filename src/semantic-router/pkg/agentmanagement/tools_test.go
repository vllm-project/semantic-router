package agentmanagement

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type exactInputScrubber struct {
	secret []byte
	calls  int
}

func (scrubber *exactInputScrubber) ScrubInput(
	_ context.Context, input json.RawMessage,
) (json.RawMessage, error) {
	scrubber.calls++
	return ScrubToolSecrets(input, scrubber.secret)
}

func (*exactInputScrubber) Invoke(
	context.Context, ToolInvocationContext, json.RawMessage,
) (ToolResult, error) {
	return ToolResult{Value: json.RawMessage(`{"ok":true}`)}, nil
}

type allowToolAuthorizer struct{}

func (allowToolAuthorizer) AuthorizeTool(
	_ context.Context, invocation ToolInvocationContext, _ ToolDefinition,
) (ToolInvocationContext, error) {
	invocation.AuthorityDigest = "sha256:authorized"
	return invocation, nil
}

func TestToolRegistryScrubsExactInputBeforeInvocationPersistence(t *testing.T) {
	handler := &exactInputScrubber{secret: []byte("transport-secret")}
	registry, err := NewToolRegistry([]RegisteredTool{{
		Definition: ToolDefinition{
			Name: "router.test.read", Description: "Read a test resource.",
			InputSchema:         json.RawMessage(`{"type":"object","additionalProperties":true}`),
			OutputSchema:        json.RawMessage(`{"type":"object","additionalProperties":true}`),
			RequiredPermissions: []accesscontrol.Permission{accesscontrol.PermissionToolInvoke},
			Class:               ToolRead, Idempotency: ToolInvocationIdempotent, TimeoutMilliseconds: 1000,
		},
		Handler: handler,
	}}, allowToolAuthorizer{})
	if err != nil {
		t.Fatalf("NewToolRegistry() error = %v", err)
	}
	clean, err := registry.ScrubInvocationInput(
		context.Background(), registry.Revision(),
		ToolPolicy{Allow: []string{"router.test.read"}}, "router.test.read",
		json.RawMessage(`{"note":"before-transport-secret-after"}`),
	)
	if err != nil {
		t.Fatalf("ScrubInvocationInput() error = %v", err)
	}
	if handler.calls != 1 || string(clean) != `{"note":"before-[redacted]-after"}` {
		t.Fatalf("ScrubInvocationInput() = %s after %d scrub calls", clean, handler.calls)
	}
}

func TestToolRegistryDoesNotResolveSecretsForWrongRevisionOrPolicy(t *testing.T) {
	handler := &exactInputScrubber{secret: []byte("transport-secret")}
	registry, err := NewToolRegistry([]RegisteredTool{{
		Definition: ToolDefinition{
			Name: "router.test.read", Description: "Read a test resource.",
			InputSchema:         json.RawMessage(`{"type":"object","additionalProperties":true}`),
			OutputSchema:        json.RawMessage(`{"type":"object","additionalProperties":true}`),
			RequiredPermissions: []accesscontrol.Permission{accesscontrol.PermissionToolInvoke},
			Class:               ToolRead, Idempotency: ToolInvocationIdempotent, TimeoutMilliseconds: 1000,
		}, Handler: handler,
	}}, allowToolAuthorizer{})
	if err != nil {
		t.Fatal(err)
	}
	input := json.RawMessage(`{"note":"transport-secret"}`)
	if _, err := registry.ScrubInvocationInput(
		context.Background(), "sha256:stale", ToolPolicy{Allow: []string{"router.test.read"}},
		"router.test.read", input,
	); err == nil {
		t.Fatal("ScrubInvocationInput() accepted a stale registry revision")
	}
	if _, err := registry.ScrubInvocationInput(
		context.Background(), registry.Revision(), ToolPolicy{Allow: []string{"router.other"}},
		"router.test.read", input,
	); err == nil {
		t.Fatal("ScrubInvocationInput() accepted a denied tool")
	}
	if handler.calls != 0 {
		t.Fatalf("secret scrubber ran %d times before registry/policy validation", handler.calls)
	}
}
