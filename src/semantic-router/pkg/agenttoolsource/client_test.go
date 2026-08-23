package agenttoolsource

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

type recordingCredentialResolver struct {
	version       string
	secret        []byte
	namespaceID   string
	credentialID  string
	requested     string
	resolvedBytes []byte
}

func (resolver *recordingCredentialResolver) Resolve(
	_ context.Context, namespaceID, credentialID, versionID string,
) (PinnedCredential, error) {
	resolver.namespaceID = namespaceID
	resolver.credentialID = credentialID
	resolver.requested = versionID
	resolver.resolvedBytes = append([]byte(nil), resolver.secret...)
	return PinnedCredential{VersionID: resolver.version, Secret: resolver.resolvedBytes}, nil
}

func TestRemoteToolScrubInputUsesExactPinnedCredentialAndClearsPlaintext(t *testing.T) {
	resolver := &recordingCredentialResolver{
		version: "credential-version-7", secret: []byte("opaque-transport-secret"),
	}
	factory := &ClientFactory{credentials: resolver}
	handler := factory.Handler(agentmanagement.ToolSource{
		ResourceIdentity: agentmanagement.ResourceIdentity{
			NamespaceID: "11111111-1111-4111-8111-111111111111",
			ID:          "22222222-2222-4222-8222-222222222222",
			Status:      agentmanagement.StatusActive,
		},
		CredentialID: "33333333-3333-4333-8333-333333333333",
	}, "credential-version-7", "read").(agentmanagement.ToolInputScrubber)
	clean, err := handler.ScrubInput(
		context.Background(),
		json.RawMessage(`{"first":"opaque-transport-secret","nested":{"second":"xopaque-transport-secretly"}}`),
	)
	if err != nil {
		t.Fatalf("ScrubInput() error = %v", err)
	}
	if bytes.Contains(clean, resolver.secret) {
		t.Fatalf("ScrubInput() retained pinned credential: %s", clean)
	}
	if resolver.namespaceID != "11111111-1111-4111-8111-111111111111" ||
		resolver.credentialID != "33333333-3333-4333-8333-333333333333" ||
		resolver.requested != "credential-version-7" {
		t.Fatalf("credential resolver was not pinned exactly: %#v", resolver)
	}
	if !allZero(resolver.resolvedBytes) {
		t.Fatal("ScrubInput() retained decrypted credential bytes after redaction")
	}
}

func TestRemoteToolScrubInputRejectsCredentialVersionSubstitution(t *testing.T) {
	resolver := &recordingCredentialResolver{
		version: "credential-version-other", secret: []byte("opaque-transport-secret"),
	}
	handler := (&ClientFactory{credentials: resolver}).Handler(agentmanagement.ToolSource{
		ResourceIdentity: agentmanagement.ResourceIdentity{
			NamespaceID: "11111111-1111-4111-8111-111111111111",
			Status:      agentmanagement.StatusActive,
		},
		CredentialID: "33333333-3333-4333-8333-333333333333",
	}, "credential-version-7", "read").(agentmanagement.ToolInputScrubber)
	if _, err := handler.ScrubInput(
		context.Background(), json.RawMessage(`{"value":"opaque-transport-secret"}`),
	); !errors.Is(err, ErrInvocationFailed) {
		t.Fatalf("ScrubInput() substitution error = %v, want ErrInvocationFailed", err)
	}
	if !allZero(resolver.resolvedBytes) {
		t.Fatal("credential substitution failure retained decrypted bytes")
	}
}

func TestRemoteToolResultScrubCoversNestedAndCrossFieldOccurrences(t *testing.T) {
	secret := []byte("remote-result-secret")
	clean, err := scrubRemoteToolPayload(json.RawMessage(`{
  "first":"remote-result-secret",
  "nested":{"second":"before-remote-result-secret-after"},
  "items":[{"third":"remote-result-secret"}]
}`), secret)
	if err != nil {
		t.Fatalf("scrubRemoteToolPayload() error = %v", err)
	}
	if bytes.Contains(clean, secret) {
		t.Fatalf("remote result retained exact credential: %s", clean)
	}
	var value map[string]any
	if err := json.Unmarshal(clean, &value); err != nil {
		t.Fatal(err)
	}
	if value["first"] != "[redacted]" ||
		value["nested"].(map[string]any)["second"] != "before-[redacted]-after" ||
		value["items"].([]any)[0].(map[string]any)["third"] != "[redacted]" {
		t.Fatalf("remote result exact scrub = %#v", value)
	}
}

func allZero(value []byte) bool {
	for _, item := range value {
		if item != 0 {
			return false
		}
	}
	return true
}
