package backendcredential

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolverUsesNamedEnvironmentReference(t *testing.T) {
	t.Setenv("TEST_BACKEND_CREDENTIAL", "  provider-secret  ")
	resolver, err := NewResolver(map[string]config.BackendCredentialConfig{
		"private": {CredentialAdapterID: "bearer", SecretEnv: "TEST_BACKEND_CREDENTIAL"},
	})
	if err != nil {
		t.Fatal(err)
	}
	publication := backendinvoker.CredentialPublication{
		NamespaceID: "ns", QuotaPartition: "ns", PublicationID: "publication",
	}
	version, err := resolver.Pin(context.Background(), publication, "private", "provider", "https://models.example")
	if err != nil {
		t.Fatal(err)
	}
	credential, err := resolver.ResolvePinned(context.Background(), publication, "private", version, "provider", "https://models.example")
	if err != nil || credential.Secret != "provider-secret" || credential.Header != "Authorization" || credential.Prefix != "Bearer " {
		t.Fatalf("ResolvePinned() = %+v, %v", credential, err)
	}
}

func TestResolverDiagnosticsNeverRenderSecretValues(t *testing.T) {
	const canary = "backend-secret-diagnostics-canary"
	t.Setenv("TEST_BACKEND_DIAGNOSTIC_CREDENTIAL", canary)
	resolver, err := NewResolver(map[string]config.BackendCredentialConfig{
		"private": {CredentialAdapterID: "bearer", SecretEnv: "TEST_BACKEND_DIAGNOSTIC_CREDENTIAL"},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, rendered := range []string{fmt.Sprintf("%v", resolver), fmt.Sprintf("%#v", resolver)} {
		if strings.Contains(rendered, canary) {
			t.Fatalf("resolver diagnostics exposed a credential: %s", rendered)
		}
	}
}

func TestResolvedSecretNeverEntersCanonicalConfigExport(t *testing.T) {
	const canary = "backend-secret-export-canary"
	t.Setenv("TEST_BACKEND_EXPORT_CREDENTIAL", canary)
	cfg := &config.RouterConfig{
		BackendCredentials: config.BackendCredentialsConfig{File: map[string]config.BackendCredentialConfig{
			"private": {CredentialAdapterID: "bearer", SecretEnv: "TEST_BACKEND_EXPORT_CREDENTIAL"},
		}},
	}
	if _, err := NewResolver(cfg.BackendCredentials.File); err != nil {
		t.Fatal(err)
	}
	exported, err := yaml.Marshal(config.CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	text := string(exported)
	if strings.Contains(text, canary) {
		t.Fatal("canonical config export exposed a resolved backend credential")
	}
	if !strings.Contains(text, "secret_env: TEST_BACKEND_EXPORT_CREDENTIAL") {
		t.Fatalf("canonical config export lost the operator-owned reference:\n%s", text)
	}
}

func TestResolverMaterializesRedactedInMemoryLiteral(t *testing.T) {
	const canary = "literal-provider-secret-canary"
	definition := config.BackendCredentialConfig{
		CredentialAdapterID: "x-api-key", SecretValue: canary,
	}
	for _, rendered := range []string{fmt.Sprintf("%v", definition), fmt.Sprintf("%#v", definition)} {
		if strings.Contains(rendered, canary) {
			t.Fatalf("credential diagnostics exposed a literal secret: %s", rendered)
		}
	}
	resolver, err := NewResolver(map[string]config.BackendCredentialConfig{"private": definition})
	if err != nil {
		t.Fatal(err)
	}
	publication := backendinvoker.CredentialPublication{
		NamespaceID: "ns", QuotaPartition: "ns", PublicationID: "publication",
	}
	version, err := resolver.Pin(context.Background(), publication, "private", "provider", "https://models.example")
	if err != nil {
		t.Fatal(err)
	}
	credential, err := resolver.ResolvePinned(
		context.Background(), publication, "private", version, "provider", "https://models.example",
	)
	if err != nil || credential.Secret != canary || credential.Header != "X-Api-Key" {
		t.Fatalf("ResolvePinned() = %+v, %v", credential, err)
	}
}

func TestResolverRejectsMissingAndEmptySecrets(t *testing.T) {
	if _, err := NewResolver(map[string]config.BackendCredentialConfig{
		"missing": {CredentialAdapterID: "bearer", SecretEnv: "TEST_MISSING_BACKEND_CREDENTIAL"},
	}); err == nil {
		t.Fatal("expected missing environment variable to fail startup")
	}
	t.Setenv("TEST_EMPTY_BACKEND_CREDENTIAL", "  \n")
	if _, err := NewResolver(map[string]config.BackendCredentialConfig{
		"empty": {CredentialAdapterID: "bearer", SecretEnv: "TEST_EMPTY_BACKEND_CREDENTIAL"},
	}); err == nil {
		t.Fatal("expected empty secret to fail startup")
	}
}

func TestCredentialHeadersAreAlwaysStripped(t *testing.T) {
	joined := strings.Join(HeadersToStrip(), ",")
	for _, name := range []string{
		"authorization", "proxy-authorization", "cookie", "set-cookie",
		"x-api-key", "api-key", "x-goog-api-key", "x-amz-security-token",
	} {
		if !strings.Contains(joined, name) {
			t.Fatalf("missing sensitive header %q from %q", name, joined)
		}
	}
}
