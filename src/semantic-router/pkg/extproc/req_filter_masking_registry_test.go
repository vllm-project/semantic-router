package extproc

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func maskingRegistryConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	dir := t.TempDir()
	key := filepath.Join(dir, "key")
	if err := os.WriteFile(key, bytes.Repeat([]byte{7}, 32), 0o600); err != nil {
		t.Fatal(err)
	}
	return &config.RouterConfig{MaskingIdentity: config.MaskingIdentityConfig{Namespaces: map[string]config.MaskingIdentityNamespace{
		"local": {
			KeyFile:  key,
			Scope:    config.MaskingIdentityScope,
			Registry: config.MaskingRegistryConfig{Backend: "file", Path: filepath.Join(dir, "registry.log")},
		},
	}}}
}

func maskingRegistryRouter(t *testing.T, cfg *config.RouterConfig) *OpenAIRouter {
	t.Helper()
	identities, err := openMaskingIdentities(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &OpenAIRouter{Config: cfg, maskingIdentities: identities}
}

func registryMaskingDecision() *config.Decision {
	return &config.Decision{Plugins: []config.DecisionPlugin{{
		Type: config.DecisionPluginMasking,
		Configuration: config.MustStructuredPayload(map[string]interface{}{
			"enabled": true, "signal": maskingRule, "identity_namespace": "local",
		}),
	}}}
}

func dispatchRegistryMasking(t *testing.T, router *OpenAIRouter, principal, text string) string {
	t.Helper()
	body := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"` + text + `"}]}`
	ctx := newMaskingContextOf(t, router, llmprotocol.OpenAIChatV1, registryMaskingDecision(), body)
	ctx.AuthenticatedPrincipal = principal
	dispatched, err := finalizeMaskingDispatch(router, ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, dispatched)
	if err != nil {
		t.Fatalf("dispatched bytes are not a Chat request: %v", err)
	}
	return request.Messages[0].Content[0].Text
}

// A value keeps its placeholder in a later request, also when an echo of the
// placeholder comes before the value, as in a compacted history, and another
// principal's values are numbered apart.
func TestMaskingRegistryKeepsPlaceholdersAcrossRequests(t *testing.T) {
	router := maskingRegistryRouter(t, maskingRegistryConfig(t))
	useMaskingDetector(t, &stubPIIDetector{values: []string{"alice@example.com", "bob@example.com"}})

	if got := dispatchRegistryMasking(t, router, "", "write alice@example.com"); got != "write [EMAIL_ADDRESS_0]" {
		t.Fatalf("first request = %q", got)
	}
	got := dispatchRegistryMasking(t, router, "", "earlier [EMAIL_ADDRESS_0], now bob@example.com and alice@example.com")
	if want := "earlier [EMAIL_ADDRESS_0], now [EMAIL_ADDRESS_1] and [EMAIL_ADDRESS_0]"; got != want {
		t.Fatalf("second request = %q, want %q", got, want)
	}
	if got := dispatchRegistryMasking(t, router, "user-b", "bob@example.com"); got != "[EMAIL_ADDRESS_0]" {
		t.Fatalf("another principal = %q", got)
	}
	if got := dispatchRegistryMasking(t, router, "", "bob@example.com"); got != "[EMAIL_ADDRESS_1]" {
		t.Fatalf("installation scope after another principal = %q", got)
	}
}

func TestMaskingRegistryFailsClosedWithoutNamespace(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	useMaskingDetector(t, &stubPIIDetector{values: []string{maskingRawEmail}})
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, registryMaskingDecision())
	generation := ctx.SemanticRequest.Generation
	_, err := finalizeMaskingDispatch(router, ctx)
	assertMaskingFailsClosed(t, router, ctx, err, generation)
}

func TestOpenMaskingIdentitiesRefusesAMissingKey(t *testing.T) {
	cfg := maskingRegistryConfig(t)
	ns := cfg.MaskingIdentity.Namespaces["local"]
	if err := os.Remove(ns.KeyFile); err != nil {
		t.Fatal(err)
	}
	if _, err := openMaskingIdentities(cfg); err == nil || !strings.Contains(err.Error(), "namespaces.local") {
		t.Fatalf("open = %v", err)
	}
}
