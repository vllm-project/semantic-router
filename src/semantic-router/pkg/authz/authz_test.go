package authz

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestCredentialsGet(t *testing.T) {
	creds := &Credentials{
		Keys: map[LLMProvider]string{
			ProviderOpenAI:    "sk-openai-123",
			ProviderAnthropic: "sk-ant-456",
		},
	}

	if got := creds.Get(ProviderOpenAI); got != "sk-openai-123" {
		t.Errorf("Get(OpenAI) = %q, want %q", got, "sk-openai-123")
	}
	if got := creds.Get(ProviderAnthropic); got != "sk-ant-456" {
		t.Errorf("Get(Anthropic) = %q, want %q", got, "sk-ant-456")
	}
	if got := creds.Get("gemini"); got != "" {
		t.Errorf("Get(gemini) = %q, want empty", got)
	}

	// Nil credentials
	var nilCreds *Credentials
	if got := nilCreds.Get(ProviderOpenAI); got != "" {
		t.Errorf("nil.Get(OpenAI) = %q, want empty", got)
	}
}

func TestCredentialsIsEmpty(t *testing.T) {
	empty := &Credentials{Keys: map[LLMProvider]string{}}
	if !empty.IsEmpty() {
		t.Error("empty credentials should be empty")
	}

	withKeys := &Credentials{Keys: map[LLMProvider]string{ProviderOpenAI: "key"}}
	if withKeys.IsEmpty() {
		t.Error("credentials with keys should not be empty")
	}

	var nilCreds *Credentials
	if !nilCreds.IsEmpty() {
		t.Error("nil credentials should be empty")
	}
}

// ---- DefaultHeaderMap ----

func TestDefaultHeaderMap(t *testing.T) {
	m := DefaultHeaderMap()

	if m["openai"] != headers.UserOpenAIKey {
		t.Errorf("DefaultHeaderMap[openai] = %q, want %q", m["openai"], headers.UserOpenAIKey)
	}
	if m["anthropic"] != headers.UserAnthropicKey {
		t.Errorf("DefaultHeaderMap[anthropic] = %q, want %q", m["anthropic"], headers.UserAnthropicKey)
	}
	if m["azure-openai"] != headers.UserAzureOpenAIKey {
		t.Errorf("DefaultHeaderMap[azure-openai] = %q, want %q", m["azure-openai"], headers.UserAzureOpenAIKey)
	}
	if m["bedrock"] != headers.UserBedrockKey {
		t.Errorf("DefaultHeaderMap[bedrock] = %q, want %q", m["bedrock"], headers.UserBedrockKey)
	}
	if m["gemini"] != headers.UserGeminiKey {
		t.Errorf("DefaultHeaderMap[gemini] = %q, want %q", m["gemini"], headers.UserGeminiKey)
	}
	if m["vertex-ai"] != headers.UserVertexAIKey {
		t.Errorf("DefaultHeaderMap[vertex-ai] = %q, want %q", m["vertex-ai"], headers.UserVertexAIKey)
	}
	if len(m) != 6 {
		t.Errorf("DefaultHeaderMap has %d entries, want 6", len(m))
	}
}

// ---- HeaderInjectionProvider ----

func TestHeaderInjectionProvider_DefaultHeaders(t *testing.T) {
	p := NewHeaderInjectionProvider(nil)

	if p.Name() != "header-injection" {
		t.Errorf("Name() = %q, want %q", p.Name(), "header-injection")
	}

	reqHeaders := map[string]string{
		headers.UserOpenAIKey:    "sk-injected-openai",
		headers.UserAnthropicKey: "sk-injected-anthropic",
	}

	if got := p.GetKey(ProviderOpenAI, "gpt-4", reqHeaders); got != "sk-injected-openai" {
		t.Errorf("GetKey(OpenAI) = %q, want %q", got, "sk-injected-openai")
	}
	if got := p.GetKey(ProviderAnthropic, "claude-3", reqHeaders); got != "sk-injected-anthropic" {
		t.Errorf("GetKey(Anthropic) = %q, want %q", got, "sk-injected-anthropic")
	}

	if got := p.GetKey("gemini", "gemini-pro", reqHeaders); got != "" {
		t.Errorf("GetKey(gemini) = %q, want empty", got)
	}

	if got := p.GetKey(ProviderOpenAI, "gpt-4", map[string]string{}); got != "" {
		t.Errorf("GetKey with empty headers = %q, want empty", got)
	}

	strip := p.HeadersToStrip()
	if len(strip) < 2 {
		t.Errorf("HeadersToStrip() returned %d headers, want at least 2", len(strip))
	}
	found := map[string]bool{}
	for _, h := range strip {
		found[h] = true
	}
	if !found[headers.UserOpenAIKey] {
		t.Errorf("HeadersToStrip missing %s", headers.UserOpenAIKey)
	}
	if !found[headers.UserAnthropicKey] {
		t.Errorf("HeadersToStrip missing %s", headers.UserAnthropicKey)
	}
}

func TestHeaderInjectionProvider_CustomHeaders(t *testing.T) {
	customMap := map[string]string{
		"openai":    "x-custom-openai-token",
		"anthropic": "x-custom-anthropic-token",
		"gemini":    "x-custom-gemini-token",
	}
	p := NewHeaderInjectionProvider(customMap)

	reqHeaders := map[string]string{
		"x-custom-openai-token":    "sk-custom-openai",
		"x-custom-anthropic-token": "sk-custom-anthropic",
		"x-custom-gemini-token":    "sk-custom-gemini",
	}

	if got := p.GetKey(ProviderOpenAI, "gpt-4", reqHeaders); got != "sk-custom-openai" {
		t.Errorf("GetKey(OpenAI) = %q, want %q", got, "sk-custom-openai")
	}
	if got := p.GetKey(ProviderAnthropic, "claude-3", reqHeaders); got != "sk-custom-anthropic" {
		t.Errorf("GetKey(Anthropic) = %q, want %q", got, "sk-custom-anthropic")
	}
	if got := p.GetKey("gemini", "gemini-pro", reqHeaders); got != "sk-custom-gemini" {
		t.Errorf("GetKey(gemini) = %q, want %q", got, "sk-custom-gemini")
	}

	strip := p.HeadersToStrip()
	if len(strip) != 3 {
		t.Errorf("HeadersToStrip() returned %d headers, want 3", len(strip))
	}
	found := map[string]bool{}
	for _, h := range strip {
		found[h] = true
	}
	for _, expected := range []string{"x-custom-openai-token", "x-custom-anthropic-token", "x-custom-gemini-token"} {
		if !found[expected] {
			t.Errorf("HeadersToStrip missing %s", expected)
		}
	}
}

func TestHeaderInjectionProvider_EmptyMap(t *testing.T) {
	p := NewHeaderInjectionProvider(map[string]string{})

	reqHeaders := map[string]string{
		headers.UserOpenAIKey: "sk-default",
	}
	if got := p.GetKey(ProviderOpenAI, "gpt-4", reqHeaders); got != "sk-default" {
		t.Errorf("GetKey(OpenAI) with empty map = %q, want %q", got, "sk-default")
	}
}

func TestHeaderInjectionProvider_GeminiViaConfig(t *testing.T) {
	geminiMap := map[string]string{
		"openai":    "x-user-openai-key",
		"anthropic": "x-user-anthropic-key",
		"gemini":    "x-user-gemini-key",
	}
	p := NewHeaderInjectionProvider(geminiMap)

	reqHeaders := map[string]string{
		"x-user-openai-key":    "sk-openai",
		"x-user-anthropic-key": "sk-anthropic",
		"x-user-gemini-key":    "AIza-gemini",
	}

	tests := []struct {
		provider LLMProvider
		want     string
	}{
		{ProviderOpenAI, "sk-openai"},
		{ProviderAnthropic, "sk-anthropic"},
		{"gemini", "AIza-gemini"},
	}
	for _, tc := range tests {
		if got := p.GetKey(tc.provider, "any-model", reqHeaders); got != tc.want {
			t.Errorf("GetKey(%s) = %q, want %q", tc.provider, got, tc.want)
		}
	}

	strip := p.HeadersToStrip()
	if len(strip) != 3 {
		t.Errorf("HeadersToStrip() returned %d headers, want 3", len(strip))
	}
}

// ---- StaticConfigProvider ----

func TestStaticConfigProvider(t *testing.T) {
	p := NewStaticConfigProvider(nil)

	if p.Name() != "static-config" {
		t.Errorf("Name() = %q, want %q", p.Name(), "static-config")
	}

	if got := p.GetKey(ProviderOpenAI, "gpt-4", nil); got != "" {
		t.Errorf("GetKey with nil config = %q, want empty", got)
	}

	if strip := p.HeadersToStrip(); strip != nil {
		t.Errorf("HeadersToStrip() = %v, want nil", strip)
	}
}

// ===========================================================================
// CredentialResolver — fail-closed (default)
// ===========================================================================

func TestCredentialResolver_FailClosed_ReturnsError(t *testing.T) {
	// Default resolver is fail-closed: when no provider resolves a key, return error
	headerProvider := NewHeaderInjectionProvider(nil)
	staticProvider := NewStaticConfigProvider(nil)
	resolver := NewCredentialResolver(headerProvider, staticProvider)

	// No headers, no static config → should error
	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", map[string]string{})
	if err == nil {
		t.Fatal("expected error when no key found with fail-closed, got nil")
	}
	if key != "" {
		t.Errorf("expected empty key on error, got %q", key)
	}
	// Error message should be actionable — mention provider tried
	if err.Error() == "" {
		t.Error("error message should not be empty")
	}
}

func TestCredentialResolver_FailClosed_SuccessPath(t *testing.T) {
	// When a key IS found, no error even in fail-closed mode
	headerProvider := NewHeaderInjectionProvider(nil)
	resolver := NewCredentialResolver(headerProvider)

	reqHeaders := map[string]string{
		headers.UserOpenAIKey: "sk-from-header",
	}
	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", reqHeaders)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if key != "sk-from-header" {
		t.Errorf("KeyForProvider = %q, want %q", key, "sk-from-header")
	}
}

// ===========================================================================
// CredentialResolver — fail-open
// ===========================================================================

func TestCredentialResolver_FailOpen_ReturnsNilError(t *testing.T) {
	// fail-open: when no provider resolves a key, return ("", nil) — no error
	headerProvider := NewHeaderInjectionProvider(nil)
	staticProvider := NewStaticConfigProvider(nil)
	resolver := NewCredentialResolver(headerProvider, staticProvider)
	resolver.SetFailOpen(true)

	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", map[string]string{})
	if err != nil {
		t.Fatalf("fail-open should not return error, got: %v", err)
	}
	if key != "" {
		t.Errorf("expected empty key in fail-open miss, got %q", key)
	}
}

func TestCredentialResolver_FailOpen_SuccessPath(t *testing.T) {
	// When a key IS found, same behavior regardless of fail-open
	headerProvider := NewHeaderInjectionProvider(nil)
	resolver := NewCredentialResolver(headerProvider)
	resolver.SetFailOpen(true)

	reqHeaders := map[string]string{
		headers.UserOpenAIKey: "sk-from-header",
	}
	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", reqHeaders)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if key != "sk-from-header" {
		t.Errorf("KeyForProvider = %q, want %q", key, "sk-from-header")
	}
}

// ===========================================================================
// CredentialResolver — nil safety
// ===========================================================================

func TestCredentialResolverNil(t *testing.T) {
	var resolver *CredentialResolver

	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", nil)
	if err == nil {
		t.Fatal("nil resolver should return error")
	}
	if key != "" {
		t.Errorf("nil resolver key = %q, want empty", key)
	}

	if got := resolver.HeadersToStrip(); got != nil {
		t.Errorf("nil resolver HeadersToStrip = %v, want nil", got)
	}
	if got := resolver.ProviderNames(); got != nil {
		t.Errorf("nil resolver ProviderNames = %v, want nil", got)
	}
	if resolver.FailOpen() {
		t.Error("nil resolver FailOpen should be false")
	}
}

// ===========================================================================
// CredentialResolver — chain metadata
// ===========================================================================

func TestCredentialResolverChainMetadata(t *testing.T) {
	headerProvider := NewHeaderInjectionProvider(nil)
	staticProvider := NewStaticConfigProvider(nil)
	resolver := NewCredentialResolver(headerProvider, staticProvider)

	names := resolver.ProviderNames()
	if len(names) != 2 || names[0] != "header-injection" || names[1] != "static-config" {
		t.Errorf("ProviderNames() = %v, want [header-injection static-config]", names)
	}

	strip := resolver.HeadersToStrip()
	if len(strip) < 2 {
		t.Errorf("HeadersToStrip() returned %d headers, want at least 2", len(strip))
	}

	if resolver.FailOpen() {
		t.Error("default FailOpen should be false")
	}
	resolver.SetFailOpen(true)
	if !resolver.FailOpen() {
		t.Error("FailOpen should be true after SetFailOpen(true)")
	}
}

// ===========================================================================
// Custom chain from YAML — simulates config-driven setup
// ===========================================================================

func TestCredentialResolverCustomChain_FailClosed(t *testing.T) {
	customHeaders := map[string]string{
		"openai":    "x-eg-openai-key",
		"anthropic": "x-eg-anthropic-key",
	}
	headerProvider := NewHeaderInjectionProvider(customHeaders)
	staticProvider := NewStaticConfigProvider(nil)
	resolver := NewCredentialResolver(headerProvider, staticProvider)

	// Custom header resolves → no error
	reqHeaders := map[string]string{
		"x-eg-openai-key": "sk-from-envoy-gateway",
	}
	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", reqHeaders)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if key != "sk-from-envoy-gateway" {
		t.Errorf("key = %q, want %q", key, "sk-from-envoy-gateway")
	}

	// Standard header does NOT resolve (not configured), static nil → error (fail-closed)
	reqHeaders2 := map[string]string{
		headers.UserOpenAIKey: "sk-wrong-header",
	}
	_, err = resolver.KeyForProvider(ProviderOpenAI, "gpt-4", reqHeaders2)
	if err == nil {
		t.Fatal("expected error when custom headers don't match and static is nil")
	}

	// Strips only the custom headers
	strip := resolver.HeadersToStrip()
	found := map[string]bool{}
	for _, h := range strip {
		found[h] = true
	}
	if !found["x-eg-openai-key"] {
		t.Errorf("HeadersToStrip missing x-eg-openai-key")
	}
	if !found["x-eg-anthropic-key"] {
		t.Errorf("HeadersToStrip missing x-eg-anthropic-key")
	}
	if found[headers.UserOpenAIKey] {
		t.Errorf("HeadersToStrip should not include default header %s", headers.UserOpenAIKey)
	}
}

func TestCredentialResolverStaticOnly(t *testing.T) {
	resolver := NewCredentialResolver(NewStaticConfigProvider(nil))

	names := resolver.ProviderNames()
	if len(names) != 1 || names[0] != "static-config" {
		t.Errorf("ProviderNames() = %v, want [static-config]", names)
	}

	strip := resolver.HeadersToStrip()
	if len(strip) != 0 {
		t.Errorf("HeadersToStrip() = %v, want empty", strip)
	}

	// Static with nil config → error in fail-closed mode
	_, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", nil)
	if err == nil {
		t.Fatal("expected error from static-only with nil config in fail-closed mode")
	}
}

// ===========================================================================
// Priority
// ===========================================================================

type mockProvider struct {
	name string
	keys map[LLMProvider]string
}

func (m *mockProvider) Name() string { return m.name }
func (m *mockProvider) GetKey(provider LLMProvider, _ string, _ map[string]string) string {
	return m.keys[provider]
}
func (m *mockProvider) HeadersToStrip() []string { return nil }

func TestCredentialResolverPriority(t *testing.T) {
	providerA := &mockProvider{name: "A", keys: map[LLMProvider]string{ProviderOpenAI: "key-from-A"}}
	providerB := &mockProvider{name: "B", keys: map[LLMProvider]string{ProviderOpenAI: "key-from-B", ProviderAnthropic: "ant-from-B"}}

	resolver := NewCredentialResolver(providerA, providerB)

	// OpenAI: A wins (first in chain)
	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if key != "key-from-A" {
		t.Errorf("OpenAI key = %q, want %q", key, "key-from-A")
	}

	// Anthropic: B wins (A doesn't have it)
	key, err = resolver.KeyForProvider(ProviderAnthropic, "claude-3", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if key != "ant-from-B" {
		t.Errorf("Anthropic key = %q, want %q", key, "ant-from-B")
	}
}

// ===========================================================================
// Misconfig detection: header name typo → fail-closed catches it
// ===========================================================================

func TestMisconfig_HeaderNameTypo_FailClosed(t *testing.T) {
	// Simulate: YAML config has a typo in the header name
	// The operator thinks they configured "x-user-openai-key" but actually wrote "x-user-opanai-key"
	typoHeaders := map[string]string{
		"openai":    "x-user-opanai-key", // typo!
		"anthropic": "x-user-anthropic-key",
	}
	headerProvider := NewHeaderInjectionProvider(typoHeaders)
	resolver := NewCredentialResolver(headerProvider)

	// ext_authz correctly injects the REAL header name
	reqHeaders := map[string]string{
		"x-user-openai-key":    "sk-real-key",
		"x-user-anthropic-key": "sk-real-anthropic",
	}

	// OpenAI: typo means header-injection misses, fail-closed → error
	_, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", reqHeaders)
	if err == nil {
		t.Fatal("expected error: typo in header name should cause fail-closed rejection, not silent bypass")
	}

	// Anthropic: correct header name → succeeds
	key, err := resolver.KeyForProvider(ProviderAnthropic, "claude-3", reqHeaders)
	if err != nil {
		t.Fatalf("unexpected error for correct header: %v", err)
	}
	if key != "sk-real-anthropic" {
		t.Errorf("key = %q, want %q", key, "sk-real-anthropic")
	}
}

func TestMisconfig_HeaderNameTypo_FailOpen(t *testing.T) {
	// Same typo but with fail-open: no error, but the request goes through without a key
	typoHeaders := map[string]string{
		"openai": "x-user-opanai-key", // typo!
	}
	resolver := NewCredentialResolver(NewHeaderInjectionProvider(typoHeaders))
	resolver.SetFailOpen(true)

	reqHeaders := map[string]string{
		"x-user-openai-key": "sk-real-key",
	}

	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", reqHeaders)
	if err != nil {
		t.Fatalf("fail-open should not error: %v", err)
	}
	if key != "" {
		t.Errorf("expected empty key (typo means miss), got %q", key)
	}
	// This is the dangerous case: fail-open silently allows a request without a key.
	// The WARN log (tested via integration) alerts operators.
}

// ===========================================================================
// Misconfig detection: header spoofing without ext_authz
// ===========================================================================

func TestMisconfig_HeaderSpoofing_FailClosed(t *testing.T) {
	// Simulate: header-injection is configured but ext_authz is NOT running.
	// A malicious client sets x-user-openai-key directly.
	// In a properly configured Envoy, ext_authz would strip/overwrite this header.
	// But if ext_authz is absent, the client header reaches the router.
	//
	// This test verifies the happy path (client-sent header is accepted).
	// The protection against spoofing is at the Envoy layer (ext_authz filter),
	// not the router layer. The router trusts what Envoy sends it.
	resolver := NewCredentialResolver(NewHeaderInjectionProvider(nil))

	spoofedHeaders := map[string]string{
		headers.UserOpenAIKey: "sk-spoofed-key",
	}

	key, err := resolver.KeyForProvider(ProviderOpenAI, "gpt-4", spoofedHeaders)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if key != "sk-spoofed-key" {
		t.Errorf("key = %q, want %q", key, "sk-spoofed-key")
	}
	// NOTE: Spoofing protection must come from Envoy's ext_authz filter, not the router.
	// The startup log should warn operators to ensure ext_authz is configured in Envoy
	// when header-injection is active in the router.
}
