package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func delegatedBackendConfig(backendRefs string, enabled bool) string {
	contract := ""
	if enabled {
		contract = `
global:
  services:
    authz:
      delegated_credential:
        enabled: true
`
	}
	return `
version: v0.5
providers:
  models:
    - name: gateway-model
      provider_model_id: gateway-model
      api_format: openai
      backend_refs:
` + backendRefs + `
recipes:
  - name: default
` + contract
}

const delegatedGatewayBackend = `
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          auth:
            credential: delegated
`

func TestDelegatedBackendMaterializesWithoutAStaticKey(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(delegatedBackendConfig(delegatedGatewayBackend, true)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	profile, err := cfg.GetProviderProfileForEndpoint("gateway-model_gateway")
	if err != nil {
		t.Fatal(err)
	}
	if !profile.Delegated() {
		t.Fatalf("profile = %+v, want a delegated backend", profile)
	}
	if key := cfg.GetModelAccessKey("gateway-model"); key != "" {
		t.Fatalf("a delegated backend has no static key, got %q", key)
	}
	if !cfg.Authz.DelegatedCredential.Enabled || cfg.Authz.DelegatedCredential.GetHeader() != headers.VSRDelegatedCredential {
		t.Fatalf("delegated credential contract = %+v", cfg.Authz.DelegatedCredential)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	var ref *CanonicalBackendRef
	for _, model := range exported.Providers.Models {
		if model.Name == "gateway-model" && len(model.BackendRefs) == 1 {
			ref = &model.BackendRefs[0]
		}
	}
	if ref == nil {
		t.Fatalf("exported providers = %+v", exported.Providers.Models)
	}
	if !ref.Auth.Delegated() || ref.APIKey != "" || ref.Auth.APIKeyEnv != "" {
		t.Fatalf("exported backend ref = %+v, want auth.credential: delegated without a key", *ref)
	}
}

func TestDelegatedBackendIsRejectedWhenTheContractIsOff(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(delegatedBackendConfig(delegatedGatewayBackend, false)))
	if err == nil || !strings.Contains(err.Error(), "providers.models[gateway-model].backend_refs[0]: "+delegatedContractRequired) {
		t.Fatalf("error = %v, want the authz contract requirement", err)
	}
}

func TestDelegatedBackendRejectsAStaticKey(t *testing.T) {
	for _, key := range []string{"          api_key: shared-secret\n", "            api_key_env: GATEWAY_KEY\n"} {
		_, err := ParseYAMLBytes([]byte(delegatedBackendConfig(delegatedGatewayBackend+key, true)))
		if err == nil || !strings.Contains(err.Error(), "auth.credential: delegated cannot combine with a static key (api_key or auth.api_key_env)") {
			t.Fatalf("%s: error = %v, want the static key rejection", key, err)
		}
	}
}

func TestDelegatedCredentialModeIsClosed(t *testing.T) {
	backend := strings.Replace(delegatedGatewayBackend, "credential: delegated", "credential: static", 1)
	_, err := ParseYAMLBytes([]byte(delegatedBackendConfig(backend, true)))
	if err == nil || !strings.Contains(err.Error(), `auth.credential "static" is not a credential mode; write delegated`) {
		t.Fatalf("error = %v, want the credential mode rejection", err)
	}
}

func TestDelegatedBackendUsesTheAuthBlockHeader(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(delegatedBackendConfig(delegatedGatewayBackend+`            header: X-Gateway-Key
            prefix: ""
`, true)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	profile, err := cfg.GetProviderProfileForEndpoint("gateway-model_gateway")
	if err != nil {
		t.Fatal(err)
	}
	auth, err := profile.ResolveAuth()
	if err != nil {
		t.Fatal(err)
	}
	if !profile.Delegated() || auth.Header != "X-Gateway-Key" || auth.Prefix != "" {
		t.Fatalf("profile = %+v, auth = %+v, want the auth block's header without a prefix", profile, auth)
	}
}

func TestDelegatedBackendNeedsAnAuthHeader(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(delegatedBackendConfig(`
        - name: gateway
          provider: ollama
          endpoint: http://gateway.example:11434/v1
          auth: {credential: delegated}
`, true)))
	if err == nil || !strings.Contains(err.Error(), "auth.credential: delegated needs a header to carry the credential") {
		t.Fatalf("error = %v, want the auth header requirement", err)
	}
}

func TestDelegatedAndStaticBackendsDoNotShareACluster(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(delegatedBackendConfig(delegatedGatewayBackend+`        - name: direct
          provider: openai-compatible
          endpoint: https://direct.example/v1
          api_key: static-secret
`, true)))
	if err == nil || !strings.Contains(err.Error(), "credential source differ") {
		t.Fatalf("error = %v, want the credential source difference", err)
	}
}

func TestDelegatedCredentialHeaderIsLowercasedWithADefault(t *testing.T) {
	if got := (DelegatedCredentialConfig{Enabled: true}).GetHeader(); got != "x-vsr-delegated-credential" {
		t.Fatalf("default header = %q", got)
	}
	if got := (DelegatedCredentialConfig{Enabled: true, Header: " X-Gateway-Token "}).GetHeader(); got != "x-gateway-token" {
		t.Fatalf("configured header = %q", got)
	}
}

func TestDelegatedCredentialHeaderRejectsClientCredentialAndIdentityHeaders(t *testing.T) {
	for _, header := range []string{
		"Authorization", "proxy-authorization", "api-key", "X-Api-Key", "x-goog-api-key", "x-user-openai-key",
		"x-authz-user-id", "x-authz-user-groups", "x-authz-team-id", "x-authz-tenant-id", "x-auth-request-user",
	} {
		_, err := ParseYAMLBytes([]byte(delegatedBackendConfig(delegatedGatewayBackend, false) + `
global:
  services:
    authz:
      identity:
        user_id_header: x-auth-request-user
      delegated_credential:
        enabled: true
        header: ` + header + `
`))
		if err == nil || !strings.Contains(err.Error(), "carries a client credential or an identity") {
			t.Fatalf("header %q: error = %v, want the reserved header rejection", header, err)
		}
	}
}

func TestTrustedGatewayFormsAreRemoved(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(delegatedBackendConfig(`
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          trusted_gateway: true
`, false) + `
global:
  services:
    authz:
      gateway_credential: {enabled: true}
`))
	for _, want := range []string{
		"trusted_gateway is removed; run `vllm-sr config migrate` (auth: {credential: delegated})",
		"gateway_credential is removed; run `vllm-sr config migrate` (global.services.authz.delegated_credential)",
	} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("error = %v, want %q", err, want)
		}
	}
}

func delegatedDeploymentConfig(auth, extra string) string {
	raw := strings.Replace(systemOneDecisionYAML, "auth: {api_key_env: HOSTED_D3_TEST_KEY}", auth, 1)
	return strings.Replace(raw, "global:\n", "global:\n  services:\n    authz:\n      delegated_credential: {enabled: true}\n"+extra, 1)
}

func TestDelegatedDecisionModelDeploymentLoads(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(delegatedDeploymentConfig("auth: {credential: delegated}", "")))
	if err != nil {
		t.Fatalf("a decision-model deployment that requests ask may be delegated: %v", err)
	}
	if !cfg.ModelDeployments["hosted-d3"].Auth.Delegated() {
		t.Fatalf("deployment auth = %+v", cfg.ModelDeployments["hosted-d3"].Auth)
	}
}

func TestDelegatedDeploymentRefusals(t *testing.T) {
	for _, test := range []struct{ name, raw, want string }{
		{
			"the contract is off",
			strings.Replace(systemOneDecisionYAML, "auth: {api_key_env: HOSTED_D3_TEST_KEY}", "auth: {credential: delegated}", 1),
			"global.model_catalog.deployments.hosted-d3: " + delegatedContractRequired,
		},
		{
			"a static key",
			delegatedDeploymentConfig("auth: {credential: delegated, api_key_env: HOSTED_D3_TEST_KEY}", ""),
			"global.model_catalog.deployments.hosted-d3: auth.credential: delegated cannot combine with a static key (auth.api_key_env or auth.api_key_file)",
		},
		{
			"a model_runtime endpoint",
			strings.Replace(delegatedDeploymentConfig("auth: {api_key_env: HOSTED_D3_TEST_KEY}", ""),
				"      gateway-d3:\n", "      attached:\n        provider: model_runtime\n        endpoint: http://127.0.0.1:9000\n        auth: {credential: delegated}\n      gateway-d3:\n", 1),
			"global.model_catalog.deployments.attached: auth.credential: delegated, but the Router checks and prepares a model_runtime endpoint on its own behalf",
		},
		{
			"only Router-own calls reach it",
			strings.Replace(delegatedDeploymentConfig("auth: {api_key_env: HOSTED_D3_TEST_KEY}", ""),
				"      gateway-d3:\n", "      unused-d3:\n        provider: systemone\n        endpoint: https://unused.example.com/v1/systemone\n        model: vllm-sr/d3-ultra\n        auth: {credential: delegated}\n      gateway-d3:\n", 1),
			"global.model_catalog.deployments.unused-d3: auth.credential: delegated, but only Router-own calls reach it; a Router-own call never carries a user's credential, give it a static key",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := ParseYAMLBytes([]byte(test.raw)); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

// A global block parsed on its own has no recipe to ask a delegated
// deployment, so the check waits for the whole config, which still runs it.
func TestGlobalBlockDefersTheDelegatedDeploymentCheck(t *testing.T) {
	global := `
version: v0.5
global:
  services:
    authz:
      delegated_credential: {enabled: true}
  model_catalog:
    deployments:
      hosted-d3:
        provider: systemone
        endpoint: https://hosted.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {credential: delegated}
`
	cfg, err := ParseGlobalYAMLBytesWithoutEnvExpansion([]byte(global))
	if err != nil {
		t.Fatalf("global block refused: %v", err)
	}
	if !cfg.ModelDeployments["hosted-d3"].Auth.Delegated() {
		t.Fatalf("deployment auth = %+v", cfg.ModelDeployments["hosted-d3"].Auth)
	}
	want := "global.model_catalog.deployments.hosted-d3: auth.credential: delegated, but only Router-own calls reach it"
	if _, err := ParseYAMLBytesWithoutEnvExpansion([]byte(global)); err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("whole config error = %v, want %q", err, want)
	}
	reserved := strings.Replace(global, "hosted-d3:", `"`+ImplicitDeploymentPrefix+`d3":`, 1)
	if _, err := ParseGlobalYAMLBytesWithoutEnvExpansion([]byte(reserved)); err == nil || !strings.Contains(err.Error(), "are reserved for module defaults") {
		t.Fatalf("global block error = %v, want the reserved deployment name refused", err)
	}
}

func TestDelegatedUpstreamsKeepTheResponseCachePerUser(t *testing.T) {
	cache := func(scope string) string {
		return delegatedBackendConfig(delegatedGatewayBackend, true) +
			"  plugins:\n    - type: response_cache\n      configuration: {enabled: true, mode: exact" + scope + "}\n"
	}
	if _, err := ParseYAMLBytes([]byte(cache(""))); err != nil {
		t.Fatalf("the default user scope keeps answers per user: %v", err)
	}
	_, err := ParseYAMLBytes([]byte(cache(", scope: global")))
	want := `global.plugins: response_cache scope "global" shares answers across users, and providers.models[gateway-model].backend_refs[0] is delegated`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("error = %v, want %q", err, want)
	}
}

// A request that matches no decision renders its automatic output on the
// default model, which a delegated backend cannot serve to the Router.
func TestDefaultAutomaticOutputRejectsADelegatedDefaultModel(t *testing.T) {
	doc := strings.Replace(delegatedBackendConfig(delegatedGatewayBackend, true), "providers:\n", "providers:\n  defaults: {model: gateway-model}\n", 1) + `  plugins:
    - type: request_params
      configuration:
        default_max_tokens: auto
`
	_, err := ParseYAMLBytes([]byte(doc))
	if err == nil || !strings.Contains(err.Error(), "the automatic output render (decisions[recipes[default].plugins]) calls it as the Router") {
		t.Fatalf("error = %v, want the default plugins' automatic output refused on a delegated default model", err)
	}
}
