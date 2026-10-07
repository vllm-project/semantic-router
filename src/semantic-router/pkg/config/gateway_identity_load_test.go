package config

import (
	"strings"
	"testing"
)

const identityLoadHead = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
`

const identityLoadProviders = `
providers:
  models:
    - name: model-a
      provider_model_id: model-a
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
`

// loadIdentityDoc loads a canonical config whose one listener trusts identity
// headers or not, followed by the routing and global sections under test.
func loadIdentityDoc(t *testing.T, trust bool, sections string) *RouterConfig {
	t.Helper()
	doc := identityLoadHead
	if trust {
		doc += "    identity:\n      trust_headers: true\n"
	}
	parsed, err := ParseYAMLBytes([]byte(doc + identityLoadProviders + sections))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	return parsed
}

func TestStandaloneLoadRefusesIdentityPolicyUntilAListenerTrustsIt(t *testing.T) {
	for name, sections := range map[string]string{
		"authz signal": `
routing:
  modelCards:
    - name: model-a
  signals:
    role_bindings:
      - name: admin
        role: admin
        subjects:
          - kind: Group
            name: platform-admins
  decisions:
    - name: admins
      priority: 10
      rules:
        operator: AND
        conditions:
          - type: authz
            name: admin
      modelRefs:
        - model: model-a
`,
		"per-user rate limit": `
routing:
  modelCards:
    - name: model-a
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: per-user
              match:
                user: "*"
              requests_per_unit: 10
              unit: minute
`,
		"authz providers": `
routing:
  modelCards:
    - name: model-a
global:
  services:
    authz:
      providers:
        - type: header-injection
          headers:
            X-Plan: premium
`,
	} {
		t.Run(name, func(t *testing.T) {
			err := ValidateGatewayCapabilities(loadIdentityDoc(t, false, sections), GatewayStandalone)
			if err == nil || !strings.Contains(err.Error(), "listeners[].identity.trust_headers") ||
				!strings.Contains(err.Error(), "--gateway extproc") {
				t.Fatalf("load without a trusting listener: error = %v, want both remedies", err)
			}
			if err := ValidateGatewayCapabilities(loadIdentityDoc(t, true, sections), GatewayStandalone); err != nil {
				t.Fatalf("a trusting listener serves it: %v", err)
			}
		})
	}
}

func TestStandaloneLoadWarnsWhenMemoryHasNoIdentity(t *testing.T) {
	sections := `
routing:
  modelCards:
    - name: model-a
global:
  stores:
    memory:
      enabled: true
`
	untrusted := loadIdentityDoc(t, false, sections)
	if err := ValidateGatewayCapabilities(untrusted, GatewayStandalone); err != nil {
		t.Fatalf("memory loads without a trusting listener: %v", err)
	}
	if warning := UntrustedIdentityWarning(untrusted); !strings.Contains(warning, "memory record") {
		t.Fatalf("warning = %q", warning)
	}
	if warning := UntrustedIdentityWarning(loadIdentityDoc(t, true, sections)); warning != "" {
		t.Fatalf("a trusting listener needs no warning: %q", warning)
	}
}
