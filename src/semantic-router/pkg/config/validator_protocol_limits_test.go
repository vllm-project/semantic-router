package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func protocolLimitsConfig(limits string) []byte {
	return []byte(`
version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: route
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: cheap
global:
  router:
    allow_lossy_translation: [tool_result.is_error]
` + limits)
}

func TestProtocolLimitsDefaultWhenOmitted(t *testing.T) {
	cfg, err := ParseYAMLBytes(protocolLimitsConfig(""))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	policy, defaults := cfg.ProtocolPolicy(), llmprotocol.DefaultPolicy()
	if policy.Limits != defaults.Limits {
		t.Fatalf("omitted protocol_limits changed the codec limits: %+v", policy.Limits)
	}
	if len(policy.AllowedLossyFields) != 1 || policy.AllowedLossyFields[0] != "tool_result.is_error" {
		t.Fatalf("policy lost allow_lossy_translation: %v", policy.AllowedLossyFields)
	}
	if (*RouterConfig)(nil).ProtocolPolicy().Limits != defaults.Limits {
		t.Fatal("nil config must yield the default policy")
	}
}

func TestProtocolLimitsReachPolicyAndExport(t *testing.T) {
	cfg, err := ParseYAMLBytes(protocolLimitsConfig(`    protocol_limits:
      tool_name_bytes: 512
      tool_description_bytes: 65536
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	limits := cfg.ProtocolPolicy().Limits
	if limits.ToolNameBytes != 512 || limits.ToolDescriptionBytes != 65536 {
		t.Fatalf("policy limits = %d/%d, want 512/65536", limits.ToolNameBytes, limits.ToolDescriptionBytes)
	}
	defaults := llmprotocol.DefaultPolicy().Limits
	if limits.BodyBytes != defaults.BodyBytes || limits.Tools != defaults.Tools {
		t.Fatal("tool text overrides must not relax the body or tool-count caps")
	}
	exported := CanonicalGlobalFromRouterConfig(cfg).Router.ProtocolLimits
	if exported != (ProtocolLimitsConfig{ToolNameBytes: 512, ToolDescriptionBytes: 65536}) {
		t.Fatalf("exported protocol_limits = %+v", exported)
	}
}

func TestProtocolLimitsZeroKeepsDefault(t *testing.T) {
	cfg, err := ParseYAMLBytes(protocolLimitsConfig(`    protocol_limits:
      tool_name_bytes: 0
      tool_description_bytes: 65536
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if got := cfg.ProtocolPolicy().Limits.ToolNameBytes; got != llmprotocol.DefaultPolicy().Limits.ToolNameBytes {
		t.Fatalf("tool_name_bytes 0 = %d, want the default", got)
	}
}

func TestProtocolLimitsRejectOutOfRange(t *testing.T) {
	cases := map[string]string{
		"tool_name_bytes: -1":              "global.router.protocol_limits.tool_name_bytes: -1 is out of range",
		"tool_name_bytes: 1025":            "global.router.protocol_limits.tool_name_bytes: 1025 is out of range",
		"tool_description_bytes: -1":       "global.router.protocol_limits.tool_description_bytes: -1 is out of range",
		"tool_description_bytes: 1048577":  "global.router.protocol_limits.tool_description_bytes: 1048577 is out of range",
		"tool_description_bytes: 99999999": "global.router.protocol_limits.tool_description_bytes: 99999999 is out of range",
	}
	for field, want := range cases {
		t.Run(field, func(t *testing.T) {
			_, err := ParseYAMLBytes(protocolLimitsConfig("    protocol_limits:\n      " + field + "\n"))
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, want)
			}
		})
	}
	if _, err := ParseYAMLBytes(protocolLimitsConfig(`    protocol_limits:
      tool_name_bytes: 1024
      tool_description_bytes: 1048576
`)); err != nil {
		t.Fatalf("maximum overrides were rejected: %v", err)
	}
}
