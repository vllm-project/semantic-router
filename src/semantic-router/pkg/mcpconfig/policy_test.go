package mcpconfig

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPolicyDisabledByDefault(t *testing.T) {
	policy := NewPolicy(config.MCPConfigServerConfig{Enabled: false})
	if err := policy.Check(ToolGetConfig, "agent", false); err == nil {
		t.Fatal("expected disabled policy to reject tool calls")
	}
}

func TestPolicyAllowlist(t *testing.T) {
	policy := NewPolicy(config.MCPConfigServerConfig{
		Enabled:      true,
		AllowedTools: []string{ToolGetConfig},
	})

	if err := policy.Check(ToolGetConfig, "agent", false); err != nil {
		t.Fatalf("get_config should be allowed: %v", err)
	}
	if err := policy.Check(ToolApplyPatch, "agent", false); err == nil {
		t.Fatal("expected apply_patch to be blocked by allowlist")
	}
}

func TestPolicyDestructiveGate(t *testing.T) {
	policy := NewPolicy(config.MCPConfigServerConfig{Enabled: true})
	if err := policy.Check("delete_signal", "agent", true); err == nil {
		t.Fatal("expected unknown/destructive tool to be rejected")
	}
}

func TestPolicyRateLimit(t *testing.T) {
	policy := NewPolicy(config.MCPConfigServerConfig{
		Enabled:            true,
		RateLimitPerMinute: 1,
	})

	if err := policy.Check(ToolGetConfig, "agent-a", false); err != nil {
		t.Fatalf("first call should pass: %v", err)
	}
	if err := policy.Check(ToolGetConfig, "agent-a", false); err == nil {
		t.Fatal("expected rate limit on second call")
	}
	if err := policy.Check(ToolGetConfig, "agent-b", false); err != nil {
		t.Fatalf("different actor should not share limit: %v", err)
	}
}
