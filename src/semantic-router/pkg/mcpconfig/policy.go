package mcpconfig

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Coarse tool names exposed in PR1.
const (
	ToolGetConfig        = "get_config"
	ToolExportConfigJSON = "export_config_json"
	ToolExportConfigYAML = "export_config_yaml"
	ToolValidateConfig   = "validate_config"
	ToolDiffConfig       = "diff_config"
	ToolApplyPatch       = "apply_patch"
)

var coarseTools = []string{
	ToolGetConfig,
	ToolExportConfigJSON,
	ToolExportConfigYAML,
	ToolValidateConfig,
	ToolDiffConfig,
	ToolApplyPatch,
}

// Policy enforces MCP config server guardrails.
type Policy struct {
	cfg  config.MCPConfigServerConfig
	rate *rateLimiter
}

func NewPolicy(cfg config.MCPConfigServerConfig) *Policy {
	return &Policy{
		cfg:  cfg.WithDefaults(),
		rate: newRateLimiter(cfg.WithDefaults().RateLimitPerMinute),
	}
}

// Check returns an error when the tool call must be denied.
func (p *Policy) Check(toolName, actor string, destructive bool) error {
	if !p.cfg.Enabled {
		return fmt.Errorf("mcp config server is disabled")
	}
	if !isCoarseTool(toolName) {
		return fmt.Errorf("tool %q is not registered", toolName)
	}
	if len(p.cfg.AllowedTools) > 0 && !containsString(p.cfg.AllowedTools, toolName) {
		return fmt.Errorf("tool %q is not in the allowlist", toolName)
	}
	if destructive && !p.cfg.AllowDestructive {
		return fmt.Errorf("destructive tool %q requires allow_destructive=true in global.services.mcp_config", toolName)
	}
	if !p.rate.allow(actorKey(actor)) {
		return fmt.Errorf("rate limit exceeded for actor %q", actorKey(actor))
	}
	return nil
}

func isCoarseTool(name string) bool {
	return containsString(coarseTools, name)
}

// CoarseTools returns the PR1 tool inventory.
func CoarseTools() []string {
	out := make([]string, len(coarseTools))
	copy(out, coarseTools)
	return out
}

func actorKey(actor string) string {
	trimmed := strings.TrimSpace(actor)
	if trimmed == "" {
		return "anonymous"
	}
	return trimmed
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

type rateLimiter struct {
	limit int
	mu    sync.Mutex
	usage map[string][]time.Time
}

func newRateLimiter(limitPerMinute int) *rateLimiter {
	if limitPerMinute <= 0 {
		limitPerMinute = 60
	}
	return &rateLimiter{
		limit: limitPerMinute,
		usage: make(map[string][]time.Time),
	}
}

func (r *rateLimiter) allow(actor string) bool {
	now := time.Now()
	cutoff := now.Add(-time.Minute)

	r.mu.Lock()
	defer r.mu.Unlock()

	timestamps := r.usage[actor]
	filtered := timestamps[:0]
	for _, ts := range timestamps {
		if ts.After(cutoff) {
			filtered = append(filtered, ts)
		}
	}
	if len(filtered) >= r.limit {
		r.usage[actor] = filtered
		return false
	}
	filtered = append(filtered, now)
	r.usage[actor] = filtered
	return true
}
