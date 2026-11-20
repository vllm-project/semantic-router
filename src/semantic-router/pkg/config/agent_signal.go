package config

import (
	"fmt"
	"slices"
	"strings"
)

const SignalTypeAgent = "agent"

// The agent roles and harnesses an AgentRule may name.
const (
	AgentRoleMain               = "main"
	AgentRoleSubagent           = "subagent"
	AgentHarnessClaudeCode      = "claude_code"
	AgentHarnessCodex           = "codex"
	AgentHarnessOpenCode        = "opencode"
	AgentHarnessDeepSeekHarness = "deepseek_harness"
	AgentHarnessCursor          = "cursor"
	AgentHarnessOpenClaw        = "openclaw"
)

// The request classes Claude Code documents for x-claude-code-request-class.
// OpenCode, Codex and OpenClaw send none; the Router gives the calls their
// built-in prompts mark the auxiliary or compaction class and every other
// call main or subagent.
const (
	AgentRequestClassMain       = "main"
	AgentRequestClassSubagent   = "subagent"
	AgentRequestClassWorkflow   = "workflow"
	AgentRequestClassCompaction = "compaction"
	AgentRequestClassAuxiliary  = "auxiliary"
)

// AgentRule matches the agent a request belongs to, as its harness's headers
// name it: the main agent or a subagent, the harness, the type the harness
// gives a subagent, and the request's class, such as a compaction. Every
// field set must match. The headers are caller-controlled routing hints,
// never an authorization.
type AgentRule struct {
	Name           string   `yaml:"name"`
	Description    string   `yaml:"description,omitempty"`
	Role           string   `yaml:"role,omitempty"`
	Harness        string   `yaml:"harness,omitempty"`
	Types          []string `yaml:"types,omitempty"`
	RequestClasses []string `yaml:"request_classes,omitempty"`
}

// SupportedAgentRoles lists the roles an AgentRule may name.
func SupportedAgentRoles() []string { return []string{AgentRoleMain, AgentRoleSubagent} }

// SupportedAgentHarnesses lists the harnesses an AgentRule may name.
func SupportedAgentHarnesses() []string {
	return []string{
		AgentHarnessClaudeCode, AgentHarnessCodex, AgentHarnessOpenCode, AgentHarnessDeepSeekHarness,
		AgentHarnessCursor, AgentHarnessOpenClaw,
	}
}

// SupportedAgentRequestClasses lists the request classes an AgentRule may name.
func SupportedAgentRequestClasses() []string {
	return []string{
		AgentRequestClassMain, AgentRequestClassSubagent, AgentRequestClassWorkflow,
		AgentRequestClassCompaction, AgentRequestClassAuxiliary,
	}
}

func validateAgentContracts(cfg *RouterConfig) (err error) {
	defer func() { err = cfg.inRecipe(err) }()
	seen := make(map[string]struct{}, len(cfg.AgentRules))
	for i, rule := range cfg.AgentRules {
		if err := ValidateAgentRuleContract(rule); err != nil {
			return fmt.Errorf("signals.agent[%d]: %w", i, err)
		}
		normalizedName := strings.ToLower(rule.Name)
		if _, exists := seen[normalizedName]; exists {
			return fmt.Errorf("signals.agent[%d]: duplicate name %q", i, rule.Name)
		}
		seen[normalizedName] = struct{}{}
	}
	return nil
}

// ValidateAgentRuleContract validates a single agent rule. Exported so the
// DSL compiler and validator share the exact same contract.
func ValidateAgentRuleContract(rule AgentRule) error {
	trimmedName := strings.TrimSpace(rule.Name)
	if trimmedName == "" {
		return fmt.Errorf("name is required")
	}
	if trimmedName != rule.Name {
		return fmt.Errorf("name must not contain surrounding whitespace")
	}
	if rule.Role == "" && rule.Harness == "" && len(rule.Types) == 0 && len(rule.RequestClasses) == 0 {
		return fmt.Errorf("set at least one of role, harness, types or request_classes")
	}
	if rule.Role != "" && !slices.Contains(SupportedAgentRoles(), rule.Role) {
		return fmt.Errorf("role must be one of %s, got %q", strings.Join(SupportedAgentRoles(), ", "), rule.Role)
	}
	if rule.Harness != "" && !slices.Contains(SupportedAgentHarnesses(), rule.Harness) {
		return fmt.Errorf("harness must be one of %s, got %q", strings.Join(SupportedAgentHarnesses(), ", "), rule.Harness)
	}
	if rule.Role == AgentRoleMain && len(rule.Types) > 0 {
		return fmt.Errorf("types name a subagent's type, so role must be subagent or unset")
	}
	for i, agentType := range rule.Types {
		if agentType == "" || strings.TrimSpace(agentType) != agentType {
			return fmt.Errorf("types[%d] must be nonempty and trimmed", i)
		}
	}
	return validateAgentRequestClasses(rule)
}

func validateAgentRequestClasses(rule AgentRule) error {
	if len(rule.RequestClasses) > 0 {
		switch rule.Harness {
		case AgentHarnessDeepSeekHarness:
			return fmt.Errorf("request_classes cannot go with harness deepseek_harness: DeepSeek Harness calls carry no request class")
		case AgentHarnessCursor:
			return fmt.Errorf("request_classes cannot go with harness cursor: Cursor calls carry no request class")
		}
	}
	for i, class := range rule.RequestClasses {
		if !slices.Contains(SupportedAgentRequestClasses(), class) {
			return fmt.Errorf("request_classes[%d] must be one of %s, got %q", i, strings.Join(SupportedAgentRequestClasses(), ", "), class)
		}
	}
	return nil
}
