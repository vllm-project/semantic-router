package config

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const DecisionPluginTools = "tools"

// Policy modes: passthrough leaves the request's tools as they are, filtered
// keeps the ones allow_tools and block_tools allow, and none removes them.
const (
	ToolsPolicyModePassthrough = "passthrough"
	ToolsPolicyModeFiltered    = "filtered"
	ToolsPolicyModeNone        = "none"
)

// Selection modes, and what a selection does when its call fails and when a
// request has more candidate tools than one question takes options.
const (
	ToolsSelectModeFilter = "filter"
	ToolsSelectModeAdd    = "add"

	ToolsSelectOnErrorKeep  = "keep"
	ToolsSelectOnErrorStrip = "strip"
	ToolsSelectOnErrorSkip  = "skip"
	ToolsSelectOnErrorFail  = "fail"

	ToolsSelectOnOverflowKeep   = "keep"
	ToolsSelectOnOverflowReject = "reject"
)

// DefaultToolsNeedThreshold is the least P(tool_need) at which a selection
// keeps or adds the tools its judge chose.
const DefaultToolsNeedThreshold = 0.5

const (
	// StickyToolSelectionDefaultMaxTools is the hard safety bound on the
	// number of tools a sticky session retains when sticky.max_tools is
	// unset. See issue #3347.
	StickyToolSelectionDefaultMaxTools = 16

	// StickyToolSelectionDefaultMaxNewToolsPerTurn bounds how many newly
	// relevant tools sticky selection may append in a single turn when
	// sticky.max_new_tools_per_turn is unset.
	StickyToolSelectionDefaultMaxNewToolsPerTurn = 2

	// StickyToolSelectionMaxToolsUpperBound is the hard ceiling accepted for
	// an explicit sticky.max_tools value.
	StickyToolSelectionMaxToolsUpperBound = 128
)

// ToolsPluginConfig is a decision's tool handling. The policy runs first and
// is deterministic; the selection then asks the decision model which of the
// tools the policy allows the request needs.
type ToolsPluginConfig struct {
	Enabled bool               `json:"enabled" yaml:"enabled"`
	Policy  *ToolsPolicyConfig `json:"policy,omitempty" yaml:"policy,omitempty"`
	Select  *ToolsSelectConfig `json:"select,omitempty" yaml:"select,omitempty"`
}

// ToolsPolicyConfig allows, blocks or removes the request's tools by name.
type ToolsPolicyConfig struct {
	// Mode is passthrough (the default), filtered or none.
	Mode       string   `json:"mode,omitempty" yaml:"mode,omitempty" jsonschema:"enum=passthrough,enum=filtered,enum=none"`
	AllowTools []string `json:"allow_tools,omitempty" yaml:"allow_tools,omitempty"`
	BlockTools []string `json:"block_tools,omitempty" yaml:"block_tools,omitempty"`
	// StripToolHistory removes provider-bound tool/function call and result
	// messages after routing has evaluated the original conversation. It does
	// not change the signal facts used to select the decision.
	StripToolHistory bool `json:"strip_tool_history,omitempty" yaml:"strip_tool_history,omitempty"`
	// TrustedFacts carries disabled-by-default typed facts that distinguish
	// capability, authorization, availability, and Looper stage roles
	// (issue #3476). When nil or with Enabled=false, the policy applies
	// without them.
	TrustedFacts *TrustedFactsConfig `json:"trusted_facts,omitempty" yaml:"trusted_facts,omitempty"`
}

// ToolsSelectConfig asks the decision model which of the candidate tools a
// request needs, in one call: tool_need, whether it needs a tool at all, and
// the match judge, which ranks the candidates (tool_rank by default).
type ToolsSelectConfig struct {
	// Mode is filter (keep the request's tools the model chooses, in the
	// request's order) or add (append the catalog's tools the model chooses
	// after the request's own).
	Mode string `json:"mode" yaml:"mode" jsonschema:"enum=filter,enum=add"`
	// ToolsDBPath is the catalog add mode chooses from. A relative path is
	// read from the Router's working directory.
	ToolsDBPath string `json:"tools_db_path,omitempty" yaml:"tools_db_path,omitempty"`
	// Match is the judge that ranks the candidates and its keep rules;
	// without it tool_rank keeps the 3 best at 0.1 of the best or more.
	Match *MatchConfig `json:"match,omitempty" yaml:"match,omitempty"`
	// NeedThreshold is the least P(tool_need) at which the judge's tools are
	// kept or added, in [0, 1] (default 0.5).
	NeedThreshold *float64 `json:"need_threshold,omitempty" yaml:"need_threshold,omitempty"`
	// PreserveCount keeps the request's first tools in filter mode, whatever
	// the model answers.
	PreserveCount int `json:"preserve_count,omitempty" yaml:"preserve_count,omitempty"`
	// OnError is what a failed or late call does: skip (the default) or fail
	// in add mode; keep (the default), strip or fail in filter mode.
	OnError string `json:"on_error,omitempty" yaml:"on_error,omitempty" jsonschema:"enum=skip,enum=keep,enum=strip,enum=fail"`
	// OnOverflow is what filter mode does with more than 255 candidate tools
	// after the policy: keep forwards them unselected (the default), reject
	// refuses the request.
	OnOverflow string `json:"on_overflow,omitempty" yaml:"on_overflow,omitempty" jsonschema:"enum=keep,enum=reject"`
	// Input names what the questions read besides the current user turn.
	Input ToolsSelectInput `json:"input,omitempty" yaml:"input,omitempty"`
	// TimeoutMs bounds the call, up to 60000; without it a systemone
	// deployment's own timeout_ms applies, else 1000.
	TimeoutMs int `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`
	// Sticky configures session-scoped sticky selection (issue #3347), which
	// this release does not run: sticky.enabled is refused at load.
	Sticky *StickyToolSelectionConfig `json:"sticky,omitempty" yaml:"sticky,omitempty"`
}

// ToolsSelectInput names what the selection reads besides the current user
// turn.
type ToolsSelectInput struct {
	// History adds the earlier user turns.
	History bool `json:"history,omitempty" yaml:"history,omitempty"`
}

// StickyToolSelectionConfig configures opt-in, bounded session-scoped
// tool-set stickiness (issue #3347). Disabled by default: while Enabled is
// false, selection runs per turn and never reads or writes session state.
//
// Only tool identities and bounded metadata are ever persisted under this
// config, in pkg/sessiontools — never full llmprotocol.Tool values,
// descriptions, JSON Schemas, arguments, results, prompts, authorization
// decisions, credentials, or raw session/principal identifiers.
type StickyToolSelectionConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`

	// MaxTools and MaxNewToolsPerTurn are pointers: for both, an explicit 0
	// is a distinct configured value from "unset" — max_tools: 0 is
	// rejected (valid range is 1..128), while max_new_tools_per_turn: 0 is
	// valid and means "no relevance-driven growth, reuse and call-pinning
	// only".
	MaxTools           *int  `json:"max_tools,omitempty" yaml:"max_tools,omitempty"`
	MaxNewToolsPerTurn *int  `json:"max_new_tools_per_turn,omitempty" yaml:"max_new_tools_per_turn,omitempty"`
	PinCalledTools     *bool `json:"pin_called_tools,omitempty" yaml:"pin_called_tools,omitempty"`
}

// TrustedFactsConfig is the decision-scoped contract for trusted tool and
// MCP capability facts (issue #3476). It is intentionally additive: when nil,
// when Enabled is false, or when omitted, the router behaves exactly as
// before. Authoritative inputs are limited to operator-owned recipe policy,
// gateway-attested authorization context, and bounded/fresh runtime
// availability. Generic client metadata, prompt text, and model output are
// never authorization sources.
type TrustedFactsConfig struct {
	// Enabled gates the entire trusted-facts path. When false, none of the
	// other fields are consulted and validation skips range checks.
	Enabled bool `json:"enabled" yaml:"enabled"`
	// Enforcement names the policy strength for candidate/verifier/advisor
	// and final stages. Must be one of "disabled", "advisory", or
	// "authoritative". When empty, EffectiveEnforcement returns "advisory"
	// (observe-only) so enabling the block without a mode never enforces.
	// Disabled-by-default means a nil or Enabled=false block; an Enabled
	// block with empty enforcement is intentionally advisory, not disabled.
	// Production authoritative use must set enforcement explicitly.
	Enforcement string `json:"enforcement,omitempty" yaml:"enforcement,omitempty"`
	// TrustSources lists the authoritative sources consulted. Allowed values
	// are "operator-policy", "gateway-attested", and "runtime-fresh".
	// Must declare at least one when Enabled. Values such as
	// "client-metadata", "prompt", or "model-output" are never allowed.
	TrustSources []string `json:"trust_sources,omitempty" yaml:"trust_sources,omitempty"`
	// FreshnessSeconds bounds how old runtime availability evidence may be.
	// 0 disables the freshness check. Must be in [0, 86400] when Enabled,
	// and must be >0 when trust_sources includes "runtime-fresh" (a
	// runtime-fresh claim without a bound would accept stale evidence).
	FreshnessSeconds int `json:"freshness_seconds,omitempty" yaml:"freshness_seconds,omitempty"`
	// StageRoles lists the Looper stage roles this decision may authorize.
	// Allowed values are "candidate", "verifier", "advisor", and "final".
	// Must declare at least one when Enabled.
	StageRoles []string `json:"stage_roles,omitempty" yaml:"stage_roles,omitempty"`
}

// Trusted enforcement modes. The canonical values are owned by
// pkg/llmprotocol so config validation and the request-path gate share one
// vocabulary.
const (
	TrustedEnforcementDisabled      = string(llmprotocol.TrustedDisabled)
	TrustedEnforcementAdvisory      = string(llmprotocol.TrustedAdvisory)
	TrustedEnforcementAuthoritative = string(llmprotocol.TrustedAuthoritative)
)

// Trusted trust-source values. The canonical values are owned by
// pkg/llmprotocol; see llmprotocol.TrustedSource.Authorizes for which
// declarations can authorize.
const (
	TrustedSourceOperatorPolicy  = string(llmprotocol.TrustedSourceOperatorPolicy)
	TrustedSourceGatewayAttested = string(llmprotocol.TrustedSourceGatewayAttested)
	TrustedSourceRuntimeFresh    = string(llmprotocol.TrustedSourceRuntimeFresh)
)

// Trusted Looper stage-role values. The canonical values are owned by
// pkg/llmprotocol.
const (
	TrustedStageCandidate = string(llmprotocol.TrustedStageCandidate)
	TrustedStageVerifier  = string(llmprotocol.TrustedStageVerifier)
	TrustedStageAdvisor   = string(llmprotocol.TrustedStageAdvisor)
	TrustedStageFinal     = string(llmprotocol.TrustedStageFinal)
)

// GetToolsConfig returns the tools plugin configuration.
func (d *Decision) GetToolsConfig() *ToolsPluginConfig {
	result := &ToolsPluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginTools, result)
}

// ActivePolicy is the policy of an enabled plugin, nil otherwise.
func (c *ToolsPluginConfig) ActivePolicy() *ToolsPolicyConfig {
	if c == nil || !c.Enabled {
		return nil
	}
	return c.Policy
}

// ActiveSelect is the selection of an enabled plugin, nil otherwise.
func (c *ToolsPluginConfig) ActiveSelect() *ToolsSelectConfig {
	if c == nil || !c.Enabled {
		return nil
	}
	return c.Select
}

// EffectiveMode returns the defaulted policy mode.
func (p *ToolsPolicyConfig) EffectiveMode() string {
	if p == nil || p.Mode == "" {
		return ToolsPolicyModePassthrough
	}
	return p.Mode
}

// Allows reports whether the policy keeps a tool by name.
func (p *ToolsPolicyConfig) Allows(name string) bool {
	switch p.EffectiveMode() {
	case ToolsPolicyModeNone:
		return false
	case ToolsPolicyModeFiltered:
		for _, blocked := range p.BlockTools {
			if blocked == name {
				return false
			}
		}
		if len(p.AllowTools) == 0 {
			return true
		}
		for _, allowed := range p.AllowTools {
			if allowed == name {
				return true
			}
		}
		return false
	default:
		return true
	}
}

// TrustedFactsEnabled reports whether an enabled plugin's policy opts in to
// trusted facts.
func (c *ToolsPluginConfig) TrustedFactsEnabled() bool {
	policy := c.ActivePolicy()
	return policy != nil && policy.TrustedFacts != nil && policy.TrustedFacts.Enabled
}

// EffectiveEnforcement returns the enforcement mode, defaulting to
// "advisory" (observe-only) when unset so that enabling the block never
// silently enforces.
func (t *TrustedFactsConfig) EffectiveEnforcement() string {
	if t == nil || t.Enforcement == "" {
		return TrustedEnforcementAdvisory
	}
	return t.Enforcement
}

// EffectiveNeedThreshold is the least P(tool_need) at which the judge's tools
// are kept or added.
func (s *ToolsSelectConfig) EffectiveNeedThreshold() float64 {
	if s.NeedThreshold == nil {
		return DefaultToolsNeedThreshold
	}
	return *s.NeedThreshold
}

// EffectiveOnError is what a failed call does in the selection's mode.
func (s *ToolsSelectConfig) EffectiveOnError() string {
	switch {
	case s.OnError != "":
		return s.OnError
	case s.Mode == ToolsSelectModeAdd:
		return ToolsSelectOnErrorSkip
	default:
		return ToolsSelectOnErrorKeep
	}
}

// EffectiveOnOverflow is what filter mode does with too many candidates.
func (s *ToolsSelectConfig) EffectiveOnOverflow() string {
	if s.OnOverflow == "" {
		return ToolsSelectOnOverflowKeep
	}
	return s.OnOverflow
}

// EffectiveMatch is the match block the selection runs: the one it writes,
// else tool_rank's default.
func (s *ToolsSelectConfig) EffectiveMatch() MatchConfig {
	return MatchToolsSelect.Effective(s.Match)
}

// ToolsSelectTimeout is a selection call's deadline on the deployment that
// answers it: its own, else a systemone deployment's own, else the default.
func (c *RouterConfig) ToolsSelectTimeout(s *ToolsSelectConfig, deployment string) time.Duration {
	if remote := c.ModelDeployments[deployment]; s.TimeoutMs == 0 && remote.IsSystemOne() && remote.TimeoutMs > 0 {
		return remote.CallTimeout()
	}
	return systemOneTimeout(s.TimeoutMs)
}

// EffectiveMaxTools returns the configured max_tools, or
// StickyToolSelectionDefaultMaxTools when unset.
func (s *StickyToolSelectionConfig) EffectiveMaxTools() int {
	if s == nil || s.MaxTools == nil {
		return StickyToolSelectionDefaultMaxTools
	}
	return *s.MaxTools
}

// EffectiveMaxNewToolsPerTurn returns the configured max_new_tools_per_turn,
// or StickyToolSelectionDefaultMaxNewToolsPerTurn when unset.
func (s *StickyToolSelectionConfig) EffectiveMaxNewToolsPerTurn() int {
	if s == nil || s.MaxNewToolsPerTurn == nil {
		return StickyToolSelectionDefaultMaxNewToolsPerTurn
	}
	return *s.MaxNewToolsPerTurn
}

// EffectivePinCalledTools returns the configured pin_called_tools, defaulting
// to true when unset.
func (s *StickyToolSelectionConfig) EffectivePinCalledTools() bool {
	if s == nil || s.PinCalledTools == nil {
		return true
	}
	return *s.PinCalledTools
}
