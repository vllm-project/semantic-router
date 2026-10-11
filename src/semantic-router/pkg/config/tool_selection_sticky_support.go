package config

import (
	"errors"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ErrToolSelectionStickyUnsupported reports a sticky-enabled decision or
// store this release cannot serve safely (issue #3347). Local sticky
// selection is supported only where the runtime can recheck every reused
// tool against current authorization before emission: a single-model
// decision whose tools plugin declares authoritative trusted facts that can
// authorize both candidate selection and final emission, backed by the local
// tool-session store. Looper algorithms and the Redis store stay rejected
// until their phases implement and test them.
var ErrToolSelectionStickyUnsupported = errors.New("tool_selection plugin: sticky selection is not supported for this configuration")

// StickyEnabled reports whether the plugin and its sticky block are both on.
func (c *ToolSelectionPluginConfig) StickyEnabled() bool {
	return c != nil && c.Enabled && c.Sticky != nil && c.Sticky.Enabled
}

// ValidateStickyToolSelectionSupport rejects the first sticky-enabled
// decision or store configuration the local runtime cannot serve. Config
// admission and router construction both call it, so a configuration that
// bypasses one entry point still fails at the other.
func ValidateStickyToolSelectionSupport(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	sticky := false
	for _, decision := range cfg.AllRoutingDecisions() {
		if !decision.GetToolSelectionConfig().StickyEnabled() {
			continue
		}
		sticky = true
		if err := validateStickyDecisionSupport(&decision); err != nil {
			return fmt.Errorf("decision '%s': %w", decision.Name, err)
		}
	}
	if !sticky {
		return nil
	}
	if backend := cfg.ToolSessions.EffectiveBackend(); backend != ToolSessionStoreBackendLocal {
		return fmt.Errorf("%w: global.stores.tool_sessions.backend %q is not supported yet; use %q",
			ErrToolSelectionStickyUnsupported, backend, ToolSessionStoreBackendLocal)
	}
	return nil
}

func validateStickyDecisionSupport(decision *Decision) error {
	if decision.Algorithm != nil && IsLooperAlgorithmType(decision.Algorithm.Type) {
		return fmt.Errorf("%w: algorithm %q sends tools to several models", ErrToolSelectionStickyUnsupported, decision.Algorithm.Type)
	}
	toolsCfg := decision.GetToolsConfig()
	if !toolsCfg.TrustedFactsEnabled() {
		return fmt.Errorf("%w: the decision's tools plugin must enable trusted_facts", ErrToolSelectionStickyUnsupported)
	}
	if toolsCfg.EffectiveMode() == ToolsPluginModeNone {
		return fmt.Errorf("%w: tools mode %q removes every tool", ErrToolSelectionStickyUnsupported, ToolsPluginModeNone)
	}
	trusted := toolsCfg.TrustedFacts
	if trusted.EffectiveEnforcement() != TrustedEnforcementAuthoritative {
		return fmt.Errorf("%w: trusted_facts.enforcement must be %q", ErrToolSelectionStickyUnsupported, TrustedEnforcementAuthoritative)
	}
	if !slices.ContainsFunc(trusted.TrustSources, func(source string) bool {
		return llmprotocol.TrustedSource(source).Authorizes()
	}) {
		return fmt.Errorf("%w: trusted_facts.trust_sources must include a source that authorizes tool use", ErrToolSelectionStickyUnsupported)
	}
	for _, stage := range []string{TrustedStageCandidate, TrustedStageFinal} {
		if !slices.Contains(trusted.StageRoles, stage) {
			return fmt.Errorf("%w: trusted_facts.stage_roles must include %q", ErrToolSelectionStickyUnsupported, stage)
		}
	}
	return nil
}
