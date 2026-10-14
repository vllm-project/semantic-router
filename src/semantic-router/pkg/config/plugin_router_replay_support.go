package config

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"

const (
	defaultRouterReplayMaxRecords   = 10000
	defaultRouterReplayMaxBodyBytes = 4096
	// defaultRouterReplayMaxToolTraceSteps caps the per-record tool-trace
	// step count by default so an agentic session that loops forever can't
	// drive the router to OOM (see #1835). 100 covers normal multi-tool
	// flows without losing the user query at the head of the timeline; the
	// truncation policy is "drop oldest" so the most recent steps stay.
	defaultRouterReplayMaxToolTraceSteps = 100
)

func DefaultRouterReplayPluginConfig() RouterReplayPluginConfig {
	return RouterReplayPluginConfig{
		Enabled:             true,
		MaxRecords:          defaultRouterReplayMaxRecords,
		CaptureRequestBody:  true,
		CaptureResponseBody: true,
		MaxBodyBytes:        defaultRouterReplayMaxBodyBytes,
		MaxToolTraceSteps:   defaultRouterReplayMaxToolTraceSteps,
	}
}

// EffectiveRouterReplayConfigForDecision returns the replay configuration that
// should apply to a decision after layering global enablement and any
// per-decision router_replay plugin overrides.
func (c *RouterConfig) EffectiveRouterReplayConfigForDecision(decisionName string) *RouterReplayPluginConfig {
	if c == nil {
		base := DefaultRouterReplayPluginConfig()
		return &base
	}
	return c.EffectiveRouterReplayConfig(c.GetDecisionByName(decisionName))
}

// EffectiveRouterReplayConfig resolves replay policy from an already scoped
// decision. Request-time callers should prefer this form because decision names
// are recipe-local and therefore cannot identify a decision globally.
func (c *RouterConfig) EffectiveRouterReplayConfig(decision *Decision) *RouterReplayPluginConfig {
	base := DefaultRouterReplayPluginConfig()
	if c == nil {
		return &base
	}
	base = c.RouterReplay.captureDefaults()

	if decision == nil {
		if base.Enabled {
			return &base
		}
		return nil
	}

	plugin := decision.GetPlugin(DecisionPluginRouterReplay)
	if plugin == nil {
		if base.Enabled {
			return &base
		}
		return nil
	}
	if plugin.Configuration == nil {
		if base.Enabled {
			return &base
		}
		return nil
	}

	if err := UnmarshalPluginConfig(plugin.Configuration, &base); err != nil {
		logging.Errorf("Failed to unmarshal %s config: %v", DecisionPluginRouterReplay, err)
		return nil
	}
	if !base.Enabled {
		return nil
	}
	return &base
}

// CapturesPersonalData reports the resolved content policy. An omitted setting
// preserves the established capture behavior.
func (p *RouterReplayPluginConfig) CapturesPersonalData() bool {
	return p == nil || p.CapturePersonalData == nil || *p.CapturePersonalData
}

func (c RouterReplayConfig) captureDefaults() RouterReplayPluginConfig {
	base := DefaultRouterReplayPluginConfig()
	base.Enabled = c.Enabled
	if c.CaptureRequestBody != nil {
		base.CaptureRequestBody = *c.CaptureRequestBody
	}
	if c.CaptureResponseBody != nil {
		base.CaptureResponseBody = *c.CaptureResponseBody
	}
	if c.CapturePersonalData != nil {
		value := *c.CapturePersonalData
		base.CapturePersonalData = &value
	}
	if c.MaxRecords != nil {
		base.MaxRecords = *c.MaxRecords
	}
	if c.MaxBodyBytes != nil {
		base.MaxBodyBytes = *c.MaxBodyBytes
	}
	if c.MaxToolTraceBytes != nil {
		base.MaxToolTraceBytes = *c.MaxToolTraceBytes
	}
	if c.MaxToolTraceSteps != nil {
		base.MaxToolTraceSteps = *c.MaxToolTraceSteps
	}
	return base
}

// ReplayNeedsPIIEvidence marks existing PII rules as consumers whenever an
// enabled global default or possible selected decision can suppress personal
// content. No rules means no model dependency: capture will omit unverified
// content conservatively at runtime.
func (c *RouterConfig) ReplayNeedsPIIEvidence() bool {
	if c == nil || len(c.PIIRules) == 0 {
		return false
	}
	return c.replayNeedsPIIEvidenceForDecisions(c.Decisions)
}

func (c *RouterConfig) replayNeedsPIIEvidenceForDecisions(decisions []Decision) bool {
	if c == nil || len(c.PIIRules) == 0 {
		return false
	}
	if policy := c.EffectiveRouterReplayConfig(nil); policy != nil && !policy.CapturesPersonalData() {
		return true
	}
	for i := range decisions {
		if policy := c.EffectiveRouterReplayConfig(&decisions[i]); policy != nil && !policy.CapturesPersonalData() {
			return true
		}
	}
	return false
}
