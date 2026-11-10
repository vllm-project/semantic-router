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

// EffectiveRouterReplayConfig resolves the replay policy a request runs, from
// an already scoped decision (nil for a request that matched none): its
// router_replay plugin, layered as RequestPlugins says, over the capture
// defaults. Recording is off without one, or where no layer enables it.
// Request-time callers pass the decision because decision names are
// recipe-local and therefore cannot identify a decision globally.
func (c *RouterConfig) EffectiveRouterReplayConfig(decision *Decision) *RouterReplayPluginConfig {
	return RouterReplayPolicyOf(c.RequestPlugins(decision))
}

// UsesRouterReplay reports whether a reachable request records replay.
func (c *RouterConfig) UsesRouterReplay() bool {
	uses := false
	c.visitReachablePlugins(func(plugins []DecisionPlugin) {
		uses = uses || RouterReplayPolicyOf(plugins) != nil
	})
	return uses
}

// RouterReplayPolicyOf is the replay policy of a request's plugins, nil when
// none records.
func RouterReplayPolicyOf(plugins []DecisionPlugin) *RouterReplayPluginConfig {
	index := pluginIndex(plugins, DecisionPluginRouterReplay)
	if index < 0 || plugins[index].Configuration == nil {
		return nil
	}
	plugin := plugins[index]
	base := DefaultRouterReplayPluginConfig()
	base.Enabled = false
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

// CapturesToolResults reports whether client tool-result text is stored.
// Configuration loading refuses it without personal-data capture; requiring
// both here keeps a policy built outside that path redacted.
func (p *RouterReplayPluginConfig) CapturesToolResults() bool {
	return p != nil && p.CaptureToolResults && p.CapturesPersonalData()
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
