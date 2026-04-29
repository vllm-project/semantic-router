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
	base := DefaultRouterReplayPluginConfig()
	if c == nil {
		return &base
	}
	base.Enabled = c.RouterReplay.Enabled

	decision := c.GetDecisionByName(decisionName)
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
