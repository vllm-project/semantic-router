package config

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"

const (
	defaultRouterReplayMaxRecords   = 10000
	defaultRouterReplayMaxBodyBytes = 4096
	// defaultRouterReplayMaxToolTraceBytes caps each structured text field
	// (Prompt, ToolDefinitions, and per-step tool-trace Arguments/Output) by
	// default. Without a default these fields are unbounded (0 = no limit):
	// they are extracted from the *full* request before MaxBodyBytes truncation,
	// so a single large request (e.g. a 10 MB input) produces a multi-MB Prompt
	// that survives in the record. A page of such records then exceeds the
	// ext-proc message-size cap, and the /v1/router_replay list endpoint returns
	// 413 — breaking the history view for everyone (reported by internal users).
	// Mirroring the raw-body budget (4096) keeps prompts useful for debugging
	// while bounding growth, consistent with the MaxBodyBytes and
	// MaxToolTraceSteps defaults. Operators who need fuller structured fields
	// can raise or zero (unlimited) this explicitly.
	defaultRouterReplayMaxToolTraceBytes = 4096
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
		MaxToolTraceBytes:   defaultRouterReplayMaxToolTraceBytes,
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
