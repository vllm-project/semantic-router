package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// What a replay record keeps of the request and its answer besides routing
// data (the decision, model, signals, timing, ids and usage).
const (
	// PromptCaptureNone keeps no request or response text, and reads no
	// prompt for personal data. It is the default.
	PromptCaptureNone = "none"
	// PromptCaptureRaw keeps the prompt, and the bodies and tool traces the
	// capture switches allow, as they came.
	PromptCaptureRaw = "raw"
	// PromptCaptureVerified keeps the exact prompt alone, once the recipe's
	// pii rules read it whole and find no personal data.
	PromptCaptureVerified = "verified"
)

const (
	defaultRouterReplayMaxRecords   = 10000
	defaultRouterReplayMaxBodyBytes = 4096
	// defaultRouterReplayMaxToolTraceSteps caps the per-record tool-trace
	// step count by default so an agentic session that loops forever can't
	// drive the router to OOM (see #1835). 100 covers normal multi-tool
	// flows without losing the user query at the head of the timeline; the
	// truncation policy is "drop oldest" so the most recent steps stay.
	defaultRouterReplayMaxToolTraceSteps = 100
	// defaultRouterReplayMaxPromptBytes caps the prompt a record keeps under
	// raw and verified capture when neither max_prompt_bytes nor
	// max_tool_trace_bytes is set: 64 KiB, about 16K tokens of English text,
	// keeps nearly every chat prompt whole and the head of a long one, so
	// 10,000 records of 1M-token prompts take some 640 MiB, not 40 GB.
	defaultRouterReplayMaxPromptBytes = 64 << 10
)

// DefaultRouterReplayPluginConfig leaves PromptCapture unset, which is none,
// so that a deprecated capture_personal_data still decides where it is set.
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

// PromptCaptureMode is the policy's prompt capture mode: the prompt_capture
// set, else what a deprecated capture_personal_data meant, else none. A value
// loading refuses is none too, so that no unknown mode keeps text.
func (p *RouterReplayPluginConfig) PromptCaptureMode() string {
	if p == nil {
		return PromptCaptureNone
	}
	mode := p.PromptCapture
	if mode == "" && p.CapturePersonalData != nil {
		mode = personalDataPromptCapture(*p.CapturePersonalData)
	}
	if mode == PromptCaptureRaw || mode == PromptCaptureVerified {
		return mode
	}
	return PromptCaptureNone
}

// personalDataPromptCapture is the prompt capture mode a deprecated
// capture_personal_data value meant.
func personalDataPromptCapture(capture bool) string {
	if capture {
		return PromptCaptureRaw
	}
	return PromptCaptureVerified
}

// ValidatePromptCapture refuses a prompt_capture other than none, raw or
// verified, and a negative max_prompt_bytes.
func (p *RouterReplayPluginConfig) ValidatePromptCapture() error {
	switch p.PromptCapture {
	case "", PromptCaptureNone, PromptCaptureRaw, PromptCaptureVerified:
	default:
		return fmt.Errorf("prompt_capture must be none, raw or verified, not %q", p.PromptCapture)
	}
	if p.MaxPromptBytes != nil && *p.MaxPromptBytes < 0 {
		return fmt.Errorf("max_prompt_bytes cannot be negative; 0 keeps the whole prompt")
	}
	return nil
}

// PromptByteLimit is how many bytes of the prompt a record keeps under raw and
// verified, 0 for the whole prompt: max_prompt_bytes from the most specific
// layer that sets it; where none does, max_tool_trace_bytes when that is above
// 0, since it bounded the prompt before max_prompt_bytes existed and no
// configuration may keep more of the prompt after an upgrade; else 64 KiB.
func (p *RouterReplayPluginConfig) PromptByteLimit() int {
	switch {
	case p == nil:
		return defaultRouterReplayMaxPromptBytes
	case p.MaxPromptBytes != nil:
		return *p.MaxPromptBytes
	case p.MaxToolTraceBytes > 0:
		return p.MaxToolTraceBytes
	default:
		return defaultRouterReplayMaxPromptBytes
	}
}

// CapturesRawContent reports whether records keep request and response text
// as it came, without reading it for personal data.
func (p *RouterReplayPluginConfig) CapturesRawContent() bool {
	return p.PromptCaptureMode() == PromptCaptureRaw
}

// VerifiesPrompt reports whether records keep the prompt alone, once the
// recipe's pii rules find it clean.
func (p *RouterReplayPluginConfig) VerifiesPrompt() bool {
	return p.PromptCaptureMode() == PromptCaptureVerified
}

// CapturesRequestText reports whether a record may hold request text at all:
// the prompt, the body and the request's tool trace. prompt_capture: none
// holds none, whatever capture_request_body says.
func (p *RouterReplayPluginConfig) CapturesRequestText() bool {
	return p != nil && p.CaptureRequestBody && p.PromptCaptureMode() != PromptCaptureNone
}

// CapturesResponseText reports the same for the answer: the body, its tool
// trace, detector excerpts and provider replies.
func (p *RouterReplayPluginConfig) CapturesResponseText() bool {
	return p != nil && p.CaptureResponseBody && p.PromptCaptureMode() != PromptCaptureNone
}

// CapturesToolResults reports whether client tool-result text is stored.
// Tool results are raw client data, so only raw capture stores them;
// configuration loading refuses them with verified.
func (p *RouterReplayPluginConfig) CapturesToolResults() bool {
	return p != nil && p.CaptureToolResults && p.CapturesRawContent()
}

// ReplayNeedsPIIEvidence marks existing PII rules as consumers whenever an
// enabled global default or possible selected decision keeps prompts only
// once verified. none and raw read no prompt for personal data. No rules
// means no model dependency: verified capture omits the prompt
// conservatively at runtime.
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
	if c.EffectiveRouterReplayConfig(nil).VerifiesPrompt() {
		return true
	}
	for i := range decisions {
		if c.EffectiveRouterReplayConfig(&decisions[i]).VerifiesPrompt() {
			return true
		}
	}
	return false
}
