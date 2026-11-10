package headers

import (
	"slices"
	"strings"
)

// TraceLimitBytes caps the encoded x-vsr-trace value. A trace that would
// exceed it loses its lowest-priority members and carries TraceTruncated.
const TraceLimitBytes = 4096

// The members of x-vsr-trace.
const (
	TraceConfigHash               = "config_hash"
	TraceInferenceCalls           = "inference_calls"
	TraceModelUsage               = "model_usage"
	TraceConfidence               = "confidence"
	TraceAlgorithm                = "algorithm"
	TraceRoutingLatencyMs         = "routing_latency_ms"
	TraceFallbackAttempts         = "fallback_attempts"
	TraceOnUnknown                = "on_unknown"
	TraceClientProtocol           = "client_protocol"
	TraceUpstreamProtocol         = "upstream_protocol"
	TraceEffectiveInputTokens     = "effective_input_tokens"
	TraceEffectiveMaxOutputTokens = "effective_max_output_tokens"
	TraceSignals                  = "signals"
	TraceContextTokens            = "context_tokens"
	TraceCategory                 = "category"
	TraceReasoning                = "reasoning"
	TraceModality                 = "modality"
	TraceSystemPrompt             = "system_prompt"
	TraceCacheSimilarity          = "cache_similarity"
	TraceRetention                = "retention"
	TracePromptCache              = "prompt_cache"
	TraceTools                    = "tools"
	TraceResponseGuard            = "response_guard"
	TraceSessionPhase             = "session_phase"
	TraceLearning                 = "learning"
	TraceLooper                   = "looper"
	TraceRanking                  = "ranking"
	TraceTruncated                = "truncated"
)

// TraceMember records one member of x-vsr-trace.
type TraceMember struct {
	Key string
	// Owner is the feature that reports the member.
	Owner string
	// Doc says what the member carries, in one sentence.
	Doc string
}

// TraceMembers lists the members of x-vsr-trace, highest priority first.
func TraceMembers() []TraceMember { return slices.Clone(traceMembers) }

var traceMembers = []TraceMember{
	{TraceConfigHash, "benchmark", "SHA-256 of the runtime document that served a config-bound evaluation request."},
	{TraceInferenceCalls, "benchmark", "Provider calls a config-bound evaluation request made."},
	{TraceModelUsage, "benchmark", "Per-model usage receipt of a config-bound evaluation request, as JSON."},
	{TraceConfidence, "routing", "Model-derived score of the selected decision, such as `0.91`."},
	{TraceAlgorithm, "routing", "Selector or executor that chose the model after the decision matched, such as `static`."},
	{TraceRoutingLatencyMs, "routing", "Time spent choosing the model, in milliseconds."},
	{TraceFallbackAttempts, "fallback", "Attempts cross-model fallback made, the primary included, when a fallback candidate served the response."},
	{TraceOnUnknown, "routing", "Decisions whose unknown result `rules.on_unknown` resolved, each with its `policy`."},
	{TraceClientProtocol, "protocol", "Client wire protocol, such as `anthropic`."},
	{TraceUpstreamProtocol, "protocol", "Provider wire protocol, such as `openai`."},
	{TraceEffectiveInputTokens, "automatic_output", "Rendered input tokens of the final automatic-output dispatch, chat template included."},
	{TraceEffectiveMaxOutputTokens, "automatic_output", "Output token limit of the final automatic-output dispatch, reasoning included."},
	{TraceSignals, "signals", "Matched signal rules as `type:name`, and `type:rule:option` for a choice rule without labels; projections are `projection:name`."},
	{TraceContextTokens, "signals", "Token count the `context` signal measured."},
	{TraceCategory, "classification", "Domain category the classifier chose, such as `math`."},
	{TraceReasoning, "classification", "Reasoning mode chosen for the request: `on` or `off`."},
	{TraceModality, "routing", "Output kinds of the selected decision's output contract, such as `(image)` or `(text image)`; omitted for a decision that declares none."},
	{TraceSystemPrompt, "system_prompt", "Present when a `system_prompt` plugin injected text into the request."},
	{TraceCacheSimilarity, "response_cache", "Similarity score of the response-cache lookup."},
	{TraceRetention, "retention", "The matched decision's retention directive: the `drop`, `ttl_turns`, `keep_current_model` and `prefer_prefix` it sets."},
	{TracePromptCache, "prompt_cache", "Outcome of the `prompt_cache` plugin, with its `reason` and the `inserted` and `preserved` marker counts."},
	{TraceTools, "tools", "Outcome of the `tools` plugin's selection, with its `candidates`, `kept` and `latency_ms`."},
	{TraceResponseGuard, "response_guard", "Outcome of the selected decision's `response_guard`, such as `blocked`, with `buffered` when the Router held a streamed answer until the guard read it."},
	{TraceSessionPhase, "router_learning", "Protection phase: `user_turn`, `tool_loop` or `provider_state`."},
	{TraceLearning, "router_learning", "Router Learning methods, each with its `action`, `scope` and `reason`."},
	{TraceLooper, "looper", "Models the Looper called, with its `algorithm`, final `model`, `iterations`, `latency_ms` and token counts."},
	{TraceRanking, "routing", "The matched decision the selected one beat, with the `reason` that settled it."},
	{TraceTruncated, "response_contract", "Present when members were left out to keep the trace within 4 KB."},
}

var traceMemberKeys = func() map[string]bool {
	keys := make(map[string]bool, len(traceMembers))
	for _, member := range traceMembers {
		keys[member.Key] = true
	}
	return keys
}()

// Trace holds the members of one x-vsr-trace value.
type Trace struct {
	members map[string]Item
}

// Set records a member. Only registered members are encoded.
func (t *Trace) Set(key string, value Item) {
	if !traceMemberKeys[key] || key == TraceTruncated {
		return
	}
	if t.members == nil {
		t.members = make(map[string]Item)
	}
	t.members[key] = value
}

// Encode serializes the members in priority order as an RFC 8941 dictionary,
// dropping the lowest-priority ones while the value exceeds TraceLimitBytes.
// An empty trace encodes as "".
func (t *Trace) Encode() string {
	parts := make([]string, 0, len(t.members))
	for _, member := range traceMembers {
		if value, ok := t.members[member.Key]; ok {
			parts = append(parts, value.member(member.Key))
		}
	}
	kept := len(parts)
	for kept > 0 && encodedTraceSize(parts, kept) > TraceLimitBytes {
		kept--
	}
	if kept == len(parts) {
		return strings.Join(parts, ", ")
	}
	return strings.Join(append(parts[:kept:kept], TraceTruncated), ", ")
}

// encodedTraceSize is the length of the first kept parts joined, with the
// truncation marker when parts are left out.
func encodedTraceSize(parts []string, kept int) int {
	size, count := 0, kept
	for _, part := range parts[:kept] {
		size += len(part)
	}
	if kept < len(parts) {
		size += len(TraceTruncated)
		count++
	}
	if count > 1 {
		size += len(", ") * (count - 1)
	}
	return size
}
