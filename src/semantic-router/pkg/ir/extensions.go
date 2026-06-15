package ir

// IRExtensions is the sidecar envelope attached to a request context when the
// inbound payload carries fields that have no representation in
// *openai.ChatCompletionNewParams.
//
// Inbound parsers populate it; outbound emitters and plugins read from it;
// nil means the request is plain OpenAI-shape and no extensions exist.
//
// Block IDs (the keys in CacheControl and ThinkingSignatures) are stable
// JSON-paths derived from the original request body (e.g. "system.0",
// "messages[3].content[1]") so plugins can mutate the IR core without
// invalidating sidecar lookups.
type IRExtensions struct {
	// SourceProtocol mirrors RequestContext.ClientProtocol at parse time
	// (e.g. "anthropic"). Stamped by the inbound parser so downstream
	// consumers do not need to look at the context separately.
	SourceProtocol string

	// SystemBlocks preserves array-form system prompts when the inbound
	// shape was a TextBlockParam[] rather than a plain string, so the
	// outbound emitter can reconstruct the multi-block form.
	SystemBlocks []SystemBlock

	// CacheControl captures cache_control markers from any block in the
	// request, keyed by stable block ID. The marked block itself is still
	// translated into the OpenAI IR; this map exists so the emitter can
	// re-attach the marker on the outbound side.
	CacheControl map[string]CacheControlSpec

	// Thinking captures the extended-thinking directive
	// (type/budget/display). nil means the inbound request did not request
	// extended thinking.
	Thinking *ThinkingSpec

	// MetadataUserID mirrors metadata.user_id from the Anthropic request.
	// Also mirrored into params.User for plugins that key on the OpenAI
	// user field.
	MetadataUserID string

	// TopK preserves the Anthropic top_k sampling parameter. There is no
	// OpenAI equivalent; OpenAI-backend emitters should append a Warning
	// and drop it.
	TopK *int64

	// ServerTools captures server-side tool definitions (web_search,
	// bash, code_execution, text_editor). The emitter decides whether the
	// backend supports them; OpenAI emitters warn-and-drop.
	ServerTools []ServerToolSpec

	// ToolChoiceDisableParallel mirrors
	// tool_choice.disable_parallel_tool_use. False is both the zero value
	// and the protocol default.
	ToolChoiceDisableParallel bool

	// ToolStrict captures the per-tool strict-schema flag
	// (tools[].strict), keyed by function name.
	ToolStrict map[string]bool

	// Documents captures Anthropic-native document blocks. OpenAI
	// emitters warn-and-drop.
	Documents []DocumentBlock

	// CitationsEnabled mirrors any document.citations.enabled=true seen
	// on a document block.
	CitationsEnabled bool

	// ThinkingSignatures stores opaque per-thinking-block signatures that
	// must round-trip for multi-turn extended-thinking continuity, keyed
	// by stable block ID.
	ThinkingSignatures map[string]string

	// AnthropicBeta captures the raw anthropic-beta header value, if
	// present, for later header-pass-through work. The parser only
	// records; it does not act on the value.
	AnthropicBeta string

	// InboundAnthropicVersion captures the inbound anthropic-version
	// header value on Anthropic-shape ingress so the body-phase routing
	// step can layer it under the provider-profile pin when the profile
	// did not supply one.
	InboundAnthropicVersion string

	// InboundAnthropicBeta captures the inbound anthropic-beta header
	// value on Anthropic-shape ingress for header pass-through. Duplicates
	// AnthropicBeta from the body parser path on purpose: the header is
	// captured at the request-header phase before the body has been seen.
	InboundAnthropicBeta string

	// InboundDangerousDirectBrowserAccess captures the
	// anthropic-dangerous-direct-browser-access header value on Anthropic
	// ingress for pass-through to compatible backends.
	InboundDangerousDirectBrowserAccess string

	// CacheReadInputTokens mirrors anthropic.Usage.CacheReadInputTokens
	// from the upstream response. Captured by the Anthropic→OpenAI
	// normalization step so the symmetric Anthropic outbound emitter can
	// replay it. Zero for OpenAI backends.
	CacheReadInputTokens int64

	// CacheCreationInputTokens mirrors
	// anthropic.Usage.CacheCreationInputTokens. Same capture/replay
	// trajectory as CacheReadInputTokens.
	CacheCreationInputTokens int64

	// Ephemeral5mInputTokens mirrors
	// anthropic.Usage.CacheCreation.Ephemeral5mInputTokens, the 5-minute
	// TTL slice of the cache-creation breakdown. Zero when the upstream
	// did not populate the per-TTL breakdown.
	Ephemeral5mInputTokens int64

	// Ephemeral1hInputTokens mirrors
	// anthropic.Usage.CacheCreation.Ephemeral1hInputTokens, the 1-hour
	// TTL slice of the cache-creation breakdown.
	Ephemeral1hInputTokens int64

	// ServerToolUseCounts mirrors anthropic.Usage.ServerToolUse counters
	// keyed by tool name ("web_search", "web_fetch"). Empty when the
	// upstream returned no server-tool invocations or when the backend is
	// OpenAI.
	ServerToolUseCounts map[string]int64

	// AnthropicStopReason captures the upstream Anthropic stop_reason when
	// it is one of the Anthropic-only values that has no OpenAI
	// equivalent ("pause_turn", "refusal"). When non-empty it overrides
	// the OpenAI-derived stop_reason on the outbound emit path. Empty for
	// stop_reasons that round-trip cleanly through OpenAI finish_reason.
	AnthropicStopReason string

	// AnthropicStopSequence mirrors anthropic.Message.StopSequence and is
	// non-empty only when stop_reason == "stop_sequence". OpenAI
	// finish_reason "length"/"stop" does not surface which sequence
	// matched, so this is populated only on Anthropic-backend cells.
	AnthropicStopSequence string

	// Warnings accumulates parse- and emit-time observations. Surfaced via
	// response headers, structured logging, and metrics.
	Warnings []Warning
}

// AppendWarning is nil-safe: when e is nil the call is a no-op, so callers
// in OpenAI-only code paths (where IRExtensions is nil) do not need to
// guard.
func (e *IRExtensions) AppendWarning(w Warning) {
	if e == nil {
		return
	}
	e.Warnings = append(e.Warnings, w)
}

// SetCacheControl is nil-safe and lazily initializes the underlying map.
func (e *IRExtensions) SetCacheControl(blockID string, spec CacheControlSpec) {
	if e == nil {
		return
	}
	if e.CacheControl == nil {
		e.CacheControl = make(map[string]CacheControlSpec)
	}
	e.CacheControl[blockID] = spec
}

// SetThinkingSignature is nil-safe and lazily initializes the underlying map.
func (e *IRExtensions) SetThinkingSignature(blockID, signature string) {
	if e == nil {
		return
	}
	if e.ThinkingSignatures == nil {
		e.ThinkingSignatures = make(map[string]string)
	}
	e.ThinkingSignatures[blockID] = signature
}

// SetToolStrict is nil-safe and lazily initializes the underlying map.
func (e *IRExtensions) SetToolStrict(toolName string, strict bool) {
	if e == nil {
		return
	}
	if e.ToolStrict == nil {
		e.ToolStrict = make(map[string]bool)
	}
	e.ToolStrict[toolName] = strict
}

// SetServerToolUseCount is nil-safe and lazily initializes the underlying
// map. Zero counts are recorded as-is so callers can distinguish "tool not
// invoked" (absent key) from "tool invoked zero times" (rare; present key
// with zero value).
func (e *IRExtensions) SetServerToolUseCount(toolName string, count int64) {
	if e == nil {
		return
	}
	if e.ServerToolUseCounts == nil {
		e.ServerToolUseCounts = make(map[string]int64)
	}
	e.ServerToolUseCounts[toolName] = count
}

// SystemBlock records the shape of one Anthropic system block. When the
// inbound system field was a plain string, no SystemBlocks are emitted —
// only when the inbound used the TextBlockParam[] form.
type SystemBlock struct {
	BlockID      string
	Text         string
	CacheControl *CacheControlSpec
}

// CacheControlSpec mirrors anthropic.CacheControlEphemeralParam. Type is
// always "ephemeral" today (the only Anthropic-supported value); TTL is one
// of "5m", "1h", or "" (use the model default).
type CacheControlSpec struct {
	Type string
	TTL  string
}

// ThinkingSpec mirrors the Anthropic thinking-config object. Display
// distinguishes summarized vs omitted reasoning output; empty means the
// inbound did not pin a display mode.
type ThinkingSpec struct {
	Type         string
	Display      string
	BudgetTokens int64
}

// ServerToolSpec captures a server-side tool definition. Parameters is the
// raw, untyped tool-config map so the outbound emitter can decide what to
// send to a server-tool-capable backend.
type ServerToolSpec struct {
	Type       string
	Name       string
	Parameters map[string]any
}

// DocumentBlock captures one Anthropic document block. SourceType is one
// of "base64", "url", "text", "content", "file"; MediaType is the MIME
// type when known; Data is the source payload (URL, base64 bytes, or
// inline text) keyed by SourceType. InlineContent holds the parsed
// sub-blocks when SourceType is "content" (custom-chunked documents).
type DocumentBlock struct {
	BlockID       string
	SourceType    string
	MediaType     string
	Data          string
	Citations     bool
	CacheControl  *CacheControlSpec
	InlineContent []SystemBlock
}
