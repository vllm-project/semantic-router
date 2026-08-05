package ir

// WarningReason is the stable machine token surfaced in the
// x-vsr-protocol-warnings response header and in the
// vsr_translation_lossy_total metric "reason" label.
//
// Stability contract: existing reason tokens are part of the wire/metric
// contract. New reasons may be added freely (clients tolerate unknown
// values); renames are breaking and require a CHANGELOG entry.
type WarningReason string

const (
	// ReasonDropped indicates a field was discarded with no equivalent on
	// the outbound side.
	ReasonDropped WarningReason = "dropped"

	// ReasonCoercedString indicates a structured value was flattened to a
	// string (e.g. tool_choice option object collapsed to a string).
	ReasonCoercedString WarningReason = "coerced_string"

	// ReasonUnsupportedBlockType indicates a content block whose type is
	// not representable in the OpenAI IR was dropped (e.g.
	// redacted_thinking).
	ReasonUnsupportedBlockType WarningReason = "unsupported_block_type"

	// ReasonUnsupportedSubtype indicates a sub-field of an otherwise
	// supported block could not be translated (e.g. document source
	// variant unknown to the emitter).
	ReasonUnsupportedSubtype WarningReason = "unsupported_subtype"

	// ReasonImageFileIDUnresolved indicates an Anthropic image-by-file_id
	// reference could not be resolved against the Files API and was
	// dropped.
	ReasonImageFileIDUnresolved WarningReason = "image_file_id_unresolved"

	// ReasonRedactedThinkingDropped indicates a redacted_thinking block
	// was dropped because the OpenAI IR has no equivalent.
	ReasonRedactedThinkingDropped WarningReason = "redacted_thinking_dropped"

	// ReasonSignatureDropOnOpenAIBackend indicates a thinking-block
	// signature was discarded because the outbound OpenAI backend cannot
	// round-trip it.
	ReasonSignatureDropOnOpenAIBackend WarningReason = "signature_drop_on_openai_backend"

	// ReasonServerToolDropOnOpenAIBackend indicates an Anthropic
	// server-side tool (web_search, bash, etc.) was dropped because the
	// OpenAI backend does not expose an equivalent.
	ReasonServerToolDropOnOpenAIBackend WarningReason = "server_tool_drop_on_openai_backend"

	// ReasonTopKDropOnOpenAIBackend indicates the Anthropic top_k
	// sampling parameter was dropped on an OpenAI backend (no
	// equivalent).
	ReasonTopKDropOnOpenAIBackend WarningReason = "top_k_drop_on_openai_backend"

	// ReasonBetaDropOnOpenAIBackend indicates the anthropic-beta header
	// was dropped on an OpenAI backend (no equivalent).
	ReasonBetaDropOnOpenAIBackend WarningReason = "beta_drop_on_openai_backend"

	// ReasonWarningsTruncated is the synthetic trailer emitted by the
	// response-header builder when the encoded warnings list exceeds the
	// header size limit. The associated field carries the count of
	// dropped entries.
	ReasonWarningsTruncated WarningReason = "warnings_truncated"

	// ReasonCacheFieldsAbsent indicates an Anthropic-source response had all
	// cache usage counters at zero. This covers both the case where the user
	// did not annotate any prompt segment with cache_control (no cache was
	// requested) and the case where an OpenAI backend was used (no cache
	// support). Info-level so it does not surface as a lossy warning.
	ReasonCacheFieldsAbsent WarningReason = "cache_fields_absent"

	// ReasonAnthropicStopReasonCoerced indicates an OpenAI finish_reason
	// could not be mapped to a known Anthropic stop_reason and was
	// coerced to "end_turn". Surfaced for diagnostics so clients can
	// detect unexpected upstream behavior.
	ReasonAnthropicStopReasonCoerced WarningReason = "anthropic_stop_reason_coerced"
)
