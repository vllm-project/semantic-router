package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// Cache detail fields are optional provider evidence. In particular, vLLM can
// omit them when usage details are disabled; absence must not invent a zero.
func decodeInputCacheUsage(usage *llmprotocol.Usage, cached, written *int64, aliases ...*int64) error {
	for _, alias := range aliases {
		if alias == nil {
			continue
		}
		if written != nil && *written != *alias {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "conflicting_cache_usage", "upstream cache-write token aliases disagree", nil)
		}
		written = alias
	}
	usage.InputCacheRead = optionalAuthoritative(cached)
	usage.InputCacheWrite = optionalAuthoritative(written)
	usage.InputUncached = unknownCount()
	if usage.InputTotal.Value != nil && cached != nil && written != nil {
		uncached := int64(-1)
		total := *usage.InputTotal.Value
		if *cached >= 0 && *written >= 0 && total >= *cached && *written <= total-*cached {
			uncached = total - *cached - *written
		}
		usage.InputUncached = llmprotocol.TokenCount{Value: llmprotocol.Int64(uncached), Provenance: llmprotocol.UsageDerived}
	}
	return nil
}

func optionalAuthoritative(value *int64) llmprotocol.TokenCount {
	if value == nil {
		return unknownCount()
	}
	return authoritative(*value)
}

// appendAnthropicUsageMarks records the usage projection caveats on the
// Anthropic Messages surfaces that project final numbers: the buffered
// response and the terminal message_delta. The streaming message_start
// carries a provisional view and is not marked. Each caveat is an
// approximation diagnostic because Messages requires exact numbers.
func appendAnthropicUsageMarks(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source llmprotocol.WireFormat,
	usage llmprotocol.Usage,
) {
	if usageUnavailable(usage) {
		appendDiagnostic(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage", llmprotocol.DiagnosticApproximated,
			"Messages requires usage; emitted an explicit zero-valued usage object")
		return
	}
	if anthropicOutputTotalIsLowerBound(usage) {
		appendDiagnostic(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage", llmprotocol.DiagnosticApproximated,
			"output total is incomplete; the known output components project as a lower bound")
	}
	if anthropicInputTotalIsLowerBound(usage) {
		appendDiagnostic(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage", llmprotocol.DiagnosticApproximated,
			"input total and uncached count are absent; the projected input count is a lower bound")
	}
}

// appendDiagnostic appends one bounded diagnostic with an explicit action.
func appendDiagnostic(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source, target llmprotocol.WireFormat,
	field string,
	action llmprotocol.DiagnosticAction,
	reason string,
) {
	*diagnostics = appendDiagnostics(*diagnostics, llmprotocol.Diagnostics{{
		Source: source, Target: target, Field: field,
		Action: action, Reason: reason,
	}}, policy.Limits.Diagnostics)
}

func appendAnthropicPartialCacheOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	readKnown, writeKnown := usage.InputCacheRead.Value != nil, usage.InputCacheWrite.Value != nil
	if readKnown != writeKnown {
		appendAccountingOmission(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage.cache", "Messages requires numeric cache buckets; the unreported bucket is zero-filled for representation, while settlement retains unknown usage")
	}
}
