package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// Cache detail fields are optional provider evidence. In particular, vLLM can
// omit them when usage details are disabled; absence must not invent a zero.
func decodeInputCacheUsage(usage *llmprotocol.Usage, cached, written, created *int64) error {
	if written != nil && created != nil && *written != *created {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "conflicting_cache_usage", "upstream cache_write_tokens and created_cache_tokens disagree", nil)
	}
	if written == nil {
		written = created
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

func appendAnthropicPartialCacheOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	readKnown, writeKnown := usage.InputCacheRead.Value != nil, usage.InputCacheWrite.Value != nil
	if readKnown != writeKnown {
		appendAccountingOmission(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage.cache", "Messages requires numeric cache buckets; the unreported bucket is zero-filled for representation, while settlement retains unknown usage")
	}
}
