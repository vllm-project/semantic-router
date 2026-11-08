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

func decodeAnthropicCacheCreationTiers(
	creation *anthropicCacheCreationUsageWire,
	aggregate int64,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) (cacheWrite5m, cacheWrite1h llmprotocol.TokenCount, derivedAggregate *int64) {
	if creation == nil {
		return unknownCount(), unknownCount(), nil
	}
	has5m := creation.Ephemeral5mInputTokens != nil
	has1h := creation.Ephemeral1hInputTokens != nil
	if !has5m && !has1h {
		return unknownCount(), unknownCount(), nil
	}

	val5m := int64(0)
	if has5m {
		val5m = *creation.Ephemeral5mInputTokens
	}
	val1h := int64(0)
	if has1h {
		val1h = *creation.Ephemeral1hInputTokens
	}

	if aggregate == 0 && (val5m > 0 || val1h > 0) {
		derived := val5m + val1h
		derivedAggregate = &derived
		aggregate = derived
	}

	if has5m && has1h {
		tierSum := val5m + val1h
		if tierSum != aggregate {
			if diagnostics != nil {
				appendProviderFieldOmission(diagnostics, policy, llmprotocol.AnthropicMessagesV1,
					"usage.cache_creation", "tier token counts do not match aggregate cache_creation_input_tokens")
			}
			return unknownCount(), unknownCount(), derivedAggregate
		}
		return authoritative(val5m), authoritative(val1h), derivedAggregate
	}

	if has5m && !has1h {
		if val5m > aggregate {
			if diagnostics != nil {
				appendProviderFieldOmission(diagnostics, policy, llmprotocol.AnthropicMessagesV1,
					"usage.cache_creation", "tier token count exceeds aggregate cache_creation_input_tokens")
			}
			return unknownCount(), unknownCount(), derivedAggregate
		}
		return authoritative(val5m), unknownCount(), derivedAggregate
	}

	if has1h && !has5m {
		if val1h > aggregate {
			if diagnostics != nil {
				appendProviderFieldOmission(diagnostics, policy, llmprotocol.AnthropicMessagesV1,
					"usage.cache_creation", "tier token count exceeds aggregate cache_creation_input_tokens")
			}
			return unknownCount(), unknownCount(), derivedAggregate
		}
		return unknownCount(), authoritative(val1h), derivedAggregate
	}

	return unknownCount(), unknownCount(), derivedAggregate
}

func appendAnthropicTierSplitOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	if usage.InputCacheWrite.Value != nil && *usage.InputCacheWrite.Value > 0 && usage.InputCacheWrite5m.Value == nil && usage.InputCacheWrite1h.Value == nil {
		appendAccountingOmission(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage.cache_creation", "tier token split is unknown; aggregate cache creation tokens are encoded as ephemeral_5m")
	}
}

func appendChatCacheCreationOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	if usage.InputCacheWrite5m.Value != nil || usage.InputCacheWrite1h.Value != nil {
		appendAccountingOmission(diagnostics, policy, source, llmprotocol.OpenAIChatV1,
			"usage.cache_creation", "Chat Completions does not support prompt cache write retention tiers")
	}
}

func appendResponsesCacheCreationOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	if usage.InputCacheWrite5m.Value != nil || usage.InputCacheWrite1h.Value != nil {
		appendAccountingOmission(diagnostics, policy, source, llmprotocol.OpenAIResponsesV1,
			"usage.cache_creation", "Responses does not support prompt cache write retention tiers")
	}
}

func appendAnthropicPartialCacheOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	readKnown, writeKnown := usage.InputCacheRead.Value != nil, usage.InputCacheWrite.Value != nil
	if readKnown != writeKnown {
		appendAccountingOmission(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage.cache", "Messages requires numeric cache buckets; the unreported bucket is zero-filled for representation, while settlement retains unknown usage")
	}
}

// The output total is the Messages client's single output number. When it is
// unreported while an output component (reasoning, other) is known, the wire
// zero-fills the total beside that component, so the omission must be named
// rather than presented as an exact zero.
func appendAnthropicPartialOutputOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	if usage.OutputTotal.Value != nil {
		return
	}
	if usage.OutputReasoning.Value == nil && usage.OutputOther.Value == nil {
		return
	}
	appendAccountingOmission(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
		"usage.output", "Messages requires a numeric output total; the unreported total is zero-filled for representation, while settlement retains unknown usage")
}
