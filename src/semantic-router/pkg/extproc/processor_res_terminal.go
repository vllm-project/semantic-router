package extproc

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// takeNeutralResponseUsage consumes the one semantic terminal emitted by
// BackendInvoker for the selected physical dispatch. No public response body
// or SSE frame is parsed on this accounting path.
func (r *OpenAIRouter) takeNeutralResponseUsage(ctx *RequestContext) responseUsageMetrics {
	if r == nil || r.ResponseTerminals == nil || ctx == nil || ctx.DispatchState == nil {
		return invalidResponseTerminalUsage("response_terminal_missing")
	}
	state := ctx.DispatchState
	state.mu.Lock()
	dispatchID := state.selectedDispatchID
	if dispatchID == "" {
		dispatchID = state.primaryDispatchID
	}
	state.mu.Unlock()
	reference, referenceFound := responseTerminalReference(state, dispatchID)
	if !referenceFound {
		return invalidResponseTerminalUsage("response_terminal_missing")
	}
	takeContext := ctx.TraceContext
	if takeContext == nil {
		takeContext = context.Background()
	}
	takeContext = context.WithoutCancel(takeContext)
	record, found, err := r.ResponseTerminals.Take(takeContext, reference)
	if err != nil {
		return invalidResponseTerminalUsage(responseTerminalFailureReason(err))
	}
	if !found {
		return invalidResponseTerminalUsage("response_terminal_missing")
	}
	if record.Reference != reference {
		return invalidResponseTerminalUsage("response_terminal_invalid")
	}
	return responseUsageFromTerminal(record)
}

func invalidResponseTerminalUsage(reason string) responseUsageMetrics {
	return responseUsageMetrics{invalid: true, invalidReason: reason}
}

func responseTerminalFailureReason(err error) string {
	switch {
	case errors.Is(err, backendinvoker.ErrResponseTerminalCapacity):
		return "response_terminal_capacity"
	case errors.Is(err, backendinvoker.ErrResponseTerminalInvalid):
		return "response_terminal_invalid"
	case errors.Is(err, backendinvoker.ErrResponseTerminalUnavailable):
		return "response_terminal_unavailable"
	default:
		return "response_terminal_unavailable"
	}
}

func responseUsageFromTerminal(record backendinvoker.ResponseTerminalRecord) responseUsageMetrics {
	terminal := record.Terminal
	if terminal.Error != nil {
		if record.Attempt.State == backendinvoker.AttemptKnownZero {
			return responseUsageMetrics{
				promptTokensReported: true, completionTokensReported: true,
				totalTokensReported: true, cachedPromptTokensReported: true,
				cacheWriteTokensReported: true,
			}
		}
		return invalidResponseTerminalUsage("response_terminal_invalid")
	}
	return responseUsageFromSemanticUsage(terminal.Usage)
}

// responseUsageFromSemanticUsage converts neutral accounting evidence into the
// metrics seam used by quota, telemetry, replay, and cache-hit accounting. It
// deliberately accepts only authoritative component counts (plus a provable
// derived total); estimated client-visible values are never charged as actual.
func responseUsageFromSemanticUsage(usage llmprotocol.Usage) responseUsageMetrics {
	if usage.State != llmprotocol.UsageAvailable {
		return invalidResponseTerminalUsage("response_terminal_invalid")
	}
	input, inputReported, inputOK := authoritativeTerminalTokenCount(usage.InputTotal)
	output, outputReported, outputOK := authoritativeTerminalTokenCount(usage.OutputTotal)
	total, totalReported, totalOK := settlementTotalTokenCount(usage, input, output)
	cacheRead, cacheReadReported, cacheReadOK := authoritativeTerminalTokenCount(usage.InputCacheRead)
	cacheWrite, cacheWriteReported, cacheWriteOK := authoritativeTerminalTokenCount(usage.InputCacheWrite)
	if !inputOK || !outputOK || !totalOK || !cacheReadOK || !cacheWriteOK ||
		!inputReported || !outputReported {
		return invalidResponseTerminalUsage("response_terminal_invalid")
	}
	return normalizeResponseUsage(responseUsageMetrics{
		promptTokens: input, promptTokensReported: inputReported,
		completionTokens: output, completionTokensReported: outputReported,
		totalTokens: total, totalTokensReported: totalReported,
		cachedPromptTokens: cacheRead, cachedPromptTokensReported: cacheReadReported,
		cacheWriteTokens: cacheWrite, cacheWriteTokensReported: cacheWriteReported,
	})
}

func authoritativeTerminalTokenCount(count llmprotocol.TokenCount) (int, bool, bool) {
	if count.Value == nil {
		return 0, false, count.Provenance == "" || count.Provenance == llmprotocol.UsageUnknown
	}
	if count.Provenance != llmprotocol.UsageAuthoritative {
		return 0, true, false
	}
	value := *count.Value
	maximum := int64(^uint(0) >> 1)
	if value < 0 || value > maximum {
		return 0, true, false
	}
	return int(value), true, true
}

// settlementTotalTokenCount accepts either an authoritative terminal total or
// the one derived value that is independently provable from authoritative
// input and output totals. Estimated or partially derived evidence is never
// charged as actual usage.
func settlementTotalTokenCount(usage llmprotocol.Usage, input, output int) (int, bool, bool) {
	if usage.Total.Value == nil {
		if input > int(^uint(0)>>1)-output {
			return 0, false, false
		}
		return input + output, true, true
	}
	value := *usage.Total.Value
	maximum := int64(^uint(0) >> 1)
	if value < 0 || value > maximum {
		return 0, true, false
	}
	switch usage.Total.Provenance {
	case llmprotocol.UsageAuthoritative:
		return int(value), true, true
	case llmprotocol.UsageDerived:
		if input > int(^uint(0)>>1)-output || int(value) != input+output {
			return 0, true, false
		}
		return int(value), true, true
	default:
		return 0, true, false
	}
}
