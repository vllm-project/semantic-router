package protocolcodec

import (
	"context"
	"errors"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (state *streamState) completeItem(event llmprotocol.Event) (llmprotocol.Event, error) {
	event, err := state.prepareItemCompletion(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	if err := state.validateCompletedItemContent(event); err != nil {
		return llmprotocol.Event{}, err
	}
	state.markItemComplete(event.ItemIndex)
	return event, nil
}

func (state *streamState) prepareItemCompletion(event llmprotocol.Event) (llmprotocol.Event, error) {
	if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream completed an inactive output item", nil)
	}
	expectedItemID := state.itemIDs[event.ItemIndex]
	if event.ItemID != "" && event.ItemID != expectedItemID {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id_mismatch", "upstream completion changed its output item ID", nil)
	}
	event.ItemID = expectedItemID
	if state.itemKinds[event.ItemIndex] == llmprotocol.ContentToolCall {
		completed, err := state.completeToolItem(event)
		if err != nil {
			return llmprotocol.Event{}, err
		}
		event = completed
	} else if event.Content == nil && state.itemKinds[event.ItemIndex] != "" {
		event.Content = &llmprotocol.Content{Kind: state.itemKinds[event.ItemIndex]}
	}
	return event, nil
}

func (state *streamState) validateCompletedItemContent(event llmprotocol.Event) error {
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentGeneratedImage {
		if err := llmprotocol.ValidateGeneratedImage(event.Content.GeneratedImage, state.policy.Limits); err != nil {
			return upstreamSemanticValidationError(err)
		}
		if err := state.validateCompletedGeneratedImage(event.ItemIndex, event.Content.GeneratedImage); err != nil {
			return err
		}
	}
	return nil
}

func (state *streamState) completeToolItem(event llmprotocol.Event) (llmprotocol.Event, error) {
	arguments, err := state.finalToolArguments(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	call := state.toolCalls[event.ItemIndex]
	if event.ToolCall != nil {
		call, err = state.mergeStreamToolIdentity(call, *event.ToolCall)
		if err != nil {
			return llmprotocol.Event{}, err
		}
	}
	if err := state.validateStreamToolIdentity(call, true); err != nil {
		return llmprotocol.Event{}, err
	}
	if err := state.claimStreamToolCallID(event.ItemIndex, call.ID); err != nil {
		return llmprotocol.Event{}, err
	}
	call.Arguments = string(arguments)
	event.ToolCall = &call
	return event, nil
}

func (state *streamState) mergeStreamToolIdentity(current, incoming llmprotocol.ToolCall) (llmprotocol.ToolCall, error) {
	if err := state.validateStreamToolIdentity(incoming, false); err != nil {
		return llmprotocol.ToolCall{}, err
	}
	if incoming.ID != "" {
		if current.ID != "" && current.ID != incoming.ID {
			return llmprotocol.ToolCall{}, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_tool_identity_mismatch",
				"upstream stream changed a tool call ID",
				nil,
			)
		}
		current.ID = incoming.ID
	}
	if incoming.Name != "" {
		if current.Name != "" && current.Name != incoming.Name {
			return llmprotocol.ToolCall{}, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_tool_identity_mismatch",
				"upstream stream changed a tool name",
				nil,
			)
		}
		current.Name = incoming.Name
	}
	return current, nil
}

func (state *streamState) claimStreamToolCallID(itemIndex int, callID string) error {
	if callID == "" {
		return nil
	}
	if index, duplicate := state.toolCallIndexes[callID]; duplicate && index != itemIndex {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"duplicate_stream_tool_call_id",
			"upstream stream reused a tool call ID",
			nil,
		)
	}
	state.toolCallIndexes[callID] = itemIndex
	return nil
}

func (state *streamState) validateStreamToolIdentity(call llmprotocol.ToolCall, required bool) error {
	if len(call.ID) > state.policy.Limits.IdentifierBytes || len(call.Name) > state.policy.Limits.ToolNameBytes {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_tool_identity_limit",
			"upstream streamed tool identity exceeds the configured limit",
			nil,
		)
	}
	if required && (strings.TrimSpace(call.ID) == "" || strings.TrimSpace(call.Name) == "") {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_tool_identity_required",
			"upstream streamed tool call requires an ID and name",
			nil,
		)
	}
	return nil
}

func (state *streamState) validateStreamToolArgumentAppend(current []byte, incoming string) error {
	for _, limit := range []int{
		state.policy.Limits.UnfinishedArguments,
		state.policy.Limits.ToolArgumentsBytes,
	} {
		if limit > 0 && (len(current) > limit || len(incoming) > limit-len(current)) {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"tool_arguments_limit",
				"streamed tool arguments exceed the configured limit",
				nil,
			)
		}
	}
	return nil
}

func (state *streamState) finalToolArguments(event llmprotocol.Event) ([]byte, error) {
	arguments := state.toolArguments[event.ItemIndex]
	if event.ToolCall != nil && event.ToolCall.Arguments != "" {
		if err := state.validateStreamToolArgumentAppend(nil, event.ToolCall.Arguments); err != nil {
			return nil, err
		}
		if len(arguments) > 0 && string(arguments) != event.ToolCall.Arguments {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_mismatch", "upstream final tool arguments do not match streamed arguments", nil)
		}
		arguments = []byte(event.ToolCall.Arguments)
	}
	if !isJSONObject(arguments, state.policy.Limits.JSONDepth) {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_tool_arguments", "upstream streamed tool arguments are not a JSON object", nil)
	}
	return arguments, nil
}

func (state *streamState) markItemComplete(itemIndex int) {
	state.completedItems[itemIndex] = true
	for key := range state.itemTextBytes {
		if key.item != itemIndex {
			continue
		}
		delete(state.itemTextBytes, key)
		delete(state.itemTextRunes, key)
		delete(state.itemCitations, key)
	}
	delete(state.toolArguments, itemIndex)
}

func (state *streamState) applyEventEvidence(event llmprotocol.Event) (llmprotocol.Event, error) {
	prepared, err := state.applyUsageEvidence(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	event = prepared
	if event.StopReason != "" {
		state.stop = event.StopReason
	}
	if event.Type != llmprotocol.EventResponseCompleted && event.Type != llmprotocol.EventResponseFailed {
		return event, nil
	}
	return state.applyTerminalEvent(event)
}

func (state *streamState) applyUsageEvidence(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.Usage == nil {
		return event, nil
	}
	merged, err := mergeMonotonicUsage(state.usage, *event.Usage)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	state.usage = merged
	usage := state.usage
	event.Usage = &usage
	return event, nil
}

func (state *streamState) applyTerminalEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	var err error
	if event.Type == llmprotocol.EventResponseCompleted {
		event, err = state.validateCompletedEvent(event)
	} else {
		if event.Usage == nil && state.usage.State == llmprotocol.UsageAvailable {
			usage := state.usage
			event.Usage = &usage
		}
		event, err = validateFailedEvent(event)
	}
	if err != nil {
		return llmprotocol.Event{}, err
	}
	state.terminal = true
	return event, nil
}

func (state *streamState) validateCompletedEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if err := state.validateCompletedLifecycle(event); err != nil {
		return llmprotocol.Event{}, err
	}
	event, err := validateCompletedStopReason(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	if event.Usage == nil {
		usage := state.usage
		event.Usage = &usage
	}
	if err := llmprotocol.ValidateUsage(*event.Usage); err != nil {
		return llmprotocol.Event{}, err
	}
	return event, nil
}

func (state *streamState) validateCompletedLifecycle(event llmprotocol.Event) error {
	if !state.started {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream completed before response start", nil)
	}
	if len(state.items) == 0 && event.StopReason != llmprotocol.StopContentFilter {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_output_missing", "upstream stream completed without output", nil)
	}
	for itemIndex := range state.items {
		if !state.completedItems[itemIndex] {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_incomplete", "upstream stream completed with an active output item", nil)
		}
	}
	if len(state.toolArguments) != 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_incomplete", "upstream stream completed with unfinished tool arguments", nil)
	}
	if event.Error != nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_terminal_shape", "completed stream cannot contain an error", nil)
	}
	return nil
}

func validateCompletedStopReason(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.StopReason == "" {
		event.StopReason = llmprotocol.StopUnknown
	}
	if event.StopReason == llmprotocol.StopSequence && event.MatchedStopSequence == "" {
		return llmprotocol.Event{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"matched_stop_sequence_required",
			"upstream stop_sequence reason requires the matched sequence",
			nil,
		)
	}
	if event.StopReason != llmprotocol.StopSequence && event.MatchedStopSequence != "" {
		return llmprotocol.Event{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"matched_stop_sequence_reason",
			"upstream matched stop sequence requires stop_sequence reason",
			nil,
		)
	}
	return event, nil
}

func validateFailedEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.Error == nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_failure_shape", "failed stream requires an error", nil)
	}
	if event.Usage != nil {
		if err := llmprotocol.ValidateUsage(*event.Usage); err != nil {
			return llmprotocol.Event{}, err
		}
	}
	switch event.Failure {
	case "":
		event.Failure = llmprotocol.FailureTransport
	case llmprotocol.FailureTransport, llmprotocol.FailureResponse:
	default:
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_failure_scope", "failed stream has an invalid failure scope", nil)
	}
	return event, nil
}

func mergeMonotonicUsage(current, update llmprotocol.Usage) (llmprotocol.Usage, error) {
	if err := llmprotocol.ValidateUsage(update); err != nil {
		return llmprotocol.Usage{}, err
	}
	if current.State == llmprotocol.UsageAvailable && update.State == llmprotocol.UsageUnavailable {
		return llmprotocol.Usage{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_evidence_decreased", "upstream streaming usage evidence became unavailable", nil)
	}
	currentCounts := usageCounts(current)
	updateCounts := usageCounts(update)
	merged := make([]llmprotocol.TokenCount, len(currentCounts))
	for index := range merged {
		value, err := mergeTokenCount(currentCounts[index], updateCounts[index])
		if err != nil {
			return llmprotocol.Usage{}, err
		}
		merged[index] = value
	}
	result := usageFromCounts(mergedUsageState(current.State, update.State), merged)
	if err := deriveUsageTotal(&result); err != nil {
		return llmprotocol.Usage{}, err
	}
	if err := llmprotocol.ValidateUsage(result); err != nil {
		return llmprotocol.Usage{}, err
	}
	return result, nil
}

func mergeTokenCount(existing, incoming llmprotocol.TokenCount) (llmprotocol.TokenCount, error) {
	if incoming.Value == nil {
		return existing, nil
	}
	if existing.Value != nil && *existing.Value > *incoming.Value {
		return llmprotocol.TokenCount{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_decreased", "upstream streaming usage counter decreased", nil)
	}
	if existing.Value != nil && usageEvidenceRank(incoming.Provenance) < usageEvidenceRank(existing.Provenance) {
		return llmprotocol.TokenCount{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_evidence_decreased", "upstream streaming usage evidence quality decreased", nil)
	}
	return incoming, nil
}

func mergedUsageState(current, update llmprotocol.UsageState) llmprotocol.UsageState {
	if update != "" {
		return update
	}
	if current != "" {
		return current
	}
	return llmprotocol.UsageUnavailable
}

func usageCounts(usage llmprotocol.Usage) []llmprotocol.TokenCount {
	return []llmprotocol.TokenCount{
		usage.InputUncached, usage.InputCacheRead, usage.InputCacheWrite,
		usage.OutputReasoning, usage.OutputOther, usage.InputTotal, usage.OutputTotal, usage.Total,
	}
}

func usageFromCounts(state llmprotocol.UsageState, counts []llmprotocol.TokenCount) llmprotocol.Usage {
	return llmprotocol.Usage{
		State: state, InputUncached: counts[0], InputCacheRead: counts[1], InputCacheWrite: counts[2],
		OutputReasoning: counts[3], OutputOther: counts[4], InputTotal: counts[5],
		OutputTotal: counts[6], Total: counts[7],
	}
}

func deriveUsageTotal(usage *llmprotocol.Usage) error {
	if usage.InputTotal.Value == nil || usage.OutputTotal.Value == nil ||
		usage.Total.Value != nil && usage.Total.Provenance != llmprotocol.UsageDerived {
		return nil
	}
	if *usage.OutputTotal.Value > math.MaxInt64-*usage.InputTotal.Value {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_overflow", "upstream streaming usage total overflowed", nil)
	}
	value := *usage.InputTotal.Value + *usage.OutputTotal.Value
	usage.Total = llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageDerived}
	return nil
}

func usageEvidenceRank(provenance llmprotocol.UsageProvenance) int {
	switch provenance {
	case llmprotocol.UsageAuthoritative:
		return 3
	case llmprotocol.UsageDerived:
		return 2
	case llmprotocol.UsageEstimated:
		return 1
	default:
		return 0
	}
}

func (state *streamState) finalize(reason error) ([]llmprotocol.Event, error) {
	if state.terminal {
		return nil, nil
	}
	if reason == nil {
		reason = errors.New("upstream stream ended without a terminal event")
	}
	protocolError := streamFinalizationError(reason, "upstream stream ended before completion")
	event, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: protocolError, StopReason: llmprotocol.StopError})
	if err != nil {
		return nil, err
	}
	return []llmprotocol.Event{event}, nil
}

func streamFinalizationError(reason error, incompleteMessage string) *llmprotocol.ProtocolError {
	var protocolError *llmprotocol.ProtocolError
	if errors.As(reason, &protocolError) {
		return protocolError
	}
	protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", incompleteMessage, reason)
	switch {
	case errors.Is(reason, context.Canceled):
		protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_canceled", "stream was canceled", reason)
	case errors.Is(reason, context.DeadlineExceeded):
		protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_timeout", "stream deadline was exceeded", reason)
	}
	return protocolError
}
