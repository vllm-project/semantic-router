package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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
	call := state.toolCalls[event.ItemIndex]
	if event.ToolCall != nil {
		merged, err := state.mergeStreamToolIdentity(call, *event.ToolCall)
		if err != nil {
			return llmprotocol.Event{}, err
		}
		call = merged
	}
	arguments, err := state.finalToolArguments(event, call.Kind)
	if err != nil {
		return llmprotocol.Event{}, err
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
	incomingKindKnown := incoming.KindKnown || incoming.Kind != ""
	currentKindKnown := current.KindKnown || current.Kind != ""
	if incomingKindKnown {
		if currentKindKnown && current.Kind != incoming.Kind {
			return llmprotocol.ToolCall{}, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_tool_identity_mismatch",
				"upstream stream changed a tool call kind",
				nil,
			)
		}
		current.Kind = incoming.Kind
		current.KindKnown = true
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
	// A Responses client learns a call's namespace with its name, which a Chat
	// stream may send after the call's first fragment.
	if incoming.Namespace != "" {
		if current.Namespace != "" && current.Namespace != incoming.Namespace {
			return llmprotocol.ToolCall{}, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_tool_identity_mismatch",
				"upstream stream changed a tool namespace",
				nil,
			)
		}
		current.Namespace = incoming.Namespace
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

func (state *streamState) finalToolArguments(event llmprotocol.Event, kind llmprotocol.ToolKind) ([]byte, error) {
	arguments := state.toolArguments[event.ItemIndex]
	if event.ToolCall != nil && event.ToolCall.Arguments != "" {
		if err := state.validateStreamToolArgumentAppend(nil, event.ToolCall.Arguments); err != nil {
			return nil, err
		}
		if state.normalizesEmptyChatToolArguments(event, kind) && strings.TrimSpace(string(arguments)) == "" && event.ToolCall.Arguments == "{}" {
			// The decoder normalized held whitespace; the encoder still has the
			// original fragments, none of which it exposed as argument deltas.
			arguments = nil
		}
		if len(arguments) > 0 && string(arguments) != event.ToolCall.Arguments {
			// A stream that ends mid-call closes the members the model finished:
			// the final arguments replace the streamed fragments when they are
			// exactly the close of that prefix, and anything else is an upstream
			// that changed its mind.
			if closed, ok := llmprotocol.TruncatedJSONObject(string(arguments), state.policy.Limits.JSONDepth); !ok || closed != event.ToolCall.Arguments {
				return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_mismatch", "upstream final tool arguments do not match streamed arguments", nil)
			}
			if state.closedToolArguments == nil {
				state.closedToolArguments = make(map[int]string)
			}
			state.closedToolArguments[event.ItemIndex] = string(arguments)
		}
		arguments = []byte(event.ToolCall.Arguments)
	}
	// Only Chat-to-Responses completed function calls have this wire policy.
	// The Chat decoder supplies the confirmed stop after any trailing usage,
	// so an output-limit cut never becomes a successful empty-object call.
	if state.normalizesEmptyChatToolArguments(event, kind) && strings.TrimSpace(string(arguments)) == "" {
		if err := state.validateStreamToolArgumentAppend(nil, "{}"); err != nil {
			return nil, err
		}
		return []byte("{}"), nil
	}
	if kind == "" && !isJSONObject(arguments, state.policy.Limits.JSONDepth) {
		if !isTruncatedJSONObject(arguments, state.policy.Limits.JSONDepth) {
			return nil, nonObjectToolArgumentsErrorWithShape(arguments)
		}
		// The output limit can cut a call mid-arguments, and Anthropic and OpenAI
		// then send the partial call with a max_tokens or length stop. Its stop
		// reason arrives after the item ends, so the terminal event decides.
		state.cutToolArguments = true
		state.cutToolArgumentsEvidence = toolArgumentsEvidence(arguments)
	}
	return arguments, nil
}

func (state *streamState) normalizesEmptyChatToolArguments(event llmprotocol.Event, kind llmprotocol.ToolKind) bool {
	return state.context.Source == llmprotocol.OpenAIChatV1 && state.context.Target == llmprotocol.OpenAIResponsesV1 && kind == "" &&
		(event.StopReason == llmprotocol.StopToolCall || event.StopReason == llmprotocol.StopEndTurn || event.StopReason == llmprotocol.StopSequence)
}

// nonObjectToolArgumentsErrorWithCause rejects a completed tool call whose
// arguments are not a JSON object. The cause never reaches the client frame;
// it is the diagnostic the stream-failure log keeps.
func nonObjectToolArgumentsErrorWithCause(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_tool_arguments", "upstream streamed tool arguments are not a JSON object", cause)
}

// nonObjectToolArgumentsErrorWithShape describes what was rejected without
// recording it: tool arguments can carry file contents, commands or user data
// headed to a tool, so the failure keeps their length and shape only, matching
// the stream-failure logging contract that keeps payloads out of the log. The
// cause never reaches the client frame; it tells a model-generation anomaly
// (prose, leaked markup) from a codec splice one (a JSON fragment).
func nonObjectToolArgumentsErrorWithShape(arguments []byte) error {
	return nonObjectToolArgumentsErrorWithCause(toolArgumentsEvidence(arguments))
}

// toolArgumentsEvidence names rejected arguments by length and shape only.
func toolArgumentsEvidence(arguments []byte) error {
	return llmprotocol.ErrorDiagnostic(fmt.Sprintf("arguments length %d, shape %s", len(arguments), toolArgumentsShape(arguments)))
}

// toolArgumentsShape classifies the first non-space byte of the rejected
// arguments, which separates the likely origins without retaining the text.
func toolArgumentsShape(arguments []byte) string {
	trimmed := bytes.TrimLeft(arguments, " \t\r\n")
	if len(trimmed) == 0 {
		return "empty"
	}
	switch trimmed[0] {
	case '{':
		return "object fragment"
	case '[':
		return "array fragment"
	case '"', '\'':
		return "quoted text"
	case '<':
		return "markup"
	}
	return "plain text"
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
		if llmprotocol.ToolCallStopAtOutputLimit(event.StopReason, state.usage, state.policy.RequestOutputLimit) {
			event.StopReason, state.stop = llmprotocol.StopMaxTokens, llmprotocol.StopMaxTokens
		}
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
	if state.cutToolArguments && state.stop != llmprotocol.StopMaxTokens {
		return llmprotocol.Event{}, nonObjectToolArgumentsErrorWithCause(state.cutToolArgumentsEvidence)
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
	if len(state.items) == 0 && !llmprotocol.StopAllowsEmptyOutput(event.StopReason) {
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
	if sameInputTotal(current, update) {
		// vLLM's Messages stream counts the whole prompt as uncached input in
		// message_start and splits it into cache reads and writes in
		// message_delta. The total is unchanged, so the later split replaces
		// the earlier one instead of reading as a decrease. Only the buckets
		// the update reports are replaced: Anthropic repeats the split in
		// message_delta without the cache-write TTL tiers.
		for _, bucket := range []struct{ current, update *llmprotocol.TokenCount }{
			{&current.InputUncached, &update.InputUncached},
			{&current.InputCacheRead, &update.InputCacheRead},
			{&current.InputCacheWrite, &update.InputCacheWrite},
			{&current.InputCacheWrite5m, &update.InputCacheWrite5m},
			{&current.InputCacheWrite1h, &update.InputCacheWrite1h},
		} {
			if bucket.update.Value != nil {
				*bucket.current = *bucket.update
			}
		}
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
	if result.InputCacheWrite.Value != nil &&
		result.InputCacheWrite5m.Value != nil && result.InputCacheWrite1h.Value != nil {
		if *result.InputCacheWrite5m.Value+*result.InputCacheWrite1h.Value != *result.InputCacheWrite.Value {
			result.InputCacheWrite5m = llmprotocol.TokenCount{}
			result.InputCacheWrite1h = llmprotocol.TokenCount{}
		}
	}
	if update.InputTotal.Value == nil {
		refreshInputTotal(&result)
	}
	if err := deriveUsageTotal(&result); err != nil {
		return llmprotocol.Usage{}, err
	}
	if err := llmprotocol.ValidateUsage(result); err != nil {
		return llmprotocol.Usage{}, err
	}
	return result, nil
}

func sameInputTotal(current, update llmprotocol.Usage) bool {
	return current.InputTotal.Value != nil && update.InputTotal.Value != nil &&
		update.InputUncached.Value != nil && *current.InputTotal.Value == *update.InputTotal.Value
}

// refreshInputTotal re-derives the input total from its buckets after an
// update that reported buckets without a total merged beside retained
// evidence. Anthropic's message delta updates individual input buckets
// without a total, so the running total must follow the merged buckets;
// usage with unknown buckets keeps its declared total, and totals that
// already match stay untouched.
func refreshInputTotal(usage *llmprotocol.Usage) {
	parts := []llmprotocol.TokenCount{usage.InputUncached, usage.InputCacheRead, usage.InputCacheWrite}
	var sum int64
	for _, part := range parts {
		if part.Value == nil || *part.Value < 0 {
			return
		}
		if *part.Value > math.MaxInt64-sum {
			return
		}
		sum += *part.Value
	}
	if usage.InputTotal.Value != nil && *usage.InputTotal.Value == sum {
		return
	}
	provenance := usage.InputTotal.Provenance
	if provenance == "" {
		provenance = llmprotocol.UsageDerived
	}
	usage.InputTotal = llmprotocol.TokenCount{Value: llmprotocol.Int64(sum), Provenance: provenance}
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
		usage.InputCacheWrite5m, usage.InputCacheWrite1h,
		usage.OutputReasoning, usage.OutputOther, usage.InputTotal, usage.OutputTotal, usage.Total,
	}
}

func usageFromCounts(state llmprotocol.UsageState, counts []llmprotocol.TokenCount) llmprotocol.Usage {
	return llmprotocol.Usage{
		State: state, InputUncached: counts[0], InputCacheRead: counts[1], InputCacheWrite: counts[2],
		InputCacheWrite5m: counts[3], InputCacheWrite1h: counts[4],
		OutputReasoning: counts[5], OutputOther: counts[6], InputTotal: counts[7],
		OutputTotal: counts[8], Total: counts[9],
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

func (state *streamState) finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if state.terminal {
		return nil, nil, nil
	}
	if events, diagnostics, repaired := state.repairCutToolCall(); repaired {
		return events, diagnostics, nil
	}
	if reason == nil {
		reason = errors.New("upstream stream ended without a terminal event")
	}
	protocolError := streamFinalizationError(reason, "upstream stream ended before completion")
	event, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: protocolError, StopReason: llmprotocol.StopError})
	if err != nil {
		return nil, nil, err
	}
	return []llmprotocol.Event{event}, nil, nil
}

// repairCutToolCall closes the one tool call whose arguments an upstream stream
// end cut short without a limit signal. Its identity streamed complete and its
// arguments stop inside a JSON object, so the members the model finished still
// close into a valid call: the item completes with that close, and the stream
// ends with the limit stop the output limit's own cut uses, plus an approximated
// diagnostic naming the repair. The public Chat filter holds the streamed
// fragments, so a same-format client sees only the closed call; a translated
// client sees the close as the final arguments of the call it was already
// reading. Several pending calls, a missing identity, or arguments that are not
// a cut object keep the stream's failure.
func (state *streamState) repairCutToolCall() ([]llmprotocol.Event, llmprotocol.Diagnostics, bool) {
	if len(state.toolArguments) != 1 {
		return nil, nil, false
	}
	index, arguments := -1, []byte(nil)
	for itemIndex, pending := range state.toolArguments {
		index, arguments = itemIndex, pending
	}
	members, ok := llmprotocol.TruncatedJSONObject(string(arguments), state.policy.Limits.JSONDepth)
	if !ok {
		return nil, nil, false
	}
	completed, err := state.next(llmprotocol.Event{
		Type:      llmprotocol.EventOutputItemCompleted,
		ItemIndex: index,
		ToolCall:  &llmprotocol.ToolCall{Arguments: members},
	})
	if err != nil {
		return nil, nil, false
	}
	terminal, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: llmprotocol.StopMaxTokens})
	if err != nil {
		return nil, nil, false
	}
	diagnostics := appendDiagnostics(nil, llmprotocol.Diagnostics{{
		Source: state.context.Source,
		Field:  "tool_call.arguments",
		Action: llmprotocol.DiagnosticApproximated,
		Reason: "upstream ended the stream mid-call without a limit signal; the arguments were closed at the stream end",
	}}, state.policy.Limits.Diagnostics)
	return []llmprotocol.Event{completed, terminal}, diagnostics, true
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
