package protocolcodec

import (
	"bytes"
	"fmt"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type streamState struct {
	context               llmprotocol.StreamContext
	policy                llmprotocol.Policy
	providerID            string
	providerModel         string
	sequence              uint64
	events                int
	wireFrames            int
	wireBytes             int
	terminal              bool
	started               bool
	usage                 llmprotocol.Usage
	stop                  llmprotocol.StopReason
	items                 map[int]bool
	completedItems        map[int]bool
	itemKinds             map[int]llmprotocol.ContentKind
	itemIDs               map[int]string
	itemIDIndexes         map[string]int
	contentBlocks         map[streamContentKey]bool
	contentKinds          map[streamContentKey]llmprotocol.ContentKind
	reasoningScopes       map[streamContentKey]llmprotocol.ReasoningScope
	itemTextBytes         map[streamContentKey]int
	itemTextRunes         map[streamContentKey]int64
	itemCitations         map[streamContentKey]int
	toolCalls             map[int]llmprotocol.ToolCall
	toolCallIndexes       map[string]int
	toolArguments         map[int][]byte
	imageProgressRank     map[int]int
	imageProgressSeen     map[int]map[llmprotocol.ImageGenerationStatus]bool
	nextPartialImageIndex map[int]int64
}

func (state *streamState) observeProviderStreamBytes(chunk []byte) error {
	limit := state.policy.Limits.BodyBytes
	if limit > 0 && (state.wireBytes > limit || len(chunk) > limit-state.wireBytes) {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_body_limit",
			"upstream response stream exceeds the configured body limit",
			nil,
		)
	}
	state.wireBytes += len(chunk)
	return nil
}

func (state *streamState) parseProviderSSEFrame(frame []byte) (sseFrame, error) {
	parsed, err := parseSSEFrameAtPosition(
		frame,
		state.policy.Limits.SSEFrameBytes,
		state.wireFrames == 0,
	)
	state.wireFrames++
	return parsed, err
}

func (state *streamState) observeProviderIdentity(responseID, model string) error {
	if responseID != "" {
		if state.policy.Limits.IdentifierBytes > 0 && len(responseID) > state.policy.Limits.IdentifierBytes {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_response_id_limit",
				"upstream stream response ID exceeds the configured limit",
				nil,
			)
		}
		if state.providerID != "" && state.providerID != responseID {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_response_id_mismatch",
				"upstream stream changed response ID",
				nil,
			)
		}
		state.providerID = responseID
	}
	if model != "" {
		if state.policy.Limits.ModelBytes > 0 && len(model) > state.policy.Limits.ModelBytes {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_model_limit",
				"upstream stream model exceeds the configured limit",
				nil,
			)
		}
		if state.providerModel != "" && state.providerModel != model {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_model_mismatch",
				"upstream stream changed model",
				nil,
			)
		}
		state.providerModel = model
	}
	return nil
}

func (state *streamState) next(event llmprotocol.Event) (llmprotocol.Event, error) {
	event, err := state.prepareEvent(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	event, err = state.applyItemEvent(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	return state.applyEventEvidence(event)
}

func (state *streamState) prepareEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if state.terminal {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	if !validStreamEventType(event.Type) {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "unknown_stream_event", "upstream stream event type is invalid", nil)
	}
	state.events++
	if state.policy.Limits.Events > 0 && state.events > state.policy.Limits.Events {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_event_limit", "stream event limit exceeded", nil)
	}
	state.sequence++
	event.Sequence = state.sequence
	if event.ResponseID == "" {
		event.ResponseID = state.context.ResponseID
	}
	if state.context.PublicModel != "" {
		event.Model = state.context.PublicModel
	}
	state.ensureCollections()
	return state.prepareLifecycleEvent(event)
}

func validStreamEventType(eventType llmprotocol.EventType) bool {
	switch eventType {
	case llmprotocol.EventResponseStarted, llmprotocol.EventOutputItemStarted,
		llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta,
		llmprotocol.EventToolCallDelta, llmprotocol.EventImageGenerationProgress,
		llmprotocol.EventOutputItemCompleted,
		llmprotocol.EventUsageUpdated, llmprotocol.EventResponseCompleted,
		llmprotocol.EventResponseFailed, llmprotocol.EventProviderOpaque:
		return true
	default:
		return false
	}
}

func (state *streamState) ensureCollections() {
	if state.items == nil {
		state.items = make(map[int]bool)
		state.completedItems = make(map[int]bool)
		state.itemKinds = make(map[int]llmprotocol.ContentKind)
		state.itemIDs = make(map[int]string)
		state.itemIDIndexes = make(map[string]int)
		state.contentBlocks = make(map[streamContentKey]bool)
		state.contentKinds = make(map[streamContentKey]llmprotocol.ContentKind)
		state.reasoningScopes = make(map[streamContentKey]llmprotocol.ReasoningScope)
		state.itemTextBytes = make(map[streamContentKey]int)
		state.itemTextRunes = make(map[streamContentKey]int64)
		state.itemCitations = make(map[streamContentKey]int)
		state.toolCalls = make(map[int]llmprotocol.ToolCall)
		state.toolCallIndexes = make(map[string]int)
		state.toolArguments = make(map[int][]byte)
		state.imageProgressRank = make(map[int]int)
		state.imageProgressSeen = make(map[int]map[llmprotocol.ImageGenerationStatus]bool)
		state.nextPartialImageIndex = make(map[int]int64)
		if state.usage.State == "" {
			state.usage.State = llmprotocol.UsageUnavailable
		}
	}
}

func (state *streamState) prepareLifecycleEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.Type == llmprotocol.EventResponseStarted {
		return state.prepareStartEvent(event)
	}
	if event.Type != llmprotocol.EventProviderOpaque && event.Type != llmprotocol.EventResponseFailed && !state.started {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream emitted output before response start", nil)
	}
	return event, nil
}

func (state *streamState) prepareStartEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	if state.started {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_start", "upstream stream started more than once", nil)
	}
	if event.ResponseID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		event.ResponseID = llmprotocol.StableID("stream-response", state.context.PublicModel, state.context.ProviderModel)
	}
	if event.ResponseID == "" {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_response_id", "upstream stream response ID is missing", nil)
	}
	if state.policy.Limits.IdentifierBytes > 0 && len(event.ResponseID) > state.policy.Limits.IdentifierBytes {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_response_id_limit", "upstream stream response ID exceeds the configured limit", nil)
	}
	if state.policy.Limits.ModelBytes > 0 && len(event.Model) > state.policy.Limits.ModelBytes {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_model_limit", "upstream stream model exceeds the configured limit", nil)
	}
	state.context.ResponseID = event.ResponseID
	state.started = true
	return event, nil
}

func (state *streamState) applyItemEvent(event llmprotocol.Event) (llmprotocol.Event, error) {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		return state.startItem(event)
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta, llmprotocol.EventToolCallDelta:
		return state.applyDelta(event)
	case llmprotocol.EventImageGenerationProgress:
		return state.applyImageGenerationProgress(event)
	case llmprotocol.EventOutputItemCompleted:
		return state.completeItem(event)
	}
	return event, nil
}

func (state *streamState) startItem(event llmprotocol.Event) (llmprotocol.Event, error) {
	prepared, err := state.prepareItemStart(event)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	event = prepared
	if err := state.prepareItemTool(event); err != nil {
		return llmprotocol.Event{}, err
	}
	state.recordItemStart(event)
	if event.ToolCall != nil {
		state.recordStartedTool(event)
		return event, nil
	}
	if err := state.prepareStartedContent(event); err != nil {
		return llmprotocol.Event{}, err
	}
	return event, nil
}

func (state *streamState) prepareStartedContent(event llmprotocol.Event) error {
	if event.Content == nil {
		return nil
	}
	if event.Content.Kind == llmprotocol.ContentGeneratedImage {
		if err := llmprotocol.ValidateGeneratedImage(event.Content.GeneratedImage, state.policy.Limits); err != nil {
			return upstreamSemanticValidationError(err)
		}
		if err := validateStartedGeneratedImage(event.Content.GeneratedImage); err != nil {
			return err
		}
	}
	if err := state.claimContentBlock(event); err != nil {
		return err
	}
	state.itemKinds[event.ItemIndex] = event.Content.Kind
	return nil
}

func (state *streamState) prepareItemStart(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.ItemIndex < 0 {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_index", "upstream output item index is invalid", nil)
	}
	if state.items[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_item_start", "upstream output item started more than once", nil)
	}
	if state.policy.Limits.OutputItems > 0 && len(state.items) >= state.policy.Limits.OutputItems {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "output_items_limit", "upstream output item limit exceeded", nil)
	}
	if event.ItemID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
		event.ItemID = llmprotocol.StableID(event.ResponseID, string(event.Type), fmt.Sprint(event.ItemIndex))
	}
	if event.ItemID == "" {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id", "upstream output item ID is missing", nil)
	}
	if state.policy.Limits.IdentifierBytes > 0 && len(event.ItemID) > state.policy.Limits.IdentifierBytes {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id_limit", "upstream output item ID exceeds the configured limit", nil)
	}
	if index, duplicate := state.itemIDIndexes[event.ItemID]; duplicate && index != event.ItemIndex {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_item_id", "upstream stream reused an output item ID", nil)
	}
	return event, nil
}

func (state *streamState) prepareItemTool(event llmprotocol.Event) error {
	if event.ToolCall == nil {
		return nil
	}
	if err := state.claimContentBlock(event); err != nil {
		return err
	}
	if err := state.validateStreamToolIdentity(*event.ToolCall, true); err != nil {
		return err
	}
	if err := state.validateStreamToolArgumentAppend(nil, event.ToolCall.Arguments); err != nil {
		return err
	}
	return state.claimStreamToolCallID(event.ItemIndex, event.ToolCall.ID)
}

func (state *streamState) recordItemStart(event llmprotocol.Event) {
	state.items[event.ItemIndex] = true
	state.itemIDs[event.ItemIndex] = event.ItemID
	state.itemIDIndexes[event.ItemID] = event.ItemIndex
}

func (state *streamState) recordStartedTool(event llmprotocol.Event) {
	state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
	state.toolCalls[event.ItemIndex] = *event.ToolCall
	state.toolArguments[event.ItemIndex] = append([]byte(nil), event.ToolCall.Arguments...)
}

func (state *streamState) applyDelta(event llmprotocol.Event) (llmprotocol.Event, error) {
	if err := state.validateDelta(event); err != nil {
		return llmprotocol.Event{}, err
	}
	if event.Type == llmprotocol.EventOutputTextDelta || event.Type == llmprotocol.EventReasoningDelta {
		if err := state.recordTextDelta(event); err != nil {
			return llmprotocol.Event{}, err
		}
	}
	if event.Type == llmprotocol.EventToolCallDelta {
		return state.recordToolDelta(event)
	}
	state.recordDeltaKind(event)
	return event, nil
}

func (state *streamState) validateDelta(event llmprotocol.Event) error {
	if event.ContentIndex < 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_content_index", "upstream content index is invalid", nil)
	}
	if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream delta does not reference an active output item", nil)
	}
	if event.ItemID != "" && event.ItemID != state.itemIDs[event.ItemIndex] {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id_mismatch", "upstream delta changed its output item ID", nil)
	}
	if event.Content != nil && len(event.Content.Citations) > 0 &&
		(event.Type != llmprotocol.EventOutputTextDelta || event.Content.Kind != llmprotocol.ContentText) {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_citation", "upstream citations require a text delta", nil)
	}
	return state.claimContentBlock(event)
}

func (state *streamState) claimContentBlock(event llmprotocol.Event) error {
	if event.ContentIndex < 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_content_index", "upstream content index is invalid", nil)
	}
	key := contentKey(event)
	if state.contentBlocks[key] {
		if err := state.observeContentKind(key, eventContentKind(event)); err != nil {
			return err
		}
		return state.observeReasoningScope(key, event)
	}
	if state.policy.Limits.ContentBlocks > 0 && len(state.contentBlocks) >= state.policy.Limits.ContentBlocks {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "content_blocks_limit", "upstream content block limit exceeded", nil)
	}
	state.contentBlocks[key] = true
	if err := state.observeContentKind(key, eventContentKind(event)); err != nil {
		return err
	}
	return state.observeReasoningScope(key, event)
}

func eventContentKind(event llmprotocol.Event) llmprotocol.ContentKind {
	if event.ToolCall != nil || event.Type == llmprotocol.EventToolCallDelta {
		return llmprotocol.ContentToolCall
	}
	if event.Content != nil && event.Content.Kind != "" {
		return event.Content.Kind
	}
	if event.Type == llmprotocol.EventReasoningDelta {
		return llmprotocol.ContentReasoning
	}
	if event.Type == llmprotocol.EventOutputTextDelta {
		return llmprotocol.ContentText
	}
	return ""
}

func (state *streamState) observeContentKind(key streamContentKey, kind llmprotocol.ContentKind) error {
	if kind == "" {
		return nil
	}
	if existing := state.contentKinds[key]; existing != "" && existing != kind {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_content_kind_mismatch",
			"upstream stream changed a content block kind",
			nil,
		)
	}
	state.contentKinds[key] = kind
	return nil
}

func (state *streamState) observeReasoningScope(key streamContentKey, event llmprotocol.Event) error {
	if event.Content == nil || event.Content.Kind != llmprotocol.ContentReasoning || event.Content.Reasoning == "" {
		return nil
	}
	scope := event.Content.Reasoning
	if scope != llmprotocol.ReasoningScopeText && scope != llmprotocol.ReasoningScopeSummary {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_reasoning_scope",
			"upstream stream emitted an unsupported reasoning scope",
			nil,
		)
	}
	if existing := state.reasoningScopes[key]; existing != "" && existing != scope {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_reasoning_scope_mismatch",
			"upstream stream changed a reasoning content scope",
			nil,
		)
	}
	state.reasoningScopes[key] = scope
	return nil
}

func (state *streamState) recordDeltaKind(event llmprotocol.Event) {
	if event.Content != nil {
		state.itemKinds[event.ItemIndex] = event.Content.Kind
	} else if event.Type == llmprotocol.EventReasoningDelta {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentReasoning
	} else if state.itemKinds[event.ItemIndex] == "" {
		state.itemKinds[event.ItemIndex] = llmprotocol.ContentText
	}
}

func (state *streamState) recordTextDelta(event llmprotocol.Event) error {
	key := contentKey(event)
	textBytes := state.itemTextBytes[key] + len(event.Delta)
	if state.policy.Limits.TextBytes > 0 && textBytes > state.policy.Limits.TextBytes {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "text_limit", "content text exceeds the configured limit", nil)
	}
	textRunes := state.itemTextRunes[key] + int64(utf8.RuneCountInString(event.Delta))
	citationCount := state.itemCitations[key]
	var citationBatch []llmprotocol.Citation
	if event.Content != nil && len(event.Content.Citations) > 0 {
		citationBatch = event.Content.Citations
		citationCount += len(citationBatch)
	}
	if err := llmprotocol.ValidateCitationBatch(textRunes, citationCount, citationBatch, state.policy.Limits); err != nil {
		return err
	}
	state.itemTextBytes[key] = textBytes
	state.itemTextRunes[key] = textRunes
	state.itemCitations[key] = citationCount
	return nil
}

func (state *streamState) recordToolDelta(event llmprotocol.Event) (llmprotocol.Event, error) {
	if event.ToolCall == nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "tool_delta_missing", "upstream tool delta is missing", nil)
	}
	state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
	call := state.toolCalls[event.ItemIndex]
	merged, err := state.mergeStreamToolIdentity(call, *event.ToolCall)
	if err != nil {
		return llmprotocol.Event{}, err
	}
	call = merged
	if err := state.claimStreamToolCallID(event.ItemIndex, call.ID); err != nil {
		return llmprotocol.Event{}, err
	}
	state.toolCalls[event.ItemIndex] = call
	event.ToolCall.ID, event.ToolCall.Name = call.ID, call.Name
	current := state.toolArguments[event.ItemIndex]
	if bytes.Equal(bytes.TrimSpace(current), []byte("{}")) && event.ToolCall.Arguments != "" {
		current = nil
	}
	if err := state.validateStreamToolArgumentAppend(current, event.ToolCall.Arguments); err != nil {
		return llmprotocol.Event{}, err
	}
	state.toolArguments[event.ItemIndex] = append(current, event.ToolCall.Arguments...)
	return event, nil
}
