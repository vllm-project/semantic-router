package protocolcodec

import (
	"encoding/json"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type completedResponsesContent struct {
	message          []responsesContentWire
	reasoningSummary []responsesContentWire
	reasoningText    []responsesContentWire
}

func (encoder *responsesStreamEncoder) collectCompletedResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
) (completedResponsesContent, [][]byte, error) {
	parts := completedResponsesContent{}
	var frames [][]byte
	for _, key := range encoder.responsesContentKeys(event, outputKey) {
		kind := encoder.encodedKinds[key]
		if kind == "" {
			kind = llmprotocol.ContentText
		}
		contentEvent := event
		contentEvent.ContentIndex = key.content
		completed, err := encoder.completeResponsesContent(contentEvent, outputKey, kind)
		if err != nil {
			return completedResponsesContent{}, nil, err
		}
		frames = append(frames, completed...)
		parts.append(kind, encoder.reasoningScopes[key], responsesContentPart(
			kind, encoder.reasoningScopes[key], encoder.responsesContentText(key), encoder.contentCitations[key],
		))
	}
	return parts, frames, nil
}

func (encoder *responsesStreamEncoder) responsesContentText(key streamContentKey) string {
	if builder := encoder.contentText[key]; builder != nil {
		return builder.String()
	}
	return ""
}

func (parts *completedResponsesContent) append(
	kind llmprotocol.ContentKind,
	reasoningScope llmprotocol.ReasoningScope,
	part responsesContentWire,
) {
	if kind != llmprotocol.ContentReasoning {
		parts.message = append(parts.message, part)
		return
	}
	if normalizedReasoningScope(reasoningScope) == llmprotocol.ReasoningScopeSummary {
		parts.reasoningSummary = append(parts.reasoningSummary, part)
		return
	}
	parts.reasoningText = append(parts.reasoningText, part)
}

func marshalCompletedResponsesItem(
	outputKey responsesOutputKey,
	id string,
	parts completedResponsesContent,
) (responsesItemWire, error) {
	item := responsesItemWire{Type: "message", ID: id, Role: "assistant", Status: "completed"}
	if outputKey.kind != responsesOutputReasoning {
		content, err := json.Marshal(parts.message)
		item.Content = content
		return item, err
	}
	item.Type, item.Role = "reasoning", ""
	var err error
	if len(parts.reasoningSummary) > 0 {
		item.Summary, err = json.Marshal(parts.reasoningSummary)
		if err != nil {
			return responsesItemWire{}, err
		}
	}
	if len(parts.reasoningText) > 0 {
		item.Content, err = json.Marshal(parts.reasoningText)
	}
	return item, err
}

func (encoder *responsesStreamEncoder) responsesContentKeys(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
) []streamContentKey {
	keys := make([]streamContentKey, 0)
	for key, kind := range encoder.encodedKinds {
		if key.item == event.ItemIndex && responsesOutputKindForContent(kind) == outputKey.kind {
			keys = append(keys, key)
		}
	}
	if len(keys) == 0 {
		key := contentKey(event)
		kind := llmprotocol.ContentText
		if event.Content != nil && event.Content.Kind != "" {
			kind = event.Content.Kind
		}
		encoder.encodedKinds[key] = kind
		keys = append(keys, key)
	}
	sort.Slice(keys, func(left, right int) bool {
		return keys[left].content < keys[right].content
	})
	return keys
}

func responsesOutputKindForContent(kind llmprotocol.ContentKind) responsesOutputKind {
	if kind == llmprotocol.ContentReasoning {
		return responsesOutputReasoning
	}
	if kind == llmprotocol.ContentGeneratedImage {
		return responsesOutputImage
	}
	return responsesOutputMessage
}

func (encoder *responsesStreamEncoder) recordResponsesCompletedOutput(index int, item json.RawMessage) {
	encoder.completedOutput[index] = append(json.RawMessage(nil), item...)
}

func (encoder *responsesStreamEncoder) responsesCompletedOutput() (json.RawMessage, error) {
	indexes := make([]int, 0, len(encoder.completedOutput))
	for index := range encoder.completedOutput {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	items := make([]json.RawMessage, 0, len(indexes))
	for _, index := range indexes {
		items = append(items, encoder.completedOutput[index])
	}
	body, err := json.Marshal(items)
	if err != nil {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorInternal,
			"responses_stream_output",
			"completed Responses stream output could not be encoded",
			err,
		)
	}
	return body, nil
}

func (encoder *responsesStreamEncoder) nextWireSequence() uint64 {
	sequence := encoder.wireSequence
	encoder.wireSequence++
	return sequence
}

func responsesContentIndex(value int) *int {
	return &value
}

func responsesOutputIndex(value int) *int {
	return &value
}

func responsesWireOutputIndex(wire responsesEventWire) int {
	if wire.OutputIndex == nil {
		return 0
	}
	return *wire.OutputIndex
}

func responsesContentPart(
	kind llmprotocol.ContentKind,
	reasoning llmprotocol.ReasoningScope,
	text string,
	citations []llmprotocol.Citation,
) responsesContentWire {
	if kind == llmprotocol.ContentRefusal {
		return responsesContentWire{Type: "refusal", Refusal: text}
	}
	if kind == llmprotocol.ContentReasoning {
		if normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeSummary {
			return responsesContentWire{Type: "summary_text", Text: text}
		}
		return responsesContentWire{Type: "reasoning_text", Text: text}
	}
	return responsesContentWire{Type: "output_text", Text: text, Annotations: responsesAnnotations(citations)}
}

func responsesTextDeltaType(kind llmprotocol.ContentKind) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.delta"
	}
	return "response.output_text.delta"
}

func eventReasoningScope(event llmprotocol.Event) llmprotocol.ReasoningScope {
	if event.Content == nil {
		return llmprotocol.ReasoningScopeText
	}
	return normalizedReasoningScope(event.Content.Reasoning)
}

func normalizedReasoningScope(scope llmprotocol.ReasoningScope) llmprotocol.ReasoningScope {
	if scope == llmprotocol.ReasoningScopeSummary {
		return scope
	}
	return llmprotocol.ReasoningScopeText
}

func responsesScopeForReasoning(scope llmprotocol.ReasoningScope) responsesContentScope {
	if normalizedReasoningScope(scope) == llmprotocol.ReasoningScopeSummary {
		return responsesContentReasoningSummary
	}
	return responsesContentReasoningText
}

func responsesContentScopeFor(
	kind llmprotocol.ContentKind,
	reasoning llmprotocol.ReasoningScope,
) responsesContentScope {
	if kind == llmprotocol.ContentReasoning {
		return responsesScopeForReasoning(reasoning)
	}
	return responsesContentMessage
}

func responsesReasoningDeltaType(scope llmprotocol.ReasoningScope) string {
	if normalizedReasoningScope(scope) == llmprotocol.ReasoningScopeSummary {
		return "response.reasoning_summary_text.delta"
	}
	return "response.reasoning_text.delta"
}

func setResponsesReasoningIndex(
	wire *responsesEventWire,
	scope llmprotocol.ReasoningScope,
	index int,
) {
	if normalizedReasoningScope(scope) == llmprotocol.ReasoningScopeSummary {
		wire.SummaryIndex = responsesContentIndex(index)
		return
	}
	wire.ContentIndex = responsesContentIndex(index)
}

func responsesTextDoneType(kind llmprotocol.ContentKind, reasoning llmprotocol.ReasoningScope) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.done"
	}
	if kind == llmprotocol.ContentReasoning {
		if normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeSummary {
			return "response.reasoning_summary_text.done"
		}
		return "response.reasoning_text.done"
	}
	return "response.output_text.done"
}

func (encoder *responsesStreamEncoder) startResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	key := contentKey(event)
	if encoder.contentStarted[key] {
		return nil, nil
	}
	encoder.contentStarted[key] = true
	reasoning := encoder.reasoningScopes[key]
	scope := responsesContentScopeFor(kind, reasoning)
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, scope)
	if kind == llmprotocol.ContentReasoning {
		if normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeText {
			return nil, nil
		}
	}
	wire := responsesEventWire{
		Type: "response.content_part.added", Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
	}
	if kind == llmprotocol.ContentReasoning {
		wire.Type = "response.reasoning_summary_part.added"
		wire.SummaryIndex = responsesContentIndex(contentIndex)
	} else {
		wire.ContentIndex = responsesContentIndex(contentIndex)
	}
	part := responsesContentPart(kind, reasoning, "", nil)
	wire.Part = &part
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return nil, err
	}
	return [][]byte{frame}, nil
}

func (encoder *responsesStreamEncoder) completeResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	frames, err := encoder.startResponsesContent(event, outputKey, kind)
	if err != nil {
		return nil, err
	}
	text := ""
	key := contentKey(event)
	reasoning := encoder.reasoningScopes[key]
	if builder := encoder.contentText[key]; builder != nil {
		text = builder.String()
	}
	scope := responsesContentScopeFor(kind, reasoning)
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, scope)
	done := responsesEventWire{
		Type: responsesTextDoneType(kind, reasoning), Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
	}
	if kind == llmprotocol.ContentReasoning {
		setResponsesReasoningIndex(&done, reasoning, contentIndex)
		done.Text = text
	} else {
		done.ContentIndex = responsesContentIndex(contentIndex)
		if kind == llmprotocol.ContentRefusal {
			done.Refusal = text
		} else {
			done.Text = text
		}
	}
	doneFrame, err := encoder.encodeResponsesStreamFrame(done)
	if err != nil {
		return nil, err
	}
	if kind == llmprotocol.ContentReasoning && normalizedReasoningScope(reasoning) == llmprotocol.ReasoningScopeText {
		return append(frames, doneFrame), nil
	}
	part := responsesContentPart(kind, reasoning, text, encoder.contentCitations[key])
	partDone := responsesEventWire{
		Type: "response.content_part.done", Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
		Part: &part,
	}
	if kind == llmprotocol.ContentReasoning {
		partDone.Type = "response.reasoning_summary_part.done"
		partDone.SummaryIndex = responsesContentIndex(contentIndex)
	} else {
		partDone.ContentIndex = responsesContentIndex(contentIndex)
	}
	partFrame, err := encoder.encodeResponsesStreamFrame(partDone)
	if err != nil {
		return nil, err
	}
	return append(frames, doneFrame, partFrame), nil
}

func (encoder *responsesStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	protocolError := streamFinalizationError(reason, "stream ended before completion")
	frame, err := encoder.encodeTransportError(protocolError)
	return [][]byte{frame}, nil, err
}

func (encoder *responsesStreamEncoder) encodeTransportError(protocolError *llmprotocol.ProtocolError) ([]byte, error) {
	wire := responsesTransportErrorEventWire{
		Type: "error", Code: optionalString(protocolError.Code), Message: protocolError.Message,
		Param: optionalString(protocolError.Parameter), Sequence: encoder.nextWireSequence(),
	}
	return encodeSSE(wire.Type, wire)
}

func (encoder *responsesStreamEncoder) encodeResponsesStreamFrame(wire responsesEventWire) ([]byte, error) {
	obfuscation, err := newStreamObfuscation(encoder.context)
	if err != nil {
		return nil, err
	}
	wire.Obfuscation = obfuscation
	return encodeSSE(wire.Type, wire)
}
