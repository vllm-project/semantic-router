package protocolcodec

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// chatSource reports a Chat backend. Its stream ends every output of its
// choice at once, at the end, so its outputs close as the backend moves on to
// other ones, and its empty fields stand for nothing, so they open no output
// or part. Other sources end each item and start each part themselves.
func (encoder *responsesStreamEncoder) chatSource() bool {
	return encoder.context.Source == llmprotocol.OpenAIChatV1
}

// currentResponsesOutput is the output an item's content of a kind streams
// into now.
func (encoder *responsesStreamEncoder) currentResponsesOutput(item int, kind responsesOutputKind) responsesOutputKey {
	key := responsesOutputKey{item: item, kind: kind}
	key.segment = encoder.outputSegments[key]
	return key
}

// openResponsesOutput makes the output an event streams into the active one,
// starting it if needed. For a Chat source the outputs the stream leaves close
// first, so items are sequential as in OpenAI's streams, and content that
// resumes after its output closed streams into a new output of its kind.
func (encoder *responsesStreamEncoder) openResponsesOutput(
	event llmprotocol.Event,
	kind responsesOutputKind,
) ([][]byte, responsesOutputKey, error) {
	key := encoder.currentResponsesOutput(event.ItemIndex, kind)
	if encoder.outputDone[key] {
		key.segment++
		encoder.outputSegments[responsesOutputKey{item: key.item, kind: key.kind}] = key.segment
	}
	if encoder.hasActiveOutput && encoder.activeOutput == key {
		return nil, key, nil
	}
	frames, err := encoder.releaseHeldMessage(false)
	if err != nil {
		return nil, key, err
	}
	if encoder.chatSource() {
		var closed [][]byte
		if closed, err = encoder.leaveResponsesOutputs(key); err != nil {
			return nil, key, err
		}
		frames = append(frames, closed...)
	}
	started, err := encoder.startResponsesOutput(event, key)
	if err != nil {
		return nil, key, err
	}
	encoder.activeOutput, encoder.hasActiveOutput = key, true
	return append(frames, started...), key, nil
}

// leaveResponsesOutputs closes the open outputs, other than the next one, that
// can take no more content: text and reasoning, since content that resumes
// starts a new output, and function calls whose arguments form a JSON object,
// which nothing can extend. The output limit ends only the last output, so each
// is complete. A custom call's input shows no end, and an image's result comes
// with its item, so they wait for their item.
func (encoder *responsesStreamEncoder) leaveResponsesOutputs(next responsesOutputKey) ([][]byte, error) {
	var frames [][]byte
	for _, open := range slices.Clone(encoder.openOutputs) {
		if open == next {
			continue
		}
		event, closable := encoder.responsesOutputClosure(open)
		if !closable {
			continue
		}
		closed, _, err := encoder.encodeCompletedResponsesOutput(event, open, false)
		if err != nil {
			return nil, err
		}
		frames = append(frames, closed...)
	}
	return frames, nil
}

// responsesOutputClosure is the completion of an output the stream left, and
// whether the output can close now.
func (encoder *responsesStreamEncoder) responsesOutputClosure(key responsesOutputKey) (llmprotocol.Event, bool) {
	event := llmprotocol.Event{ItemIndex: key.item}
	switch key.kind {
	case responsesOutputMessage:
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText}
	case responsesOutputReasoning:
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}
	case responsesOutputTool:
		call := encoder.toolCalls[key.item]
		call.Arguments = string(encoder.toolArguments[key.item])
		if call.Kind != "" || !isJSONObject([]byte(call.Arguments), encoder.policy.Limits.JSONDepth) {
			return event, false
		}
		event.ToolCall = &call
	default:
		return event, false
	}
	return event, true
}

// finishResponsesOutput marks an output whose done events went out, or wait
// as a held message.
func (encoder *responsesStreamEncoder) finishResponsesOutput(key responsesOutputKey) {
	encoder.outputDone[key] = true
	encoder.openOutputs = slices.DeleteFunc(encoder.openOutputs, func(open responsesOutputKey) bool { return open == key })
	if encoder.activeOutput == key {
		encoder.hasActiveOutput = false
	}
}

// bindResponsesContent moves a content block into the output it streams into
// now, where it starts as a new part.
func (encoder *responsesStreamEncoder) bindResponsesContent(key streamContentKey, output responsesOutputKey) {
	if encoder.contentSegments[key] == output.segment {
		return
	}
	encoder.contentSegments[key] = output.segment
	delete(encoder.contentStarted, key)
	delete(encoder.contentIndexes, key)
	delete(encoder.contentText, key)
	delete(encoder.contentCitations, key)
}
