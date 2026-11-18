package protocolcodec

import (
	"bytes"
	"encoding/json"
)

// replayAnswerSeparatorTrimmer is implemented by a codec that can drop the
// answer separator from the bytes it replays to a same-format client, so the
// reply keeps every other member the provider sent.
type replayAnswerSeparatorTrimmer interface {
	trimReplayAnswerSeparator(replay []byte) ([]byte, bool)
}

// A Chat answer has one text block, so its citation offsets share one key.
var chatAnswerKey = streamContentKey{}

// trimReplayAnswerSeparator drops the separator from each choice's message and
// reports false when nothing changed or the body is not a Chat reply.
func (OpenAIChatCodec) trimReplayAnswerSeparator(replay []byte) ([]byte, bool) {
	var reply map[string]json.RawMessage
	var choices []json.RawMessage
	if json.Unmarshal(replay, &reply) != nil || json.Unmarshal(reply["choices"], &choices) != nil {
		return nil, false
	}
	changed := false
	for index, raw := range choices {
		var choice, message map[string]json.RawMessage
		if json.Unmarshal(raw, &choice) != nil || json.Unmarshal(choice["message"], &message) != nil || message == nil {
			continue
		}
		var separator answerSeparator
		if !separator.trimChatObject(message) {
			continue
		}
		var err error
		if choice["message"], err = json.Marshal(message); err != nil {
			return nil, false
		}
		if choices[index], err = json.Marshal(choice); err != nil {
			return nil, false
		}
		changed = true
	}
	if !changed {
		return nil, false
	}
	var err error
	if reply["choices"], err = json.Marshal(choices); err != nil {
		return nil, false
	}
	patched, err := json.Marshal(reply)
	return patched, err == nil
}

// trimChatChoice drops the separator from a stream chunk's choice and returns
// nil when the chunk is unchanged.
func (separator *answerSeparator) trimChatChoice(raw json.RawMessage) json.RawMessage {
	if !separator.mayTrimChatChoice(raw) {
		return nil
	}
	var choice, delta map[string]json.RawMessage
	if json.Unmarshal(raw, &choice) != nil || json.Unmarshal(choice["delta"], &delta) != nil || delta == nil {
		return nil
	}
	if index, found := choice["index"]; found && string(bytes.TrimSpace(index)) != "0" {
		return nil
	}
	if !separator.trimChatObject(delta) {
		return nil
	}
	encodedDelta, err := json.Marshal(delta)
	if err != nil {
		return nil
	}
	choice["delta"] = encodedDelta
	trimmed, err := json.Marshal(choice)
	if err != nil {
		return nil
	}
	return trimmed
}

// mayTrimChatChoice is a byte scan that lets most chunks pass untouched: those
// after the answer has started that carry no annotations to shift.
func (separator *answerSeparator) mayTrimChatChoice(raw []byte) bool {
	if !separator.settled {
		return true
	}
	return separator.removed[chatAnswerKey] > 0 && bytes.Contains(raw, []byte(`"annotations"`))
}

// trimChatObject drops the separator from the content of a Chat message or
// stream delta and shifts the offsets of its annotations to match.
func (separator *answerSeparator) trimChatObject(object map[string]json.RawMessage) bool {
	if chatObjectHasReasoning(object) {
		separator.observeReasoning()
	}
	changed := false
	var text *string
	if json.Unmarshal(object["content"], &text) == nil && text != nil {
		if trimmed := separator.trim(chatAnswerKey, *text); trimmed != *text {
			if encoded, err := json.Marshal(trimmed); err == nil {
				object["content"], changed = encoded, true
			}
		}
	}
	if annotations, found := object["annotations"]; found && separator.removed[chatAnswerKey] > 0 {
		if shifted, ok := separator.shiftChatAnnotations(annotations); ok {
			object["annotations"], changed = shifted, true
		}
	}
	return changed
}

func chatObjectHasReasoning(object map[string]json.RawMessage) bool {
	for _, field := range []string{"reasoning_content", "reasoning"} {
		var text string
		if json.Unmarshal(object[field], &text) == nil && text != "" {
			return true
		}
	}
	var details []json.RawMessage
	return json.Unmarshal(object["reasoning_details"], &details) == nil && len(details) > 0
}

func (separator *answerSeparator) shiftChatAnnotations(raw json.RawMessage) (json.RawMessage, bool) {
	var annotations []chatAnnotationWire
	if json.Unmarshal(raw, &annotations) != nil || len(annotations) == 0 {
		return nil, false
	}
	citations, err := decodeChatAnnotations(annotations)
	if err != nil {
		return nil, false
	}
	shifted, err := json.Marshal(encodeChatAnnotations(separator.shiftCitations(chatAnswerKey, citations)))
	return shifted, err == nil
}
