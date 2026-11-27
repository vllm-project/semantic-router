package protocolcodec

import "encoding/json"

// trimReplayAnswerSeparator drops the separator from a Messages reply's
// first text block and reports false when nothing changed, the body is not a
// Messages reply, or a block has a shape this patch does not recognize. The
// caller falls back to the canonical re-encode in that case. Messages
// rejects citations on decode (validateAnthropicContentExtensions), so no
// offsets ever need shifting here.
func (AnthropicMessagesCodec) trimReplayAnswerSeparator(replay []byte) ([]byte, bool) {
	var reply map[string]json.RawMessage
	var content []json.RawMessage
	if json.Unmarshal(replay, &reply) != nil || json.Unmarshal(reply["content"], &content) != nil {
		return nil, false
	}
	var separator answerSeparator
	changed := false
	for index, raw := range content {
		var block map[string]json.RawMessage
		var blockType string
		if json.Unmarshal(raw, &block) != nil || json.Unmarshal(block["type"], &blockType) != nil {
			return nil, false
		}
		switch blockType {
		case "thinking":
			if anthropicBlockHasReasoning(block) {
				separator.observeReasoning()
			}
			continue
		case "text":
		default:
			continue
		}
		var text string
		if json.Unmarshal(block["text"], &text) != nil {
			return nil, false
		}
		trimmed := separator.trim(streamContentKey{content: index}, text)
		if trimmed == text {
			continue
		}
		encoded, err := json.Marshal(trimmed)
		if err != nil {
			return nil, false
		}
		block["text"] = encoded
		patchedBlock, err := json.Marshal(block)
		if err != nil {
			return nil, false
		}
		content[index] = patchedBlock
		changed = true
	}
	if !changed {
		return nil, false
	}
	encodedContent, err := json.Marshal(content)
	if err != nil {
		return nil, false
	}
	reply["content"] = encodedContent
	patched, err := json.Marshal(reply)
	return patched, err == nil
}

func anthropicBlockHasReasoning(block map[string]json.RawMessage) bool {
	var thinking, signature string
	hasThinking := json.Unmarshal(block["thinking"], &thinking) == nil && thinking != ""
	hasSignature := json.Unmarshal(block["signature"], &signature) == nil && signature != ""
	return hasThinking || hasSignature
}
