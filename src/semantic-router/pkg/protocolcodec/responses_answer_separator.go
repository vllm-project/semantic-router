package protocolcodec

import (
	"bytes"
	"encoding/json"
)

// trimReplayAnswerSeparator drops the separator from a Responses reply's
// first output_text part and reports false when nothing changed, the body is
// not a Responses reply, or an item has a shape this patch does not
// recognize. The caller falls back to the canonical re-encode in that case.
func (OpenAIResponsesCodec) trimReplayAnswerSeparator(replay []byte) ([]byte, bool) {
	var reply map[string]json.RawMessage
	var output []json.RawMessage
	if json.Unmarshal(replay, &reply) != nil || json.Unmarshal(reply["output"], &output) != nil {
		return nil, false
	}
	var separator answerSeparator
	changed := false
	for itemIndex, rawItem := range output {
		var item map[string]json.RawMessage
		var itemType string
		if json.Unmarshal(rawItem, &item) != nil || json.Unmarshal(item["type"], &itemType) != nil {
			return nil, false
		}
		switch itemType {
		case "reasoning":
			if responsesItemHasReasoning(item) {
				separator.observeReasoning()
			}
		case "message":
			patchedItem, itemChanged, ok := trimResponsesMessageItem(&separator, itemIndex, item)
			if !ok {
				return nil, false
			}
			if itemChanged {
				output[itemIndex] = patchedItem
				changed = true
			}
		}
	}
	if !changed {
		return nil, false
	}
	encodedOutput, err := json.Marshal(output)
	if err != nil {
		return nil, false
	}
	reply["output"] = encodedOutput
	patched, err := json.Marshal(reply)
	return patched, err == nil
}

// trimResponsesMessageItem drops the separator from a message item's first
// output_text part and shifts that part's citation offsets to match.
func trimResponsesMessageItem(separator *answerSeparator, itemIndex int, item map[string]json.RawMessage) (json.RawMessage, bool, bool) {
	var parts []json.RawMessage
	if json.Unmarshal(item["content"], &parts) != nil {
		return nil, false, false
	}
	changed := false
	for partIndex, rawPart := range parts {
		var part map[string]json.RawMessage
		var partType string
		if json.Unmarshal(rawPart, &part) != nil || json.Unmarshal(part["type"], &partType) != nil {
			return nil, false, false
		}
		if partType != "output_text" {
			continue
		}
		var text string
		if json.Unmarshal(part["text"], &text) != nil {
			return nil, false, false
		}
		key := streamContentKey{item: itemIndex, content: partIndex}
		trimmed := separator.trim(key, text)
		if trimmed == text {
			continue
		}
		encodedText, err := json.Marshal(trimmed)
		if err != nil {
			return nil, false, false
		}
		part["text"] = encodedText
		if separator.removed[key] > 0 {
			if !shiftResponsesPartAnnotations(separator, key, part) {
				return nil, false, false
			}
		}
		patchedPart, err := json.Marshal(part)
		if err != nil {
			return nil, false, false
		}
		parts[partIndex] = patchedPart
		changed = true
	}
	if !changed {
		return nil, false, true
	}
	encodedParts, err := json.Marshal(parts)
	if err != nil {
		return nil, false, false
	}
	item["content"] = encodedParts
	patchedItem, err := json.Marshal(item)
	if err != nil {
		return nil, false, false
	}
	return patchedItem, true, true
}

func shiftResponsesPartAnnotations(separator *answerSeparator, key streamContentKey, part map[string]json.RawMessage) bool {
	annotations, found := part["annotations"]
	if !found || len(bytes.TrimSpace(annotations)) == 0 {
		return true
	}
	var objects []map[string]json.RawMessage
	if json.Unmarshal(annotations, &objects) != nil {
		return false
	}
	if len(objects) == 0 {
		return true
	}
	removed := separator.removed[key]
	for _, object := range objects {
		var annotationType string
		if json.Unmarshal(object["type"], &annotationType) != nil || annotationType != "url_citation" {
			return false
		}
		var start, end int64
		if json.Unmarshal(object["start_index"], &start) != nil || json.Unmarshal(object["end_index"], &end) != nil {
			return false
		}
		startEncoded, err := json.Marshal(max(start-removed, 0))
		if err != nil {
			return false
		}
		endEncoded, err := json.Marshal(max(end-removed, 0))
		if err != nil {
			return false
		}
		object["start_index"], object["end_index"] = startEncoded, endEncoded
	}
	encoded, err := json.Marshal(objects)
	if err != nil {
		return false
	}
	part["annotations"] = encoded
	return true
}

func responsesItemHasReasoning(item map[string]json.RawMessage) bool {
	if responsesTextPartsCarryText(item["summary"], "summary_text") {
		return true
	}
	if responsesTextPartsCarryText(item["content"], "reasoning_text") {
		return true
	}
	// Encrypted reasoning carries no visible text but is still reasoning
	// content once present, matching decodeResponsesEncryptedReasoning's own
	// null check.
	encrypted := bytes.TrimSpace(item["encrypted_content"])
	return len(encrypted) > 0 && !bytes.Equal(encrypted, []byte("null"))
}

func responsesTextPartsCarryText(raw json.RawMessage, partType string) bool {
	var parts []map[string]json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return false
	}
	for _, part := range parts {
		var typ, text string
		if json.Unmarshal(part["type"], &typ) != nil || typ != partType {
			continue
		}
		if json.Unmarshal(part["text"], &text) == nil && text != "" {
			return true
		}
	}
	return false
}
