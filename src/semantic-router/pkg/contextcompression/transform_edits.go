package contextcompression

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (request *RequestIR) validateEdits(kind TransformationKind, edits TransformationEdits) error {
	if (kind == TransformCompress && len(edits.RemoveMessages) > 0) || (kind != TransformCompress && len(edits.ReplaceText) > 0) {
		return fmt.Errorf("edit does not match declared output")
	}
	messages := make(map[int]*MessageIR, len(request.Messages))
	for _, message := range request.Messages {
		messages[message.Index] = message
	}
	removed := make(map[int]bool, len(edits.RemoveMessages))
	for _, id := range edits.RemoveMessages {
		message := messages[id]
		if message == nil || removed[id] || message.Eligibility&EligibleHistoryRemoval == 0 {
			return fmt.Errorf("message is not eligible")
		}
		removed[id] = true
	}
	if err := request.validateRemovalGroups(kind, removed); err != nil {
		return err
	}
	return request.validateTextReplacements(messages, edits.ReplaceText)
}

func (request *RequestIR) validateTextReplacements(messages map[int]*MessageIR, replacements []TextReplacement) error {
	replaced := make(map[[2]int]bool)
	for _, replacement := range replacements {
		key := [2]int{replacement.MessageID, replacement.BlockID}
		message := messages[replacement.MessageID]
		block := findBlock(message, replacement.BlockID)
		if block == nil || replaced[key] || !request.compressionBlockAllowed(message, block) {
			return fmt.Errorf("block is not eligible")
		}
		replaced[key] = true
	}
	return nil
}

func (request *RequestIR) validateRemovalGroups(kind TransformationKind, removed map[int]bool) error {
	turns := make(map[int]bool)
	exchanges := make(map[string]bool)
	for _, message := range request.Messages {
		if removed[message.Index] {
			turns[message.TurnID] = true
			for _, id := range message.ExchangeIDs {
				exchanges[id] = true
			}
		}
	}
	for _, message := range request.Messages {
		if removed[message.Index] {
			continue
		}
		if kind != TransformDeduplicate && turns[message.TurnID] {
			return fmt.Errorf("incomplete conversation turn")
		}
		for _, id := range message.ExchangeIDs {
			if exchanges[id] {
				return fmt.Errorf("incomplete tool exchange")
			}
		}
	}
	return request.validateRawRemoval(removed)
}

func (request *RequestIR) validateRawRemoval(removed map[int]bool) error {
	if len(removed) > 0 && request.Semantic == nil {
		raw, _ := request.Raw["messages"].([]interface{})
		if len(raw) != len(request.Messages) {
			return fmt.Errorf("unrecognized raw messages")
		}
	}
	return nil
}

func (request *RequestIR) compressionBlockAllowed(message *MessageIR, block *TextBlockIR) bool {
	if message.Protection&(ProtectInstructions|ProtectAuthorization|ProtectSafety) != 0 {
		return false
	}
	// Enabling a history policy opts into the shared live-turn protection.
	// With no new policy enabled, retain the legacy compression eligibility.
	if request.Transformations.historyEnabled && block.Source == TargetHistory &&
		message.Protection&(ProtectLiveTurn|ProtectMultimodal|ProtectUnknown) != 0 {
		return false
	}
	// Preserve configured tool/RAG/Memory text compression, including results
	// in the live exchange. Calls, IDs, media and metadata are not editable.
	return block.Source != TargetHistory || (!message.Protected && message.Role != "tool" && message.Role != "function")
}

func findBlock(message *MessageIR, id int) *TextBlockIR {
	if message != nil {
		for _, block := range message.Blocks {
			if block.BlockIndex == id {
				return block
			}
		}
	}
	return nil
}

// commitEdits retains surviving semantic content and codec metadata. Undo is
// used only by the compressor's non-reducing-budget check.
func (request *RequestIR) commitEdits(edits TransformationEdits) func() {
	var undo []func()
	for _, replacement := range edits.ReplaceText {
		for _, message := range request.Messages {
			if message.Index != replacement.MessageID {
				continue
			}
			block := findBlock(message, replacement.BlockID)
			original := block.Text
			undo = append(undo, func() { block.SetText(original) })
			block.SetText(replacement.Text)
		}
	}
	if len(edits.RemoveMessages) > 0 {
		request.removeMessages(edits.RemoveMessages)
	}
	return func() {
		for _, restore := range undo {
			restore()
		}
	}
}

func (request *RequestIR) removeMessages(ids []int) {
	removed := make(map[int]bool, len(ids))
	for _, id := range ids {
		removed[id] = true
	}
	kept := make([]*MessageIR, 0, len(request.Messages)-len(ids))
	var semantic []llmprotocol.Message
	var raw []interface{}
	for position, message := range request.Messages {
		if removed[message.Index] {
			continue
		}
		kept = append(kept, message)
		if request.Semantic != nil {
			semantic = append(semantic, request.Semantic.Messages[position])
		} else {
			raw = append(raw, request.Raw["messages"].([]interface{})[position])
		}
	}
	request.Messages = kept
	if request.Semantic != nil {
		request.Semantic.Messages = semantic
		request.Semantic.Generation++
	} else {
		request.Raw["messages"] = raw
	}
}
