package protocolcodec

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// answerSeparator removes the whitespace a backend's reasoning parser leaves
// between the end of the reasoning and the answer: vLLM splits Qwen output at
// the closing think tag and keeps the blank line the model writes after it.
// Only the leading whitespace of the first visible answer text after reasoning
// goes. Replies without reasoning, the reasoning itself and tool calls are
// never touched.
type answerSeparator struct {
	reasoned bool // reasoning came before this point
	settled  bool // visible answer text has started
	// removed counts the runes dropped from the start of each answer block.
	// Citation offsets index into the text as the provider sent it.
	removed map[streamContentKey]int64
}

func (separator *answerSeparator) observeReasoning() { separator.reasoned = true }

func (separator *answerSeparator) trim(key streamContentKey, text string) string {
	if separator.settled {
		return text
	}
	trimmed := strings.TrimLeftFunc(text, unicode.IsSpace)
	separator.settled = trimmed != ""
	if !separator.reasoned {
		return text
	}
	if runes := utf8.RuneCountInString(text[:len(text)-len(trimmed)]); runes > 0 {
		if separator.removed == nil {
			separator.removed = make(map[streamContentKey]int64)
		}
		separator.removed[key] += int64(runes)
	}
	return trimmed
}

func (separator *answerSeparator) shiftCitations(key streamContentKey, citations []llmprotocol.Citation) []llmprotocol.Citation {
	removed := separator.removed[key]
	if removed == 0 || len(citations) == 0 {
		return citations
	}
	shifted := slices.Clone(citations)
	for index := range shifted {
		shifted[index].StartIndex = max(shifted[index].StartIndex-removed, 0)
		shifted[index].EndIndex = max(shifted[index].EndIndex-removed, 0)
	}
	return shifted
}

// carriesReasoning is false for the empty reasoning some providers send beside
// an answer that was never preceded by thinking.
func carriesReasoning(content llmprotocol.Content) bool {
	return content.Kind == llmprotocol.ContentReasoning &&
		(content.Text != "" || content.Signature != "" || content.ReasoningDetail != nil)
}

// trimResponseAnswerSeparator removes the separator from every choice of a
// decoded reply and reports whether any text changed.
func trimResponseAnswerSeparator(response *llmprotocol.Response) bool {
	var changed bool
	response.Output, changed = trimItemsAnswerSeparator(response.Output)
	for index := range response.Alternatives {
		var alternativeChanged bool
		response.Alternatives[index], alternativeChanged = trimItemsAnswerSeparator(response.Alternatives[index])
		changed = changed || alternativeChanged
	}
	return changed
}

func trimItemsAnswerSeparator(items []llmprotocol.OutputItem) ([]llmprotocol.OutputItem, bool) {
	hasReasoning := slices.ContainsFunc(items, func(item llmprotocol.OutputItem) bool {
		return slices.ContainsFunc(item.Content, carriesReasoning)
	})
	if !hasReasoning {
		return items, false
	}
	var separator answerSeparator
	changed := false
	kept := make([]llmprotocol.OutputItem, 0, len(items))
	for itemIndex, item := range items {
		content := make([]llmprotocol.Content, 0, len(item.Content))
		for blockIndex, block := range item.Content {
			if carriesReasoning(block) {
				separator.observeReasoning()
			} else if block.Kind == llmprotocol.ContentText {
				key := streamContentKey{item: itemIndex, content: blockIndex}
				if trimmed := separator.trim(key, block.Text); trimmed != block.Text {
					changed = true
					block.Text = trimmed
					block.Citations = separator.shiftCitations(key, block.Citations)
					if trimmed == "" && len(block.Citations) == 0 {
						continue
					}
				}
			}
			content = append(content, block)
		}
		if len(item.Content) > 0 && len(content) == 0 {
			continue
		}
		item.Content = content
		kept = append(kept, item)
	}
	if !changed {
		return items, false
	}
	return kept, true
}

// trimDecodedAnswerSeparator drops the separator from a decoded reply and keeps
// its envelope from replaying provider bytes that still hold it. A codec that
// can patch its own replay keeps the rest of the provider body; any other
// reply is encoded again from the neutral response.
func trimDecodedAnswerSeparator(
	codec Codec,
	replayable bool,
	response *llmprotocol.Response,
	envelope *llmprotocol.Envelope,
) {
	if !trimResponseAnswerSeparator(response) {
		return
	}
	if trimmer, ok := codec.(replayAnswerSeparatorTrimmer); ok && replayable && len(envelope.Response) > 0 {
		if patched, patchedOK := trimmer.trimReplayAnswerSeparator(envelope.Response); patchedOK {
			envelope.Response = patched
			return
		}
	}
	response.Generation++
	envelope.ResponseReencodeRequired = true
}

// trimEvent applies the rule to a decoded stream event. The separator can
// arrive as a delta of its own, split over several deltas, or in front of the
// first word, so whitespace is dropped until visible text starts. It trims an
// answer text delta in place and returns ErrOmitStreamEvent for a delta that
// held only the separator.
func (separator *answerSeparator) trimEvent(event *llmprotocol.Event) error {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		if event.Content != nil && carriesReasoning(*event.Content) {
			separator.observeReasoning()
		}
	case llmprotocol.EventReasoningDelta:
		if event.Delta != "" || event.Content != nil && carriesReasoning(*event.Content) {
			separator.observeReasoning()
		}
	case llmprotocol.EventOutputTextDelta:
		return separator.trimEventText(event)
	}
	return nil
}

func (separator *answerSeparator) trimEventText(event *llmprotocol.Event) error {
	if event.Content != nil && event.Content.Kind != llmprotocol.ContentText {
		return nil
	}
	key := contentKey(*event)
	trimmed := separator.trim(key, event.Delta)
	hasCitations := event.Content != nil && len(event.Content.Citations) > 0
	if trimmed == event.Delta && (!hasCitations || separator.removed[key] == 0) {
		return nil
	}
	if trimmed == "" && !hasCitations {
		return ErrOmitStreamEvent
	}
	event.Delta = trimmed
	if event.Content != nil {
		content := *event.Content
		content.Text = trimmed
		content.Citations = separator.shiftCitations(key, content.Citations)
		event.Content = &content
	}
	return nil
}
