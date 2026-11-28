package llmprotocol

import (
	"cmp"
	"fmt"
	"slices"
)

// ProjectSystemMessages sends a request's system and developer messages in a
// form that chat templates accepting one system message, first, can read.
// Qwen3.5 and later templates reject a second system message, or one after
// the first position, and others drop it, so the same request would pass or
// fail with the selected model. The caller applies it only when the operator
// lists LossyFieldSystemMessageRole and must report the returned diagnostic.
//
// A Chat backend receives the instructions that lead the conversation as one
// system message, their content in order with a blank line between them: a
// Chat client's leading system and developer messages, a Responses request's
// instructions and leading developer and system items, or a Messages
// request's system prompt. A single system instruction is left as it is.
//
// A system or developer message inside the conversation becomes user text in
// the same place, whichever client protocol sent it, toward Chat and Responses
// backends. One between a tool call and its results first moves after the
// results. Claude Code inserts these messages at the point they refer to (a
// file changed on disk, a truncated read, a new working directory), so the
// text joins the user message right before it, or becomes a user message of
// its own after a tool result or an assistant turn. Earlier turns are never
// rewritten, which keeps the conversation's cached prefix. A message that also
// sets per-message effort keeps an empty system turn, so that control is still
// reported as dropped by the encoder.
func ProjectSystemMessages(request Request, target WireFormat) (Request, Diagnostics) {
	if target != OpenAIChatV1 && target != OpenAIResponsesV1 {
		return request, nil
	}
	request, omitted := omitEmptyInstructionMessages(request)
	var leading, later []InstructionBlock
	if target == OpenAIChatV1 {
		for _, instruction := range request.Instructions {
			if instruction.Position > 0 {
				later = append(later, instruction)
			} else {
				leading = append(leading, instruction)
			}
		}
	}
	merged := 0
	if len(leading) > 1 || len(leading) == 1 && leading[0].Role != RoleSystem {
		merged = len(leading)
		leading = []InstructionBlock{oneSystemInstruction(leading)}
	}
	projected, diagnostics := ProjectInstructionsAfterToolResults(request, target)
	messages, inPlace, changed := instructionsAsUserText(projected.Messages, later)
	if merged == 0 && !changed && !omitted {
		return projected, diagnostics
	}
	if target == OpenAIChatV1 {
		projected.Instructions = leading
	}
	projected.Messages = messages
	projected.Generation++ // The source envelope still carries the messages as the client sent them.
	if merged == 0 && inPlace == 0 {
		return projected, diagnostics
	}
	reason := fmt.Sprintf("%d system messages sent as user text in place", inPlace)
	switch {
	case merged > 0 && inPlace > 0:
		reason = fmt.Sprintf("%d system and developer messages sent as one system message, %d as user text in place", merged, inPlace)
	case merged > 0:
		reason = fmt.Sprintf("%d system and developer messages sent as one system message", merged)
	}
	return projected, append(diagnostics, Diagnostic{
		Source: request.Trusted.SourceFormat, Target: target, Field: LossyFieldSystemMessageRole,
		Action: DiagnosticApproximated, Reason: reason,
	})
}

// Omit instructions that this projection would discard before tool placement
// can report moving them. Effort controls still need their own message, and
// positioned instruction blocks must retain their place among the kept turns.
func omitEmptyInstructionMessages(request Request) (Request, bool) {
	empty := func(message Message) bool {
		return isInstructionMessage(message) && message.ReasoningEffort == "" && emptyText(message.Content)
	}
	if !slices.ContainsFunc(request.Messages, empty) {
		return request, false
	}
	projected := request
	projected.Messages = make([]Message, 0, len(request.Messages))
	kept := make([]bool, len(request.Messages))
	for index, message := range request.Messages {
		kept[index] = !empty(message)
		if kept[index] {
			projected.Messages = append(projected.Messages, message)
		}
	}
	projected.Instructions = KeepInstructionPositions(request.Instructions, kept)
	return projected, true
}

// instructionsAsUserText sends the later instructions at their positions, and
// the system and developer messages inside the conversation, as user text, and
// returns how many it sent. A later instruction without text is left out.
func instructionsAsUserText(conversation []Message, later []InstructionBlock) ([]Message, int, bool) {
	slices.SortStableFunc(later, func(a, b InstructionBlock) int { return cmp.Compare(a.Position, b.Position) })
	messages := make([]Message, 0, len(conversation)+len(later))
	count := 0
	changed := false
	asUserText := func(content []Content) {
		if last := len(messages) - 1; last >= 0 && messages[last].Role == RoleUser {
			messages[last].Content = append(append([]Content(nil), messages[last].Content...), content...)
		} else {
			messages = append(messages, Message{Role: RoleUser, Content: content})
		}
	}
	for index, next := 0, 0; ; index++ {
		for ; next < len(later) && (later[next].Position <= index || index == len(conversation)); next++ {
			changed = true
			if !emptyText(later[next].Content) {
				count++
				asUserText(later[next].Content)
			}
		}
		if index == len(conversation) {
			break
		}
		message := conversation[index]
		if !isInstructionMessage(message) || len(message.Content) == 0 && message.ReasoningEffort != "" {
			messages = append(messages, message)
			continue
		}
		changed = true
		if !emptyText(message.Content) {
			count++
			asUserText(message.Content)
		}
		if message.ReasoningEffort != "" {
			messages = append(messages, Message{ID: message.ID, Role: RoleSystem, ReasoningEffort: message.ReasoningEffort})
		}
	}
	if !changed {
		return conversation, 0, false
	}
	return messages, count, true
}

// oneSystemInstruction joins instructions into one system instruction with a
// blank line between them. Text runs into the plain text before it, so the
// result reads as one text, and a cache boundary stays at the end of the
// content that set it.
func oneSystemInstruction(instructions []InstructionBlock) InstructionBlock {
	joined := InstructionBlock{Role: RoleSystem, Name: instructions[0].Name}
	for _, instruction := range instructions {
		if instruction.Name != joined.Name {
			joined.Name = ""
		}
		joined.Content = appendInstructionContent(joined.Content, instruction.Content)
	}
	if len(joined.Content) == 0 {
		joined.Content = instructions[0].Content
	}
	return joined
}

func appendInstructionContent(joined, content []Content) []Content {
	if emptyText(content) {
		return joined
	}
	if len(joined) == 0 {
		return append(joined, content...)
	}
	first, last := content[0], &joined[len(joined)-1]
	plainLast := last.Kind == ContentText && last.Cache == nil && len(last.Citations) == 0
	textFirst := first.Kind == ContentText && len(first.Citations) == 0
	switch {
	case plainLast && textFirst:
		last.Text += "\n\n" + first.Text
		last.Cache = first.Cache
		return append(joined, content[1:]...)
	case plainLast:
		last.Text += "\n\n"
	case textFirst:
		first.Text = "\n\n" + first.Text
		return append(append(joined, first), content[1:]...)
	default:
		joined = append(joined, Content{Kind: ContentText, Text: "\n\n"})
	}
	return append(joined, content...)
}

func emptyText(content []Content) bool {
	for _, part := range content {
		if part.Kind != ContentText || part.Text != "" || part.Cache != nil {
			return false
		}
	}
	return true
}
