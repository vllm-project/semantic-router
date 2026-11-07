package llmprotocol

import "fmt"

// ProjectUnreadableFiles replaces each file in the conversation, in a message
// or inside a tool result, with a short note saying it was not sent. A coding
// agent that reads a PDF returns it as a file in a tool result, and every later
// call resends that result, so a model without file input would otherwise fail
// the rest of the session. With the note the agent can read the file another
// way. The caller applies it only when the operator lists
// LossyFieldUnreadableFile and the selected model declares no file input, and
// must report the returned diagnostic.
func ProjectUnreadableFiles(request Request, target WireFormat) (Request, Diagnostics) {
	count := 0
	messages := make([]Message, len(request.Messages))
	for i, message := range request.Messages {
		var replaced int
		message.Content, replaced = replaceFiles(message.Content)
		messages[i] = message
		count += replaced
	}
	if count == 0 {
		return request, nil
	}
	projected := request
	projected.Messages = messages
	projected.Generation++ // The source envelope still carries the files.
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat,
		Target: target,
		Field:  LossyFieldUnreadableFile,
		Action: DiagnosticDropped,
		Reason: fmt.Sprintf("%d files the selected model cannot read sent as a note", count),
	}}
}

func replaceFiles(contents []Content) ([]Content, int) {
	count := 0
	out := make([]Content, len(contents))
	for i, content := range contents {
		switch {
		case content.Kind == ContentFile:
			content = Content{Kind: ContentText, Text: unreadableFileNote(content), Cache: content.Cache}
			count++
		case content.Kind == ContentToolResult && content.ToolResult != nil:
			inner, replaced := replaceFiles(content.ToolResult.Content)
			if replaced > 0 {
				result := *content.ToolResult
				result.Content = inner
				content.ToolResult = &result
				count += replaced
			}
		}
		out[i] = content
	}
	if count == 0 {
		return contents, 0
	}
	return out, count
}

func unreadableFileNote(file Content) string {
	name := file.Filename
	if name == "" {
		name = file.MediaType
	}
	if name == "" {
		name = "file"
	}
	return fmt.Sprintf("A file (%s) was not sent because the selected model cannot read files.", name)
}
