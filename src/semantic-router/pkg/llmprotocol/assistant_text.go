package llmprotocol

import "strings"

// TextContent joins the nonempty text blocks of a message with spaces.
func TextContent(contents []Content) string {
	parts := make([]string, 0, len(contents))
	for _, content := range contents {
		if content.Kind == ContentText && content.Text != "" {
			parts = append(parts, content.Text)
		}
	}
	return strings.Join(parts, " ")
}

// AssistantText renders the primary assistant output in order. Text and refusal
// parts form one prose run until a visible tool call. transformProse, when
// supplied, receives each complete run and never receives tool arguments.
func (response *Response) AssistantText(transformProse func(string) string) string {
	return response.renderAssistantText(transformProse, true)
}

// AssistantProse joins the primary assistant's text and refusal blocks in
// order. Consumers of a textual answer, such as a session title, use this
// projection even when the response also contains tool calls.
func (response *Response) AssistantProse() string {
	return response.renderAssistantText(nil, false)
}

func (response *Response) renderAssistantText(transformProse func(string) string, includeToolCalls bool) string {
	if response == nil {
		return ""
	}
	var sections []string
	var prose strings.Builder
	flushProse := func() {
		if prose.Len() == 0 {
			return
		}
		run := prose.String()
		if transformProse != nil {
			run = transformProse(run)
		}
		if run != "" {
			sections = append(sections, run)
		}
		prose.Reset()
	}
	for _, item := range response.Output {
		if item.Role != RoleAssistant {
			continue
		}
		for _, content := range item.Content {
			switch content.Kind {
			case ContentText, ContentRefusal:
				prose.WriteString(content.Text)
			case ContentToolCall:
				if !includeToolCalls {
					continue
				}
				if call := content.ToolCall.answerText(); call != "" {
					flushProse()
					sections = append(sections, call)
				}
			}
		}
	}
	flushProse()
	return strings.Join(sections, "\n")
}

// answerText renders the client-visible name and arguments of an assistant
// tool call for consumers of the answer as text.
func (call *ToolCall) answerText() string {
	if call == nil {
		return ""
	}
	name := strings.TrimSpace(call.Name)
	if ns := strings.TrimSpace(call.Namespace); ns != "" {
		if name == "" {
			name = ns
		} else {
			name = ns + "." + name
		}
	}
	if name == "" && call.Arguments == "" {
		return ""
	}
	header := "tool"
	if name != "" {
		header += " " + name
	}
	if call.Arguments == "" {
		return header
	}
	return header + "\n" + call.Arguments
}
