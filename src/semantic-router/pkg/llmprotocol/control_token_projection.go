package llmprotocol

import (
	"unicode/utf8"
)

// ControlTokenEscaper breaks a model's control tokens in text: the text of
// its tokenizer's added tokens, which a model server that tokenizes a
// rendered chat template reads as the tokens themselves (tokencount.Encoding
// escapes them with a word joiner).
type ControlTokenEscaper interface {
	// EscapeAddedTokens returns text with every control token in it broken,
	// and the byte offsets in text where a character went in; none when text
	// holds no control token.
	EscapeAddedTokens(text string) (string, []int)
	// OpensAddedToken reports whether text ends with the beginning of a
	// control token, which text written right after it could complete.
	OpensAddedToken(text string) bool
}

// ProjectControlTokens escapes the control tokens in the text a request
// supplies inside the turns a chat template frames: the text of user
// messages and of tool results. A model server that renders the template and
// tokenizes it with added tokens recognized would otherwise read a typed
// "<think>" or "<|im_end|>" as the template's own structure. Text parts that
// the template writes one after the other are escaped as one text: a part
// that ends with the beginning of a control token gets joiner, the
// character the escaper puts in, after it, so the next part cannot complete
// the token. Every other field, and every other role's text, stays as the
// client sent it. It reports whether it changed the request, and never edits
// the caller's slices.
func ProjectControlTokens(request Request, escaper ControlTokenEscaper, joiner string) (Request, bool) {
	var messages []Message
	for i, message := range request.Messages {
		escapeText := message.Role == RoleUser || message.Role == RoleTool
		content, changed := escapedControlTokens(message.Content, escapeText, escaper, joiner)
		if !changed {
			continue
		}
		if messages == nil {
			messages = append([]Message(nil), request.Messages...)
		}
		messages[i].Content = content
	}
	if messages == nil {
		return request, false
	}
	projected := request
	projected.Messages = messages
	projected.Generation++ // The source envelope still carries the client's text.
	return projected, true
}

// escapedControlTokens copies contents with the control tokens of their
// text escaped when escapeText is set, and of the text of every tool result
// among them in any case, and reports whether it changed anything.
func escapedControlTokens(contents []Content, escapeText bool, escaper ControlTokenEscaper, joiner string) ([]Content, bool) {
	var out []Content
	edit := func(i int, content Content) {
		if out == nil {
			out = append([]Content(nil), contents...)
		}
		out[i] = content
	}
	for i, content := range contents {
		switch {
		case content.Kind == ContentText && escapeText:
			text, at := escaper.EscapeAddedTokens(content.Text)
			nextIsText := i+1 < len(contents) && contents[i+1].Kind == ContentText
			if nextIsText && escaper.OpensAddedToken(text) {
				at = append(at, len(content.Text))
				text += joiner
			}
			if len(at) == 0 {
				continue
			}
			content.Citations = shiftedCitations(content.Citations, content.Text, at)
			content.Text = text
			edit(i, content)
		case content.Kind == ContentToolResult && content.ToolResult != nil:
			nested, changed := escapedControlTokens(content.ToolResult.Content, true, escaper, joiner)
			if !changed {
				continue
			}
			result := *content.ToolResult
			result.Content = nested
			content.ToolResult = &result
			edit(i, content)
		}
	}
	return out, out != nil
}

// shiftedCitations moves the citations of text, whose offsets count code
// points, past the characters that went in at the byte offsets at (one each,
// in order). A cited span that ends where a character went in does not take
// it.
func shiftedCitations(citations []Citation, text string, at []int) []Citation {
	if len(citations) == 0 {
		return citations
	}
	inserted := make([]int64, len(at))
	for i, offset := range at {
		inserted[i] = int64(utf8.RuneCountInString(text[:offset]))
	}
	shift := func(index int64, end bool) int64 {
		moved := index
		for _, point := range inserted {
			if point < index || (point == index && !end) {
				moved++
			}
		}
		return moved
	}
	out := make([]Citation, len(citations))
	for i, citation := range citations {
		citation.StartIndex, citation.EndIndex = shift(citation.StartIndex, false), shift(citation.EndIndex, true)
		out[i] = citation
	}
	return out
}
