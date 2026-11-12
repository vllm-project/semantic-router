package llmprotocol

import "testing"

func TestStartsConversationTurn(t *testing.T) {
	text := Content{Kind: ContentText, Text: "hi"}
	result := Content{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "c1"}}
	image := Content{Kind: ContentImage, URL: "https://example.invalid/a.png"}
	cases := []struct {
		name    string
		message Message
		want    bool
	}{
		{"user text", Message{Role: RoleUser, Content: []Content{text}}, true},
		{"user image only", Message{Role: RoleUser, Content: []Content{image}}, true},
		{"user tool results only", Message{Role: RoleUser, Content: []Content{result, result}}, false},
		{"user tool result and text", Message{Role: RoleUser, Content: []Content{result, text}}, true},
		{"user empty text block", Message{Role: RoleUser, Content: []Content{{Kind: ContentText}}}, true},
		{"user without content", Message{Role: RoleUser}, false},
		{"assistant text", Message{Role: RoleAssistant, Content: []Content{text}}, false},
		{"tool role", Message{Role: RoleTool, Content: []Content{result}}, false},
	}
	for _, tc := range cases {
		if got := StartsConversationTurn(tc.message); got != tc.want {
			t.Errorf("%s: StartsConversationTurn = %v, want %v", tc.name, got, tc.want)
		}
	}
}
