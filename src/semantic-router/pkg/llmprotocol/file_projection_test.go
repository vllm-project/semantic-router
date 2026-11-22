package llmprotocol

import (
	"strings"
	"testing"
)

// Claude Code returns a PDF it read as a file inside the tool result. A model
// without file input gets a note in its place, inside the same tool result,
// and the original request is left as it was.
func TestProjectUnreadableFilesLeavesANoteInPlace(t *testing.T) {
	pdf := Content{Kind: ContentFile, MediaType: "application/pdf", Data: "JVBERi0="}
	request := Request{
		Trusted: TrustedMetadata{SourceFormat: AnthropicMessagesV1},
		Messages: []Message{
			{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "summarize"}, {Kind: ContentFile, Filename: "notes.txt", MediaType: "text/plain", Data: "bm90ZXM="}}},
			{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "toolu_1", Content: []Content{{Kind: ContentText, Text: "PDF file read"}, pdf}}}}},
			{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "and now?"}}},
		},
	}
	projected, diagnostics := ProjectUnreadableFiles(request, OpenAIChatV1)

	if got := projected.Messages[0].Content[1]; got.Kind != ContentText || !strings.Contains(got.Text, "notes.txt") {
		t.Fatalf("message file = %+v", got)
	}
	result := projected.Messages[1].Content[0].ToolResult
	if len(result.Content) != 2 || result.Content[0].Text != "PDF file read" ||
		result.Content[1].Kind != ContentText || !strings.Contains(result.Content[1].Text, "application/pdf") {
		t.Fatalf("tool result = %+v", result.Content)
	}
	if request.Messages[1].Content[0].ToolResult.Content[1].Kind != ContentFile || request.Messages[0].Content[1].Kind != ContentFile {
		t.Fatal("the source request was changed")
	}
	if projected.Generation != request.Generation+1 {
		t.Fatalf("generation = %d", projected.Generation)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != LossyFieldUnreadableFile || diagnostics[0].Action != DiagnosticDropped ||
		!strings.HasPrefix(diagnostics[0].Reason, "2 files") {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
	if again, none := ProjectUnreadableFiles(projected, OpenAIChatV1); none != nil || again.Generation != projected.Generation {
		t.Fatalf("a request without files changed: %+v", none)
	}
}
