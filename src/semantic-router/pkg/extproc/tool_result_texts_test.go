package extproc

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestExtractToolResultTextsPreservesOrderAndSelectsText(t *testing.T) {
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentText, Text: "user text"},
				},
			},
			{
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{
					{
						Kind:     llmprotocol.ContentToolCall,
						ToolCall: &llmprotocol.ToolCall{ID: "call-1", Name: "lookup"},
					},
					{
						Kind: llmprotocol.ContentToolResult,
						ToolResult: &llmprotocol.ToolResult{
							CallID: "call-1",
							Content: []llmprotocol.Content{
								{Kind: llmprotocol.ContentText, Text: "first"},
								{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"},
								{Kind: llmprotocol.ContentText, Text: "second"},
							},
						},
					},
				},
			},
			{
				Role: llmprotocol.RoleTool,
				Content: []llmprotocol.Content{
					{
						Kind: llmprotocol.ContentToolResult,
						ToolResult: &llmprotocol.ToolResult{
							CallID: "call-2",
							Content: []llmprotocol.Content{
								{Kind: llmprotocol.ContentText, Text: "third"},
							},
						},
					},
				},
			},
		},
	}

	got := extractToolResultTexts(req)
	want := []string{"first", "second", "third"}
	if !reflect.DeepEqual(got.texts, want) {
		t.Fatalf("extractToolResultTexts().texts = %#v, want %#v", got.texts, want)
	}
	if !got.incomplete || got.skippedBlocks != 1 {
		t.Fatalf("extraction status = %#v, want incomplete with one skipped block", got)
	}
}

func TestExtractToolResultTextsKeepsDuplicateTextForLaterDeduplication(t *testing.T) {
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{
				Role: llmprotocol.RoleTool,
				Content: []llmprotocol.Content{
					{
						Kind: llmprotocol.ContentToolResult,
						ToolResult: &llmprotocol.ToolResult{
							Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "same"}},
						},
					},
					{
						Kind: llmprotocol.ContentToolResult,
						ToolResult: &llmprotocol.ToolResult{
							Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "same"}},
						},
					},
				},
			},
		},
	}

	got := extractToolResultTexts(req)
	want := []string{"same", "same"}
	if !reflect.DeepEqual(got.texts, want) {
		t.Fatalf("extractToolResultTexts().texts = %#v, want %#v", got.texts, want)
	}
	if got.incomplete {
		t.Fatalf("duplicate text blocks should not make extraction incomplete: %#v", got)
	}
}

func TestExtractToolResultTextsIncludesErrorTextAndReportsMissingContent(t *testing.T) {
	isError := true
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{
				Role: llmprotocol.RoleTool,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentToolResult},
					{
						Kind: llmprotocol.ContentToolResult,
						ToolResult: &llmprotocol.ToolResult{
							IsError: &isError,
							Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "error details"}},
						},
					},
				},
			},
		},
	}

	got := extractToolResultTexts(req)
	want := []string{"error details"}
	if !reflect.DeepEqual(got.texts, want) {
		t.Fatalf("extractToolResultTexts().texts = %#v, want %#v", got.texts, want)
	}
	if !got.incomplete || got.skippedBlocks != 1 {
		t.Fatalf("extraction status = %#v, want incomplete with one skipped block", got)
	}
}

func TestExtractToolResultTextsIncludesTextBearingBlocks(t *testing.T) {
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentRefusal, Text: "refusal text"},
					{Kind: llmprotocol.ContentReasoning, Text: "reasoning text"},
				}},
			}},
		}},
	}

	got := extractToolResultTexts(req)
	if !reflect.DeepEqual(got.texts, []string{"refusal text", "reasoning text"}) || got.incomplete {
		t.Fatalf("extraction = %#v, want two text-bearing blocks and complete status", got)
	}
}

func TestExtractToolResultTextsReportsNonTextOnlyResultAsIncomplete(t *testing.T) {
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{Content: []llmprotocol.Content{{
					Kind:   llmprotocol.ContentFile,
					FileID: "file-1",
				}}},
			}},
		}},
	}

	got := extractToolResultTexts(req)
	if len(got.texts) != 0 || !got.incomplete || got.skippedBlocks != 1 {
		t.Fatalf("extraction = %#v, want no texts and one skipped block", got)
	}
}

func TestExtractToolResultTextsNilRequest(t *testing.T) {
	if got := extractToolResultTexts(nil); got.texts != nil || got.incomplete || got.skippedBlocks != 0 {
		t.Fatalf("extractToolResultTexts(nil) = %#v, want empty extraction", got)
	}
}

// An Anthropic server tool's result, such as a fetched page, is tool output
// the tool_result source reads; its call and the result's metadata are not. A
// fetched PDF and a search result's encrypted content stay unscanned.
func TestExtractToolResultTextsReadsServerToolResults(t *testing.T) {
	call := llmprotocol.Content{Kind: llmprotocol.ContentServerTool, Raw: []byte(`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_fetch","input":{"url":"https://example.invalid"}}`)}
	for name, test := range map[string]struct {
		result     string
		texts      []string
		incomplete bool
	}{
		"fetched text": {
			result: `{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_1","content":{"type":"web_fetch_result","url":"https://example.invalid/u/bob","content":{"type":"document","title":"bob@example.com","source":{"type":"text","media_type":"text/plain","data":"owner: alice@example.com"}}}}`,
			texts:  []string{"owner: alice@example.com", "bob@example.com", "https://example.invalid/u/bob"},
		},
		"fetched pdf": {
			result:     `{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_1","content":{"type":"web_fetch_result","url":"https://example.invalid/a.pdf","content":{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0xLjQ="}}}}`,
			texts:      []string{"https://example.invalid/a.pdf"},
			incomplete: true,
		},
		"search results": {
			result:     `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://example.invalid","title":"A page","encrypted_content":"EqgfCioIARgBIiQ3"}]}`,
			incomplete: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			req := &llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{call, {Kind: llmprotocol.ContentServerTool, Raw: []byte(test.result)}},
			}}}
			got := extractToolResultTexts(req)
			if got.incomplete != test.incomplete || !reflect.DeepEqual(got.texts, test.texts) {
				t.Fatalf("extraction = %+v, want texts %q and incomplete %v", got, test.texts, test.incomplete)
			}
		})
	}
}
