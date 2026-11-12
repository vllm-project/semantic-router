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
