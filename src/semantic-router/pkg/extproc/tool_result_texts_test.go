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
						Kind: llmprotocol.ContentToolCall,
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
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("extractToolResultTexts() = %#v, want %#v", got, want)
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
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("extractToolResultTexts() = %#v, want %#v", got, want)
	}
}

func TestExtractToolResultTextsIncludesErrorTextAndIgnoresMissingContent(t *testing.T) {
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
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("extractToolResultTexts() = %#v, want %#v", got, want)
	}
}

func TestExtractToolResultTextsNilRequest(t *testing.T) {
	if got := extractToolResultTexts(nil); got != nil {
		t.Fatalf("extractToolResultTexts(nil) = %#v, want nil", got)
	}
}
