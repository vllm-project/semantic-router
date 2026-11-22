package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A video rule reads every video the request carries: one in an earlier user
// turn and one in a tool result are collected beside the last turn's, and a
// video that is not an inline data URL carries as the empty string so the
// reader can report the input incomplete.
func TestExtractSemanticRequestSignals_VideosFromEveryTurn(t *testing.T) {
	request := &llmprotocol.Request{Model: "general", Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentVideo, MediaType: "video/mp4", Data: "ZWFybGllcg=="},
		}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c1", Name: "fetch"}},
		}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c1", Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentVideo, MediaType: "video/webm", Data: "ZnJvbXRvb2w="},
			}}},
		}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "watch these"},
			{Kind: llmprotocol.ContentVideo, URL: "https://cdn.example/clip.mp4"},
		}},
	}}
	videos := extractSemanticRequestSignals(request).UserVideos
	want := []string{
		"data:video/mp4;base64,ZWFybGllcg==",
		"data:video/webm;base64,ZnJvbXRvb2w=",
		"",
	}
	if len(videos) != len(want) {
		t.Fatalf("videos = %q, want %q", videos, want)
	}
	for index, video := range want {
		if videos[index] != video {
			t.Fatalf("videos[%d] = %q, want %q", index, videos[index], video)
		}
	}
}
