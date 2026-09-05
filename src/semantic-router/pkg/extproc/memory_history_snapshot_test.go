package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestExtractMemoryInfo_DetachesNestedHistory(t *testing.T) {
	isError, imageResult, partialIndex := false, "original-image", int64(1)
	reqCtx := persistenceRegressionContext("nested-history")
	reqCtx.SemanticRequest.Messages[0].Content = []llmprotocol.Content{
		{
			Kind:      llmprotocol.ContentText,
			Text:      "original text",
			Citations: []llmprotocol.Citation{{URL: "https://example.com/original"}},
			Cache:     &llmprotocol.CacheDirective{Type: "ephemeral", TTL: "5m"},
		},
		{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: "call", Name: "lookup", Arguments: "original arguments"},
		},
		{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{
				CallID:  "call",
				IsError: &isError,
				Content: []llmprotocol.Content{{
					Kind:      llmprotocol.ContentText,
					Text:      "original tool result",
					Citations: []llmprotocol.Citation{{Title: "original nested citation"}},
				}},
			},
		},
		{
			Kind: llmprotocol.ContentGeneratedImage,
			GeneratedImage: &llmprotocol.GeneratedImage{
				Result:       &imageResult,
				PartialIndex: &partialIndex,
				Size:         "original size",
			},
		},
	}
	// Serialize the expected value before mutation, independently of the clone helper.
	want, err := json.Marshal(reqCtx.SemanticRequest.Messages)
	require.NoError(t, err)
	sessionID, userID, history, err := extractMemoryInfo(reqCtx)
	require.NoError(t, err)

	reqCtx.SessionID = "changed-session"
	reqCtx.Headers[headers.AuthzUserID] = "changed-user"
	reqCtx.SemanticRequest.Messages[0].Role = llmprotocol.RoleAssistant
	content := reqCtx.SemanticRequest.Messages[0].Content
	content[0].Text = "changed text"
	content[0].Citations[0].URL = "https://example.com/changed"
	content[0].Cache.TTL = "1h"
	content[1].ToolCall.Arguments = "changed arguments"
	content[2].ToolResult.CallID = "changed-call"
	content[2].ToolResult.Content[0].Text = "changed tool result"
	content[2].ToolResult.Content[0].Citations[0].Title = "changed nested citation"
	content[3].GeneratedImage.Size = "changed size"
	isError, imageResult, partialIndex = true, "changed-image", 99

	got, err := json.Marshal(history)
	require.NoError(t, err)
	assert.JSONEq(t, string(want), string(got))
	assert.Equal(t, "original-session", sessionID)
	assert.Equal(t, "original-user", userID)
}
