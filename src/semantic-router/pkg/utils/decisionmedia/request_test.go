package decisionmedia

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestFromRequestIncludesOnlyUserAttachmentsInOrder(t *testing.T) {
	request := &llmprotocol.Request{}
	for _, role := range []llmprotocol.Role{llmprotocol.RoleSystem, llmprotocol.RoleUser, llmprotocol.RoleAssistant, llmprotocol.RoleTool, llmprotocol.RoleUser} {
		request.Messages = append(request.Messages, llmprotocol.Message{Role: role, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "text"},
			{Kind: llmprotocol.ContentImage, URL: string(role) + " image"},
			{Kind: llmprotocol.ContentVideo, Data: "video", MediaType: "video/mp4"},
		}})
	}
	sources := FromRequest(request)
	require.Len(t, sources, 4)
	for i := 0; i < len(sources); i += 2 {
		require.Equal(t, "user image", sources[i].URL)
		require.Equal(t, "video", sources[i+1].Kind)
		require.Equal(t, "video", sources[i+1].Data)
		require.Equal(t, "video/mp4", sources[i+1].MediaType)
	}
	require.Nil(t, FromRequest(nil))
}
