package anthropic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildPassthrough_EmptyBodyReturnsNil(t *testing.T) {
	pt, err := BuildPassthroughFromAnthropicBody(nil)
	require.NoError(t, err)
	assert.Nil(t, pt)

	pt, err = BuildPassthroughFromAnthropicBody([]byte{})
	require.NoError(t, err)
	assert.Nil(t, pt)
}

func TestBuildPassthrough_OpenAIShapeBodyReturnsEmptyCarrier(t *testing.T) {
	// An OpenAI-shape body has no top_k / metadata.user_id / system-array /
	// image blocks / tool_result.is_error. The lenient parser should return
	// an empty carrier without error.
	raw := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.NotNil(t, pt)
	assert.Nil(t, pt.TopK)
	assert.Empty(t, pt.MetadataUserID)
	assert.Empty(t, pt.SystemBlocks)
	assert.Empty(t, pt.CacheControl)
	assert.Empty(t, pt.ToolResultErrors)
	assert.Empty(t, pt.ToolResultArrayContent)
	assert.Empty(t, pt.UserMessageImageBlocks)
}

func TestBuildPassthrough_MalformedJSONIsLenient(t *testing.T) {
	raw := []byte(`{"top_k":`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.NotNil(t, pt)
	assert.Nil(t, pt.TopK)
}

func TestBuildPassthrough_TopK(t *testing.T) {
	raw := []byte(`{"top_k":40}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.NotNil(t, pt.TopK)
	assert.Equal(t, int64(40), *pt.TopK)
}

func TestBuildPassthrough_TopKMissing(t *testing.T) {
	raw := []byte(`{}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	assert.Nil(t, pt.TopK)
}

func TestBuildPassthrough_MetadataUserID(t *testing.T) {
	raw := []byte(`{"metadata":{"user_id":"alice"}}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	assert.Equal(t, "alice", pt.MetadataUserID)
}

func TestBuildPassthrough_SystemArrayWithCacheControl(t *testing.T) {
	raw := []byte(`{
		"system": [
			{"type":"text","text":"You are helpful.","cache_control":{"type":"ephemeral","ttl":"5m"}},
			{"type":"text","text":"Be concise."}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Len(t, pt.SystemBlocks, 2)
	assert.Equal(t, "You are helpful.", pt.SystemBlocks[0].Text)
	require.NotNil(t, pt.SystemBlocks[0].CacheControl)
	assert.Equal(t, "ephemeral", pt.SystemBlocks[0].CacheControl.Type)
	assert.Equal(t, "5m", pt.SystemBlocks[0].CacheControl.TTL)
	assert.Nil(t, pt.SystemBlocks[1].CacheControl)

	require.Contains(t, pt.CacheControl, "system[0]")
	assert.Equal(t, "5m", pt.CacheControl["system[0]"].TTL)
	assert.NotContains(t, pt.CacheControl, "system[1]")
}

func TestBuildPassthrough_PerMessageCacheControl(t *testing.T) {
	raw := []byte(`{
		"messages": [
			{"role":"user","content":[
				{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Contains(t, pt.CacheControl, "messages[0].content[0]")
	assert.Equal(t, "ephemeral", pt.CacheControl["messages[0].content[0]"].Type)
	assert.Empty(t, pt.CacheControl["messages[0].content[0]"].TTL)
}

func TestBuildPassthrough_ToolsCacheControl(t *testing.T) {
	raw := []byte(`{
		"tools": [
			{"name":"get_weather","cache_control":{"type":"ephemeral","ttl":"1h"}},
			{"name":"get_time"}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Contains(t, pt.CacheControl, "tools[0]")
	assert.Equal(t, "1h", pt.CacheControl["tools[0]"].TTL)
	assert.NotContains(t, pt.CacheControl, "tools[1]")
}

func TestBuildPassthrough_ImageBlocksBase64(t *testing.T) {
	raw := []byte(`{
		"messages": [
			{"role":"user","content":[
				{"type":"text","text":"what is this"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Contains(t, pt.UserMessageImageBlocks, 0)
	images := pt.UserMessageImageBlocks[0]
	require.Len(t, images, 1)
	assert.Equal(t, "base64", images[0].Source.Type)
	assert.Equal(t, "image/png", images[0].Source.MediaType)
	assert.Equal(t, "AAAA", images[0].Source.Data)
}

func TestBuildPassthrough_ImageBlocksURL(t *testing.T) {
	raw := []byte(`{
		"messages": [
			{"role":"user","content":[
				{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Contains(t, pt.UserMessageImageBlocks, 0)
	images := pt.UserMessageImageBlocks[0]
	require.Len(t, images, 1)
	assert.Equal(t, "url", images[0].Source.Type)
	assert.Equal(t, "https://example.com/cat.png", images[0].Source.URL)
}

func TestBuildPassthrough_ImageBlockOnlyOnUserMessage(t *testing.T) {
	// Image content blocks on assistant messages are not captured: the
	// translation only re-attaches images to user messages in this PR.
	raw := []byte(`{
		"messages": [
			{"role":"assistant","content":[
				{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	assert.Empty(t, pt.UserMessageImageBlocks)
}

func TestBuildPassthrough_UserMessageIndexCountsUserMessagesOnly(t *testing.T) {
	// Index 0 is the first user message, even when assistant/system messages
	// precede it.
	raw := []byte(`{
		"messages": [
			{"role":"assistant","content":[{"type":"text","text":"hi"}]},
			{"role":"user","content":[
				{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}
			]},
			{"role":"assistant","content":[{"type":"text","text":"ok"}]},
			{"role":"user","content":[
				{"type":"image","source":{"type":"url","url":"https://example.com/b.png"}}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Contains(t, pt.UserMessageImageBlocks, 0)
	require.Contains(t, pt.UserMessageImageBlocks, 1)
	assert.Equal(t, "https://example.com/a.png", pt.UserMessageImageBlocks[0][0].Source.URL)
	assert.Equal(t, "https://example.com/b.png", pt.UserMessageImageBlocks[1][0].Source.URL)
}

func TestBuildPassthrough_ToolResultIsError(t *testing.T) {
	raw := []byte(`{
		"messages": [
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"call_abc","is_error":true,"content":"oops"}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Contains(t, pt.ToolResultErrors, "call_abc")
	assert.True(t, pt.ToolResultErrors["call_abc"])
}

func TestBuildPassthrough_ToolResultIsErrorFalseNotCaptured(t *testing.T) {
	// is_error: false is the SDK default; not capturing it keeps the carrier
	// minimal and back-compat: replay only emits the flag when explicitly true.
	raw := []byte(`{
		"messages": [
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"call_abc","is_error":false,"content":"ok"}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	assert.Empty(t, pt.ToolResultErrors)
}

func TestBuildPassthrough_ToolResultArrayContent(t *testing.T) {
	raw := []byte(`{
		"messages": [
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"call_abc","content":[
					{"type":"text","text":"see image"},
					{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"BBBB"}}
				]}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	require.Contains(t, pt.ToolResultArrayContent, "call_abc")
	blocks := pt.ToolResultArrayContent["call_abc"]
	require.Len(t, blocks, 2)
	assert.Equal(t, "text", blocks[0].Type)
	assert.Equal(t, "see image", blocks[0].Text)
	assert.Equal(t, "image", blocks[1].Type)
	require.NotNil(t, blocks[1].Source)
	assert.Equal(t, "image/jpeg", blocks[1].Source.MediaType)
}

func TestBuildPassthrough_ToolResultMissingToolUseIDSkipped(t *testing.T) {
	raw := []byte(`{
		"messages": [
			{"role":"user","content":[
				{"type":"tool_result","is_error":true,"content":"oops"}
			]}
		]
	}`)
	pt, err := BuildPassthroughFromAnthropicBody(raw)
	require.NoError(t, err)
	assert.Empty(t, pt.ToolResultErrors)
	assert.Empty(t, pt.ToolResultArrayContent)
}

func TestSetHeadersFromIncoming_NilSafeAndCaseInsensitive(t *testing.T) {
	var pt *AnthropicPassthrough
	assert.NotPanics(t, func() { pt.SetHeadersFromIncoming(map[string]string{"x": "y"}) })

	pt = &AnthropicPassthrough{}
	pt.SetHeadersFromIncoming(map[string]string{
		"Anthropic-Version": "2024-10-22",
		"ANTHROPIC-BETA":    "prompt-caching-2024-07-31",
		"unrelated":         "ignored",
	})
	assert.Equal(t, "2024-10-22", pt.AnthropicVersion)
	assert.Equal(t, "prompt-caching-2024-07-31", pt.AnthropicBeta)
}

func TestSetHeadersFromIncoming_SkipsEmptyValues(t *testing.T) {
	pt := &AnthropicPassthrough{AnthropicVersion: "existing"}
	pt.SetHeadersFromIncoming(map[string]string{"anthropic-version": ""})
	assert.Equal(t, "existing", pt.AnthropicVersion)
}
