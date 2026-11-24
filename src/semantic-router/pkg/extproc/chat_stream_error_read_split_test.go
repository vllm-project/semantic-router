package extproc

import (
	"fmt"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An upstream Chat stream ends in an error event, alone or after content (and
// a usage chunk) in the same read. [DONE] and the end of the stream then come
// in the same read or later ones. Each client should get the same bytes in
// every layout; a Chat client, or a Looper hop's caller, gets the public
// filter's output up to the event, then the Router's error frame.
func TestChatUpstreamErrorEventDoesNotDependOnReadSplit(t *testing.T) {
	event := `data: {"error":{"message":"mock provider stream failed","type":"vendor_overloaded","param":null,"code":"provider_overloaded"}}` + "\n\n"
	routerFrame := `data: {"error":{"type":"server_error","code":"provider_overloaded","message":"mock provider stream failed","param":null}}` + "\n\n"
	content := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","system_fingerprint":"fp1","choices":[{"index":0,"delta":{"content":"partial"},"logprobs":null,"finish_reason":null}]}` + "\n\n"
	// The public filter leaves out the server's fingerprint, and adds the null
	// usage for a client that asks for usage.
	publicContent := `data: {"choices":[{"index":0,"delta":{"content":"partial"},"logprobs":null,"finish_reason":null}],"created":1,"id":"c1","model":"provider-model","object":"chat.completion.chunk"}` + "\n\n"
	publicContentWithNullUsage := `data: {"choices":[{"index":0,"delta":{"content":"partial"},"logprobs":null,"finish_reason":null}],"created":1,"id":"c1","model":"provider-model","object":"chat.completion.chunk","usage":null}` + "\n\n"
	usage := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}` + "\n\n"
	done := "data: [DONE]\n\n"
	type caller struct {
		name   string
		client llmprotocol.WireFormat
		looper bool
	}
	callers := []caller{{name: "looper " + string(llmprotocol.OpenAIChatV1), client: llmprotocol.OpenAIChatV1, looper: true}}
	for _, client := range extProcMatrixFormats {
		callers = append(callers, caller{name: string(client), client: client})
	}
	for _, caller := range callers {
		for _, includeUsage := range []bool{true, false} {
			for _, c := range []struct{ name, prefix string }{
				{"error alone", ""},
				{"content in the error's read", content},
				{"content and usage in the error's read", content + usage},
			} {
				name, prefix := c.name, c.prefix
				t.Run(fmt.Sprintf("%s/include_usage=%v/%s", caller.name, includeUsage, name), func(t *testing.T) {
					body := func(reads ...string) string {
						return chatUpstreamClientBody(t, caller.client, caller.looper, includeUsage, reads)
					}
					sameRead := body(prefix + event + done)
					if caller.client == llmprotocol.OpenAIChatV1 {
						want := strings.ReplaceAll(prefix, content, publicContentWithNullUsage)
						if !includeUsage {
							want = strings.ReplaceAll(strings.ReplaceAll(prefix, content, publicContent), usage, "")
						}
						require.Equal(t, want+routerFrame, sameRead)
					}
					unterminated := prefix + strings.TrimSuffix(event, "\n\n")
					for _, layout := range []struct {
						name  string
						reads []string
					}{
						{"[DONE] in a later read", []string{prefix + event, done}},
						{"[DONE] then an empty final read", []string{prefix + event, done, ""}},
						{"no [DONE], empty final read", []string{prefix + event, ""}},
						{"no [DONE], end of stream in the error's read", []string{prefix + event}},
						{"error ends at end of stream", []string{unterminated}},
						{"error ends at end of stream after an empty read", []string{unterminated, ""}},
					} {
						t.Run(layout.name, func(t *testing.T) {
							require.Equal(t, sameRead, body(layout.reads...))
						})
					}
				})
			}
		}
	}
}

// An error event the codec cannot decode keeps the codec's generic failure
// frame, in every layout, as on main.
func TestChatUndecodableUpstreamErrorKeepsTheGenericFrame(t *testing.T) {
	bad := `data: {"error":7}` + "\n\n"
	generic := `data: {"error":{"type":"server_error","code":"invalid_upstream_json","message":"upstream response JSON contains a non-canonical field","param":null}}` + "\n\n"
	for _, reads := range [][]string{{bad + "data: [DONE]\n\n"}, {bad, "data: [DONE]\n\n"}, {bad, ""}} {
		require.Equal(t, generic, chatUpstreamClientBody(t, llmprotocol.OpenAIChatV1, false, true, reads))
	}
}

// A chunk the codec rejects fails the stream before the provider's error event
// in the same read. The client gets only the Router's failure, as on main, and
// never the rejected chunk.
func TestChatUpstreamErrorAfterARejectedChunkKeepsTheRouterFailure(t *testing.T) {
	bogus := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"bogus"}]}` + "\n\n"
	event := `data: {"error":{"message":"mock provider stream failed","type":"vendor_overloaded","param":null,"code":"provider_overloaded"}}` + "\n\n"
	failure := `data: {"error":{"type":"server_error","code":"invalid_chat_finish_reason","message":"Chat finish reason is not recognized","param":null}}` + "\n\n"
	for _, reads := range [][]string{{bogus + event}, {bogus + event + "data: [DONE]\n\n"}, {bogus, event}} {
		require.Equal(t, failure, chatUpstreamClientBody(t, llmprotocol.OpenAIChatV1, false, true, reads))
	}
}

// The new branch is for a Chat client of a Chat upstream. A same-format
// stream of another protocol, here Messages to Messages, keeps the content it
// already sent ahead of the provider's error.
func TestSameFormatMessagesStreamErrorKeepsItsContent(t *testing.T) {
	read := strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
	}, "")
	router := &OpenAIRouter{}
	ctx := looperTransportContext(llmprotocol.AnthropicMessagesV1, true)
	ctx.LooperRequest = false
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	_, err := router.handleResponseHeaders(looperTransportHeaders(200, "text/event-stream"), ctx)
	require.NoError(t, err)
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(read), EndOfStream: true},
	}, ctx)
	require.NoError(t, err)
	body := read
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		body = string(mutation.GetBody())
	}
	require.Contains(t, body, `"text":"partial"`)
	require.Contains(t, body, "Overloaded")
}

// The codec fails on an error value it cannot decode, ahead of an unterminated
// content frame, and the stream then ends in a later empty read. The client
// gets the same bytes as when it all comes in one read.
func TestChatRouterFailureInAnEarlierReadKeepsItsFrames(t *testing.T) {
	read := `data: {"error":{}}` + "\n\n" + `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[{"index":0,"delta":{"content":"late"},"finish_reason":null}]}`
	oneRead := chatUpstreamClientBody(t, llmprotocol.OpenAIChatV1, false, true, []string{read})
	require.Contains(t, oneRead, "late")
	require.Equal(t, oneRead, chatUpstreamClientBody(t, llmprotocol.OpenAIChatV1, false, true, []string{read, ""}))
}

// An error event that also carries usage, for a client that did not ask for
// usage, still reaches the client as the Router's error frame when [DONE]
// comes later.
func TestChatUpstreamErrorEventWithUsageIsNotDroppedAsUsage(t *testing.T) {
	event := `data: {"error":{"message":"mock provider stream failed","type":"vendor_overloaded","param":null,"code":"provider_overloaded"},"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}` + "\n\n"
	want := `data: {"error":{"type":"server_error","code":"provider_overloaded","message":"mock provider stream failed","param":null}}` + "\n\n"
	require.Equal(t, want, chatUpstreamClientBody(t, llmprotocol.OpenAIChatV1, false, false, []string{event, "data: [DONE]\n\n"}))
}

// An unset client format still needs a rewrite (requiresClientResponseRewrite),
// so the client gets the codec's frames, never the provider's bytes, even when
// the error event shares a read with content.
func TestChatUpstreamErrorEventKeepsTheRewriteForAnUnsetClientFormat(t *testing.T) {
	event := `data: {"error":{"message":"mock provider stream failed","type":"vendor_overloaded","param":null,"code":"provider_overloaded"}}` + "\n\n"
	content := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","system_fingerprint":"fp1","choices":[{"index":0,"delta":{"content":"partial"},"logprobs":null,"finish_reason":null}]}` + "\n\n"
	body := chatUpstreamClientBody(t, "", false, true, []string{content + event + "data: [DONE]\n\n"})
	// The codec's frames name the model the provider reports and drop its
	// system_fingerprint, which the provider's bytes carry.
	require.Contains(t, body, `"model":"provider-model"`)
	require.NotContains(t, body, "fp1")
}

// chatUpstreamClientBody runs a Chat upstream's reads through the response
// header and body handlers, in that order, and returns what the client, or the
// Looper hop's caller, gets.
func chatUpstreamClientBody(t *testing.T, client llmprotocol.WireFormat, looper, includeUsage bool, reads []string) string {
	t.Helper()
	router := &OpenAIRouter{}
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, true)
	ctx.LooperRequest = looper
	ctx.SemanticRequest.StreamOptions.IncludeUsage = &includeUsage
	ctx.SourceFormat = client
	_, err := router.handleResponseHeaders(looperTransportHeaders(200, "text/event-stream"), ctx)
	require.NoError(t, err)
	var out strings.Builder
	for i, read := range reads {
		response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(read), EndOfStream: i == len(reads)-1},
		}, ctx)
		require.NoError(t, err)
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			read = string(mutation.GetBody())
		}
		out.WriteString(read)
	}
	return out.String()
}

// A rejected tool delta must keep the accepted prefix regardless of how
// Envoy groups the SSE frames into body reads.
func TestChatToolKindSwitchPreservesAcceptedPrefix(t *testing.T) {
	chunk := func(delta, finish string) string {
		return `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[{"index":0,"delta":` +
			delta + `,"finish_reason":` + finish + `}]}` + "\n\n"
	}
	role := chunk(`{"role":"assistant"}`, "null")
	valid := chunk(`{"tool_calls":[{"index":0,"id":"call_mock_custom_kind","type":"custom","custom":{"name":"apply_patch","input":"abc"}}]}`, "null")
	invalid := chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]}`, "null")
	finish := chunk("{}", `"tool_calls"`)
	done := "data: [DONE]\n\n"
	partial := strings.TrimSuffix(invalid, "\n\n")
	for _, layout := range []struct {
		name  string
		reads []string
	}{
		{"separate reads", []string{role, valid, invalid, finish, done}},
		{"one read", []string{role + valid + invalid + finish + done}},
		{"mismatch and finish", []string{role, valid, invalid + finish, done}},
		{"mismatch and done", []string{role, valid, invalid + done}},
		{"mismatch and both terminal frames", []string{role, valid, invalid + finish + done}},
		{"mismatch in final read", []string{role, valid, invalid}},
		{"mismatch in unterminated final frame", []string{role, valid, partial}},
		{"tool delta before unterminated frame in final read", []string{role, valid + partial}},
		{"tool delta before unterminated frame and empty read", []string{role, valid + partial, ""}},
		{"mismatch finalized by empty read", []string{role, valid, partial, ""}},
	} {
		t.Run(layout.name, func(t *testing.T) {
			body := chatUpstreamClientBody(t, llmprotocol.OpenAIChatV1, false, false, layout.reads)
			frames := strings.Split(strings.TrimSuffix(body, "\n\n"), "\n\n")
			require.Len(t, frames, 3, "%s", body)
			require.Contains(t, frames[0], `"role":"assistant"`)
			require.Contains(t, frames[1], `"id":"call_mock_custom_kind"`)
			require.Contains(t, frames[1], `"type":"custom"`)
			require.Contains(t, frames[1], `"name":"apply_patch"`)
			require.Contains(t, frames[1], `"input":"abc"`)
			require.NotContains(t, frames[1], `"function":`)
			require.Contains(t, frames[2], `"code":"stream_tool_identity_mismatch"`)
		})
	}
}
