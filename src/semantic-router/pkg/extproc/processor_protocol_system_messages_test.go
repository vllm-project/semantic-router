//go:build !windows

package extproc

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// systemMessagesRouter serves every request from one vLLM Chat model. accepted
// lists messages[].role in allow_lossy_translation, and noDeveloperRole
// declares that the model lacks the developer role.
func systemMessagesRouter(t *testing.T, accepted, noDeveloperRole bool) *OpenAIRouter {
	t.Helper()
	lossy, unsupported := "[]", ""
	if accepted {
		lossy = `["messages[].role"]`
	}
	if noDeveloperRole {
		unsupported = "\n      unsupported_capabilities: [developer_role]"
	}
	return dispatchPluginsRouter(t, fmt.Sprintf(`
version: v0.5
providers:
  defaults:
    model: chat-model
  models:
    - name: chat-model
      provider_model_id: chat-model
      api_format: chat_completions%s
      backend_refs:
        - name: chat-backend
          provider: vllm
          endpoint: http://127.0.0.1:8000/v1
global:
  services:
    response_api: {enabled: true, store_backend: memory}
  router:
    allow_lossy_translation: %s
recipes:
  - name: default
    decisions:
      - name: chat
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: chat-model
`, unsupported, lossy))
}

// sendSystemMessagesRequest runs body through the request phases and returns
// each message the Chat backend receives, as its role and its text parts, and
// the answer's x-vsr-warnings.
func sendSystemMessagesRequest(t *testing.T, router *OpenAIRouter, path, body string) ([]string, string) {
	t.Helper()
	response, ctx := sendThroughPlugins(t, router, path, body)
	var sent []string
	for _, message := range forwardedJSON(t, response)["messages"].([]any) {
		entry := message.(map[string]any)
		var parts []string
		switch content := entry["content"].(type) {
		case string:
			parts = append(parts, content)
		case []any:
			for _, part := range content {
				parts = append(parts, part.(map[string]any)["text"].(string))
			}
		}
		sent = append(sent, entry["role"].(string)+": "+strings.Join(parts, " + "))
	}
	contentType := "application/json"
	if ctx.ExpectStreamingResponse {
		contentType = "text/event-stream"
	}
	headers, err := router.handleResponseHeaders(samplingResponseHeaders(contentType), ctx)
	require.NoError(t, err)
	return sent, protocolDiagnosticsHeader(headers.GetResponseHeaders().GetResponse().GetHeaderMutation())
}

const geographyTutorMessages = `{"role":"system","content":"You are a helpful geography tutor."},` +
	`{"role":"system","content":"Answer every question in one word."},` +
	`{"role":"user","content":"What is the capital of Spain?"}`

// The geography tutor gives its two instructions as two system messages.
// Qwen3.5 and later chat templates answer the second with 400 "System message
// must be at the beginning.", so once the operator lists messages[].role the
// Chat backend receives one system message, first, with both texts, buffered
// or streamed. Without the opt-in it receives them as the client sent them.
func TestChatSystemMessagesReachAChatBackendAsOneSystemMessage(t *testing.T) {
	const warning = `protocol;action=approximated;reason="2 system and developer messages sent as one system message";field="messages[].role"`
	for _, stream := range []bool{false, true} {
		for _, accepted := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%t/opt_in=%t", stream, accepted), func(t *testing.T) {
				sent, warnings := sendSystemMessagesRequest(t, systemMessagesRouter(t, accepted, false), "/v1/chat/completions",
					fmt.Sprintf(`{"model":"chat-model","stream":%t,"messages":[%s]}`, stream, geographyTutorMessages))
				want := []string{
					"system: You are a helpful geography tutor.\n\nAnswer every question in one word.",
					"user: What is the capital of Spain?",
				}
				if !accepted {
					want = []string{
						"system: You are a helpful geography tutor.", "system: Answer every question in one word.",
						"user: What is the capital of Spain?",
					}
				}
				require.Equal(t, want, sent)
				require.Equal(t, accepted, strings.Contains(warnings, warning), "x-vsr-warnings = %q", warnings)
				require.Equal(t, accepted, strings.Contains(warnings, `field="messages[].role"`), "x-vsr-warnings = %q", warnings)
			})
		}
	}
}

// A system or developer message inside the conversation reaches the Chat
// backend as user text where the client put it: after the assistant's answer
// as a user message of its own, and after a user message as part of it.
// Without the opt-in every one is still sent in its own role.
func TestChatSystemMessagesInsideTheConversationBecomeUserTextInPlace(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, accepted := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%t/opt_in=%t", stream, accepted), func(t *testing.T) {
				sent, warnings := sendSystemMessagesRequest(t, systemMessagesRouter(t, accepted, false), "/v1/chat/completions", fmt.Sprintf(
					`{"model":"chat-model","stream":%t,"messages":[`+
						`{"role":"system","content":"You are a helpful geography tutor."},`+
						`{"role":"user","content":"What is the capital of Spain?"},`+
						`{"role":"assistant","content":"Madrid"},`+
						`{"role":"system","content":"Answer in French from now on."},`+
						`{"role":"user","content":"And of Italy?"},`+
						`{"role":"developer","content":"Answer in one word."}]}`, stream))
				if !accepted {
					require.Equal(t, []string{
						"system: You are a helpful geography tutor.", "user: What is the capital of Spain?", "assistant: Madrid",
						"system: Answer in French from now on.", "user: And of Italy?", "developer: Answer in one word.",
					}, sent)
					require.NotContains(t, warnings, `field="messages[].role"`)
					return
				}
				require.Equal(t, []string{
					"system: You are a helpful geography tutor.",
					"user: What is the capital of Spain?",
					"assistant: Madrid",
					"user: Answer in French from now on.",
					"user: And of Italy? + Answer in one word.",
				}, sent)
				require.Contains(t, warnings, `protocol;action=approximated;reason="2 system messages sent as user text in place";field="messages[].role"`)
			})
		}
	}
}

// One system message suits every template: the Chat backend receives the
// request as the client sent it, and nothing is reported.
func TestChatSingleSystemMessageIsSentUnchanged(t *testing.T) {
	for _, stream := range []bool{false, true} {
		sent, warnings := sendSystemMessagesRequest(t, systemMessagesRouter(t, true, false), "/v1/chat/completions", fmt.Sprintf(
			`{"model":"chat-model","stream":%t,"messages":[{"role":"system","content":"You are a helpful geography tutor."},`+
				`{"role":"user","content":"What is the capital of Spain?"}]}`, stream))
		require.Equal(t, []string{"system: You are a helpful geography tutor.", "user: What is the capital of Spain?"}, sent)
		require.NotContains(t, warnings, `field="messages[].role"`)
	}
}

// A Responses request's instructions and its developer items reach a Chat
// backend as one system message.
func TestResponsesInstructionsReachAChatBackendAsOneSystemMessage(t *testing.T) {
	sent, warnings := sendSystemMessagesRequest(t, systemMessagesRouter(t, true, false), "/v1/responses",
		`{"model":"chat-model","store":false,"instructions":"You are a helpful geography tutor.","input":[`+
			`{"role":"developer","content":"Answer every question in one word."},`+
			`{"role":"user","content":"What is the capital of Spain?"}]}`)
	require.Equal(t, []string{
		"system: You are a helpful geography tutor.\n\nAnswer every question in one word.",
		"user: What is the capital of Spain?",
	}, sent)
	require.Contains(t, warnings, `reason="2 system and developer messages sent as one system message";field="messages[].role"`)
}

// A model that lacks the developer role receives the one merged system
// message, and the answer reports one messages[].role entry, not one per
// projection. Without the opt-in its developer message still arrives in the
// system role.
func TestSystemMessagesComposeWithAModelThatLacksTheDeveloperRole(t *testing.T) {
	body := `{"model":"chat-model","messages":[{"role":"developer","content":"The secret word is PAPAYA."},` +
		`{"role":"system","content":"Answer in one word."},{"role":"user","content":"What is the secret word?"}]}`
	sent, warnings := sendSystemMessagesRequest(t, systemMessagesRouter(t, true, true), "/v1/chat/completions", body)
	require.Equal(t, []string{"system: The secret word is PAPAYA.\n\nAnswer in one word.", "user: What is the secret word?"}, sent)
	require.Equal(t, 1, strings.Count(warnings, `field="messages[].role"`), "x-vsr-warnings = %q", warnings)
	require.Contains(t, warnings, `reason="2 system and developer messages sent as one system message"`)

	sent, warnings = sendSystemMessagesRequest(t, systemMessagesRouter(t, false, true), "/v1/chat/completions", body)
	require.Equal(t, []string{"system: The secret word is PAPAYA.", "system: Answer in one word.", "user: What is the secret word?"}, sent)
	require.Contains(t, warnings, llmprotocol.DeveloperRoleSentAsSystem)
}

func TestEmptyInstructionInToolSpanProducesNoProjectionWarning(t *testing.T) {
	router := systemMessagesRouter(t, true, false)
	messages, warnings := sendSystemMessagesRequest(t, router, "/v1/chat/completions", `{
		"model":"chat-model","messages":[
			{"role":"user","content":"lookup"},
			{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
			{"role":"system","content":""},
			{"role":"tool","tool_call_id":"a","content":"found"}
		]}`)
	require.Equal(t, []string{"user: lookup", "assistant: ", "tool: found"}, messages)
	require.Empty(t, warnings)
}
