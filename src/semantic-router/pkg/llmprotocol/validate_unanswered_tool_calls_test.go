package llmprotocol

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func assistantCalls(position int, calls ...ToolCall) Message {
	message := Message{Role: RoleAssistant, SourcePosition: position}
	for index := range calls {
		message.Content = append(message.Content, Content{Kind: ContentToolCall, ToolCall: &calls[index]})
	}
	return message
}

func toolAnswer(callID, text string, isError bool) Message {
	return Message{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{
		CallID: callID, Content: []Content{{Kind: ContentText, Text: text}}, IsError: Bool(isError),
	}}}}
}

func userText(text string) Message {
	return Message{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: text}}}
}

func requireUnansweredToolCall(t *testing.T, err error, message, parameter string) {
	t.Helper()
	var protocolError *ProtocolError
	require.ErrorAs(t, err, &protocolError)
	require.Equal(t, ErrorInvalidRequest, protocolError.Category)
	require.Equal(t, "unanswered_tool_call", protocolError.Code)
	require.Equal(t, message, protocolError.Message)
	require.Equal(t, parameter, protocolError.Parameter)
}

var getWeather = ToolCall{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`}

// A call that the conversation moves on from without a result is refused as
// each client's API refuses it, naming the call.
func TestRefuseUnansweredToolCallsInTheClientsWords(t *testing.T) {
	for _, test := range []struct {
		name               string
		format             WireFormat
		message, parameter string
	}{
		{
			"chat completions", OpenAIChatV1,
			"An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. " +
				"The following tool_call_ids did not have response messages: call_1",
			"messages",
		},
		{"responses", OpenAIResponsesV1, "No tool output found for function call call_1.", "input"},
		{
			"anthropic messages", AnthropicMessagesV1,
			"messages.1: `tool_use` ids were found without `tool_result` blocks immediately after: call_1. " +
				"Each `tool_use` block must have a corresponding `tool_result` block in the next message.",
			"",
		},
		{"unknown client API", "", "tool calls have no later tool result: call_1", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := Request{Trusted: TrustedMetadata{SourceFormat: test.format}, Messages: []Message{
				userText("What is the weather in Paris?"), assistantCalls(2, getWeather), userText("And in Rome?"),
			}}
			requireUnansweredToolCall(t, RefuseUnansweredToolCalls(request), test.message, test.parameter)

			request.Messages = request.Messages[:2]
			requireUnansweredToolCall(t, RefuseUnansweredToolCalls(request), test.message, test.parameter)
		})
	}
}

func TestRefuseUnansweredToolCallsNamesACustomResponsesCall(t *testing.T) {
	request := Request{Trusted: TrustedMetadata{SourceFormat: OpenAIResponsesV1}, Messages: []Message{
		userText("Fix the typo."),
		assistantCalls(0, ToolCall{Kind: ToolKindCustom, ID: "call_patch", Name: "apply_patch", Arguments: "*** Begin Patch"}),
	}}
	requireUnansweredToolCall(t, RefuseUnansweredToolCalls(request), "No tool output found for custom tool call call_patch.", "input")
}

// Only the calls without a result are named, those of the first assistant
// message that has one, and a result before its call answers nothing.
func TestRefuseUnansweredToolCallsNamesTheFirstUnansweredTurn(t *testing.T) {
	paris, rome, lima := getWeather, ToolCall{ID: "call_2", Name: "get_weather", Arguments: `{"city":"Rome"}`},
		ToolCall{ID: "call_3", Name: "get_weather", Arguments: `{"city":"Lima"}`}
	request := Request{Trusted: TrustedMetadata{SourceFormat: OpenAIChatV1}, Messages: []Message{
		userText("Weather in Paris, Rome and Lima?"),
		assistantCalls(0, paris, rome, lima),
		toolAnswer("call_2", "sunny", false),
		userText("Go on."),
		assistantCalls(0, ToolCall{ID: "call_4", Name: "get_weather", Arguments: `{"city":"Oslo"}`}),
	}}
	requireUnansweredToolCall(t, RefuseUnansweredToolCalls(request),
		"An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. "+
			"The following tool_call_ids did not have response messages: call_1, call_3", "messages")

	request = Request{Trusted: TrustedMetadata{SourceFormat: OpenAIResponsesV1}, Messages: []Message{
		toolAnswer("call_1", "sunny", false), assistantCalls(0, paris),
	}}
	requireUnansweredToolCall(t, RefuseUnansweredToolCalls(request), "No tool output found for function call call_1.", "input")
}

// Agent clients answer an interrupted, denied or failed call with a result
// that says so, in the next message or later, so those histories pass.
func TestRefuseUnansweredToolCallsAcceptsEveryAnsweredCall(t *testing.T) {
	rome := ToolCall{ID: "call_2", Name: "get_weather", Arguments: `{"city":"Rome"}`}
	for name, messages := range map[string][]Message{
		"answered":                {userText("Paris?"), assistantCalls(2, getWeather), toolAnswer("call_1", "sunny", false)},
		"interrupted by the user": {userText("Paris?"), assistantCalls(2, getWeather), toolAnswer("call_1", "[Request interrupted by user for tool use]", true), userText("Stop.")},
		"aborted":                 {userText("Paris?"), assistantCalls(2, getWeather), toolAnswer("call_1", "aborted", false)},
		"parallel, answered out of order": {
			userText("Paris and Rome?"), assistantCalls(2, getWeather, rome),
			toolAnswer("call_2", "rainy", false), toolAnswer("call_1", "sunny", false),
		},
		"text beside the result": {
			userText("Paris?"), assistantCalls(2, getWeather), toolAnswer("call_1", "sunny", false),
			{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "<system-reminder>"}}, WithToolResults: true},
		},
		"no tool calls":                   {userText("Hello")},
		"a result whose call is retained": {toolAnswer("call_retained", "done", false)},
	} {
		t.Run(name, func(t *testing.T) {
			for _, format := range []WireFormat{OpenAIChatV1, OpenAIResponsesV1, AnthropicMessagesV1} {
				require.NoError(t, RefuseUnansweredToolCalls(Request{Trusted: TrustedMetadata{SourceFormat: format}, Messages: messages}))
			}
		})
	}
}
