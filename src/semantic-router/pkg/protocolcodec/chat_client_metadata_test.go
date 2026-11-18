package protocolcodec

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const clientReplayTools = `"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]`

// Turn 2 of a tool loop as litellm 1.82.6 sends it when the client replays
// message.model_dump(), as mini-SWE-agent 2.4.6 does: the reply members that
// litellm's Message type has no attribute for, vLLM's reasoning and refusal,
// ride in provider_specific_fields, and the reasoning is also copied to
// reasoning_content.
const litellmReasoningReplay = `{"model":"client-model","messages":[
  {"role":"user","content":"List the files."},
  {"role":"assistant","tool_calls":[{"function":{"arguments":"{\"command\": \"ls\"}","name":"bash"},"id":"call_1","type":"function"}],
   "reasoning_content":"The user wants the files listed.",
   "provider_specific_fields":{"reasoning":"The user wants the files listed.","refusal":null,"reasoning_content":"The user wants the files listed."}},
  {"role":"tool","tool_call_id":"call_1","content":"README.md"}],` + clientReplayTools + `}`

// The same replay of a reply whose reasoning and refusal were both null.
const litellmNullFieldsReplay = `{"model":"client-model","messages":[
  {"role":"user","content":"List the files."},
  {"role":"assistant","tool_calls":[{"function":{"arguments":"{\"command\": \"ls\"}","name":"bash"},"id":"call_1","type":"function"}],
   "provider_specific_fields":{"reasoning":null,"refusal":null}},
  {"role":"tool","tool_call_id":"call_1","content":"README.md"}],` + clientReplayTools + `}`

// The assistant message the openai SDK's stream helper returns
// (chat.completions.stream(...).get_final_completion()), replayed as the
// object or its model_dump(): parsed and parsed_arguments are the SDK's own
// parse results.
const openAISDKParsedReplay = `{"model":"client-model","messages":[
  {"role":"user","content":"List the files."},
  {"content":"","refusal":null,"role":"assistant","annotations":null,"audio":null,"function_call":null,
   "tool_calls":[{"id":"call_1","function":{"arguments":"{\"command\": \"ls\"}","name":"bash","parsed_arguments":null},"type":"function","index":0}],
   "parsed":null,"reasoning":"The user wants the files listed."},
  {"role":"tool","tool_call_id":"call_1","content":"README.md"}],` + clientReplayTools + `}`

var openAISDKParsedValues = strings.NewReplacer(
	`"parsed_arguments":null`, `"parsed_arguments":{"command":"ls"}`,
	`"parsed":null`, `"parsed":{"files":1}`,
).Replace(openAISDKParsedReplay)

var clientReplays = map[string]struct {
	body      string
	reasoning bool
	dropped   []string
}{
	"litellm reasoning":        {litellmReasoningReplay, true, []string{"messages[].provider_specific_fields"}},
	"litellm null fields":      {litellmNullFieldsReplay, false, []string{"messages[].provider_specific_fields"}},
	"litellm null container":   {strings.Replace(litellmNullFieldsReplay, `{"reasoning":null,"refusal":null}`, "null", 1), false, nil},
	"openai sdk null parse":    {openAISDKParsedReplay, true, nil},
	"openai sdk parse results": {openAISDKParsedValues, true, []string{"messages[].parsed", "messages[].tool_calls[].function.parsed_arguments"}},
}

func TestOpenAIChatRequestDropsClientReplayMetadata(t *testing.T) {
	for name, replay := range clientReplays {
		t.Run(name, func(t *testing.T) {
			request, envelope, diagnostics, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(replay.body))
			if err != nil {
				t.Fatalf("DecodeRequestForMutation() error = %v", err)
			}
			if len(replay.dropped) == 0 && len(diagnostics) > 0 {
				t.Fatalf("diagnostics = %+v, want none for null members", diagnostics)
			}
			if len(replay.dropped) > 0 {
				assertDiagnosticFields(t, diagnostics, replay.dropped...)
			}
			for _, diagnostic := range diagnostics {
				if diagnostic.Source != llmprotocol.OpenAIChatV1 || diagnostic.Action != llmprotocol.DiagnosticDropped ||
					diagnostic.Reason != "client metadata is not model input" {
					t.Fatalf("diagnostic = %+v", diagnostic)
				}
			}
			if envelope.Request != nil {
				t.Fatal("the source bytes still carry the client metadata and must not be replayed")
			}
			assertClientReplayConversation(t, request, replay.reasoning)
		})
	}
}

func assertClientReplayConversation(t *testing.T, request llmprotocol.Request, reasoning bool) {
	t.Helper()
	if len(request.Messages) != 3 {
		t.Fatalf("messages = %+v", request.Messages)
	}
	var call *llmprotocol.ToolCall
	var reasoningText string
	for _, content := range request.Messages[1].Content {
		switch content.Kind {
		case llmprotocol.ContentToolCall:
			call = content.ToolCall
		case llmprotocol.ContentReasoning:
			reasoningText = content.Text
		}
	}
	if call == nil || call.ID != "call_1" || call.Name != "bash" || call.Arguments != `{"command": "ls"}` {
		t.Fatalf("assistant tool call = %+v", request.Messages[1].Content)
	}
	if reasoning != (reasoningText == "The user wants the files listed.") {
		t.Fatalf("assistant reasoning = %q, want it kept: %v", reasoningText, reasoning)
	}
	result := request.Messages[2].Content[0].ToolResult
	if result == nil || result.CallID != "call_1" {
		t.Fatalf("tool result = %+v", request.Messages[2])
	}
}

// A request a mutation does not touch is forwarded as the client's bytes, so
// the dropped members would reach the backend if the envelope kept them.
func TestOpenAIChatClientReplayMetadataNeverReachesTheBackend(t *testing.T) {
	for name, replay := range clientReplays {
		t.Run(name, func(t *testing.T) {
			translated, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(replay.body), nil)
			if err != nil {
				t.Fatalf("TranslateRequest() error = %v", err)
			}
			body := string(translated.Body)
			for _, member := range []string{`"provider_specific_fields"`, `"parsed"`, `"parsed_arguments"`} {
				if strings.Contains(body, member) {
					t.Fatalf("backend request carries %s: %s", member, body)
				}
			}
			if !strings.Contains(body, `"tool_call_id":"call_1"`) || !strings.Contains(body, `"name":"bash"`) {
				t.Fatalf("backend request lost the tool loop: %s", body)
			}
		})
	}
}

// Members that carry meaning for the backend, or that no client library
// adds, are still refused.
func TestOpenAIChatRequestStillRefusesOtherReplayFields(t *testing.T) {
	assistant := func(members string) string {
		return `{"model":"client-model","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok",` + members + `}]}`
	}
	toolCall := func(members string) string {
		return `{"model":"client-model","messages":[{"role":"user","content":"hi"},{"role":"assistant","tool_calls":[` +
			`{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{}"}` + members + `}]}]}`
	}
	for name, test := range map[string]struct{ body, code string }{
		"signed thinking blocks":    {assistant(`"thinking_blocks":[{"type":"thinking","thinking":"x","signature":"sig"}]`), "invalid_json"},
		"output audio":              {assistant(`"audio":{"id":"audio_1","data":"","expires_at":1,"transcript":"ok"}`), "unsupported_output_audio"},
		"legacy function call":      {assistant(`"function_call":{"name":"bash","arguments":"{}"}`), "unsupported_legacy_function_call"},
		"an unknown member":         {assistant(`"x_client":1`), "invalid_json"},
		"another spelling":          {assistant(`"Parsed":null`), "invalid_json"},
		"a duplicate member":        {assistant(`"parsed":null,"content":"again"`), "duplicate_json_field"},
		"parsed outside messages":   {`{"model":"client-model","parsed":null,"messages":[{"role":"user","content":"hi"}]}`, "invalid_json"},
		"parsed_arguments on calls": {toolCall(`,"parsed_arguments":null`), "invalid_json"},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(test.body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != test.code {
				t.Fatalf("DecodeRequestForMutation() error = %v, want %s", err, test.code)
			}
		})
	}
}

// The openai SDK's stream helper returns reasoning items with a status. A
// client that sends response.output back as input replays it, as it does the
// status of every other item.
func TestOpenAIResponsesRequestAcceptsReasoningItemStatus(t *testing.T) {
	for _, status := range []string{"completed", "incomplete", "in_progress"} {
		t.Run(status, func(t *testing.T) {
			body := `{"model":"client-model","input":[
			  {"role":"user","content":"List the files."},
			  {"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"The user wants the files listed."}],"status":"` + status + `"},
			  {"id":"fc_1","type":"function_call","call_id":"call_1","name":"bash","arguments":"{\"command\":\"ls\"}","status":"completed"},
			  {"type":"function_call_output","call_id":"call_1","output":"README.md"}]}`
			request, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
			if err != nil {
				t.Fatalf("DecodeRequestForMutation() error = %v", err)
			}
			if len(request.Messages) != 4 || len(request.Messages[1].Content) != 1 ||
				request.Messages[1].Content[0].Kind != llmprotocol.ContentReasoning ||
				request.Messages[1].Content[0].Text != "The user wants the files listed." {
				t.Fatalf("messages = %+v", request.Messages)
			}
		})
	}
}
