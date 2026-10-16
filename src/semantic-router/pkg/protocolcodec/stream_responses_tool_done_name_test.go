package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesToolDoneOptionalNameAcrossClientFormats(t *testing.T) {
	for _, test := range []struct {
		name     string
		doneName json.RawMessage
	}{
		{name: "openai_omitted_name"},
		{name: "azure_empty_name", doneName: json.RawMessage(`""`)},
		{name: "nullable_name", doneName: json.RawMessage(`null`)},
		{name: "matching_name", doneName: json.RawMessage(`"lookup"`)},
	} {
		payload := responsesToolStreamWithDoneFields(t, func(fields map[string]json.RawMessage) {
			if test.doneName == nil {
				delete(fields, "name")
			} else {
				fields["name"] = test.doneName
			}
		})
		for _, target := range []llmprotocol.WireFormat{
			llmprotocol.OpenAIChatV1,
			llmprotocol.OpenAIResponsesV1,
			llmprotocol.AnthropicMessagesV1,
		} {
			t.Run(test.name+"/"+string(target), func(t *testing.T) {
				stream, err := NewBuiltinEngine().NewStream(
					llmprotocol.OpenAIResponsesV1, target,
					llmprotocol.StreamContext{
						Context: context.Background(), PublicModel: "public-model", ProviderModel: "source-model",
					},
				)
				if err != nil {
					t.Fatal(err)
				}
				frames, events, _, err := stream.Push(payload)
				if err != nil {
					t.Fatal(err)
				}
				finalFrames, finalEvents, _, err := stream.Finalize(nil)
				if err != nil {
					t.Fatal(err)
				}
				frames = append(frames, finalFrames...)
				events = append(events, finalEvents...)
				assertStreamToolCall(t, events)
				if !bytes.Contains(bytes.Join(frames, nil), []byte("lookup")) {
					t.Fatalf("%s client stream lost the name from output_item.added: %s", target, bytes.Join(frames, nil))
				}
			})
		}
	}
}

func TestResponsesToolDoneRejectsConflictingName(t *testing.T) {
	payload := responsesToolStreamWithDoneFields(t, func(fields map[string]json.RawMessage) {
		fields["name"] = json.RawMessage(`"different_tool"`)
	})
	decoder := newProviderStreamDecoder(llmprotocol.OpenAIResponsesV1)
	_, _, err := decoder.Push(payload)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_tool_identity_mismatch")
}

func TestResponsesToolDoneStillRequiresArguments(t *testing.T) {
	for _, test := range []struct {
		name      string
		arguments json.RawMessage
	}{
		{name: "omitted"},
		{name: "null", arguments: json.RawMessage(`null`)},
	} {
		t.Run(test.name, func(t *testing.T) {
			payload := responsesToolStreamWithDoneFields(t, func(fields map[string]json.RawMessage) {
				delete(fields, "name")
				if test.arguments == nil {
					delete(fields, "arguments")
				} else {
					fields["arguments"] = test.arguments
				}
			})
			decoder := newProviderStreamDecoder(llmprotocol.OpenAIResponsesV1)
			_, _, err := decoder.Push(payload)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_required_field")
		})
	}
}

func responsesToolStreamWithDoneFields(t *testing.T, edit func(map[string]json.RawMessage)) []byte {
	t.Helper()
	const header = "event: response.function_call_arguments.done\ndata: "
	frames := strings.Split(string(toolStreamFixture(llmprotocol.OpenAIResponsesV1)), "\n\n")
	found := false
	for index, frame := range frames {
		if !strings.HasPrefix(frame, header) {
			continue
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal([]byte(strings.TrimPrefix(frame, header)), &fields); err != nil {
			t.Fatal(err)
		}
		edit(fields)
		body, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		frames[index] = header + string(body)
		found = true
	}
	if !found {
		t.Fatal("tool stream fixture has no function-call done event")
	}
	return []byte(strings.Join(frames, "\n\n"))
}
