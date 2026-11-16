package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A store:false client that asks for provider-encrypted reasoning keeps the
// item the Router returns and sends it back on the next turn. Accepting the
// include and then rejecting the item leaves that client with no continuation.
func TestResponsesEncryptedReasoningReplayIsAccepted(t *testing.T) {
	engine := NewBuiltinEngine()

	request := []byte(`{"model":"m","input":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAA"}]}`)
	decoded, _, diagnostics, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, request)
	if err != nil {
		t.Fatalf("replaying a reasoning item that carries encrypted_content: %v", err)
	}
	if len(diagnostics) != 0 {
		t.Fatalf("unexpected loss: %+v", diagnostics)
	}
	if len(decoded.Messages) != 1 || decoded.Messages[0].Content[0].ReasoningDetail.Data.Value != "gAAAA" {
		t.Fatalf("ciphertext missing: %+v", decoded.Messages)
	}
}

// The reply's own items are what a client sends back on the next turn, so the
// reasoning item arrives with the status the provider stamped on it. Rejecting
// that status would break the continuation the include path promised.
func TestResponsesReplayedReasoningItemKeepsItsProviderStatus(t *testing.T) {
	engine := NewBuiltinEngine()

	input := `[{"type":"reasoning","id":"rs_1","status":"completed","summary":[{"type":"summary_text","text":"thought"}],"encrypted_content":"gAAAA"},` +
		`{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"answer"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"more"}]}]`
	request, _, diagnostics, err := engine.DecodeRequest(
		llmprotocol.OpenAIResponsesV1,
		[]byte(`{"model":"m","input":`+input+`}`),
	)
	if err != nil {
		t.Fatalf("replaying a reply's own items: %v", err)
	}
	if len(diagnostics) != 0 {
		t.Fatalf("unexpected loss: %+v", diagnostics)
	}

	kept := false
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentReasoning && content.Text == "thought" {
				kept = true
			}
		}
	}
	if !kept {
		t.Fatalf("the replayed reasoning summary is missing: %+v", request.Messages)
	}
}

// Encrypted-only items complete with ciphertext and no invented reasoning text.
func TestResponsesOpaqueReasoningReachesTheClientWithNoReasoningText(t *testing.T) {
	frames := responsesReasoningFrames(t)
	completed := frames[10]["item"].(map[string]any)
	terminal := frames[17]["response"].(map[string]any)["output"].([]any)[0].(map[string]any)
	for _, item := range []map[string]any{completed, terminal} {
		item["summary"] = []any{}
		item["content"] = []any{}
		item["encrypted_content"] = "gAAAA"
	}
	events, err := decodeResponsesReasoningFrames(t, append(frames[:2], frames[10:]...))
	if err != nil {
		t.Fatal(err)
	}
	encoder := (OpenAIResponsesCodec{}).NewEncoder(llmprotocol.StreamContext{}, llmprotocol.DefaultPolicy())
	var stream strings.Builder
	for _, event := range events {
		frames, _, pushErr := encoder.Push(event)
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		for _, frame := range frames {
			stream.Write(frame)
		}
	}
	if strings.Contains(stream.String(), "reasoning_text") {
		t.Fatalf("the client was given reasoning text the provider never sent:\n%s", stream.String())
	}
	if !strings.Contains(stream.String(), `"type":"reasoning"`) {
		t.Fatalf("the reasoning item itself is missing:\n%s", stream.String())
	}
	if strings.Count(stream.String(), `"encrypted_content":"gAAAA"`) != 2 {
		t.Fatalf("encrypted-only item lost its blob at item completion or terminal output: %s", stream.String())
	}
	// The client replays the exact completed item on its next turn.
	var done map[string]any
	for _, line := range strings.Split(stream.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") || !strings.Contains(line, `"response.output_item.done"`) {
			continue
		}
		var frame map[string]any
		if decodeErr := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if item := frame["item"].(map[string]any); item["type"] == "reasoning" {
			done = item
		}
	}
	replayed, err := json.Marshal([]any{done, map[string]any{
		"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "more"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":`+string(replayed)+`}`))
	if err != nil {
		t.Fatalf("replaying the completed reasoning item: %v", err)
	}
	body, _, err := (OpenAIResponsesCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"encrypted_content":"gAAAA"`) || !strings.Contains(string(body), `"content":[]`) {
		t.Fatalf("the replayed ciphertext or empty content is missing: %s", body)
	}
}

// A client that already holds such an empty reasoning item sends it back; the
// item carries nothing, so it is left out rather than sent on as plaintext.
func TestResponsesEmptyReasoningTextIsLeftOut(t *testing.T) {
	engine := NewBuiltinEngine()
	input := `[{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":""}]},` +
		`{"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"thought"}],"content":[{"type":"reasoning_text","text":""}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"more"}]}]`
	request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":`+input+`}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _, err := (OpenAIResponsesCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "reasoning_text") || strings.Contains(string(body), "rs_1") {
		t.Fatalf("an empty reasoning item reached the target: %s", body)
	}
	if !strings.Contains(string(body), "thought") {
		t.Fatalf("the summary beside an empty reasoning text is missing: %s", body)
	}
}

func TestResponsesEncryptedOnlyBufferedReply(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"resp_1","object":"response","created_at":0,"model":"m","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque+/=\\value"}],"error":null,"incomplete_details":null}`)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			result, err := engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, target, body, func(response *llmprotocol.Response) error { response.Model = "public"; return nil })
			if err != nil {
				t.Fatal(err)
			}
			if target != llmprotocol.OpenAIResponsesV1 {
				if strings.Contains(string(result.Body), "opaque") {
					t.Fatalf("ciphertext leaked: %s", result.Body)
				}
				assertDroppedFields(t, result.Diagnostics, "reasoning_details")
				return
			}
			var wire responsesResponseWire
			if err := json.Unmarshal(result.Body, &wire); err != nil {
				t.Fatal(err)
			}
			var items []responsesItemWire
			if err := json.Unmarshal(wire.Output, &items); err != nil {
				t.Fatal(err)
			}
			if len(items) != 1 || items[0].ID != "rs_1" || string(items[0].EncryptedContent) != `"opaque+/=\\value"` || strings.Contains(string(items[0].Content), "reasoning_text") {
				t.Fatalf("encrypted-only item changed: %s", result.Body)
			}
		})
	}
}

func TestResponsesStreamUsesItemDoneCiphertext(t *testing.T) {
	frames := responsesReasoningFrames(t)
	frames[10]["item"].(map[string]any)["encrypted_content"] = "item-done"
	frames[17]["response"].(map[string]any)["output"].([]any)[0].(map[string]any)["encrypted_content"] = "terminal"
	events, err := decodeResponsesReasoningFrames(t, frames)
	if err != nil {
		t.Fatal(err)
	}
	encoder := (OpenAIResponsesCodec{}).NewEncoder(llmprotocol.StreamContext{}, llmprotocol.DefaultPolicy())
	var stream strings.Builder
	details := 0
	for _, event := range events {
		if event.Content != nil && event.Content.ReasoningDetail != nil {
			details++
			if event.Content.ReasoningDetail.Data.Value != "item-done" {
				t.Fatalf("wrong ciphertext snapshot: %+v", event)
			}
		}
		encoded, _, err := encoder.Push(event)
		if err != nil {
			t.Fatal(err)
		}
		for _, frame := range encoded {
			stream.Write(frame)
		}
	}
	if details != 1 || strings.Count(stream.String(), `"encrypted_content":"item-done"`) != 2 || strings.Contains(stream.String(), `"encrypted_content":"terminal"`) {
		t.Fatalf("ciphertext snapshot was lost, replaced or concatenated: %s", stream.String())
	}
}

func TestResponsesEncryptedReasoningValidation(t *testing.T) {
	for _, value := range []string{`1`, `{}`, `[]`, `true`} {
		_, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":[{"type":"reasoning","id":"rs","encrypted_content":`+value+`}]}`))
		if err == nil {
			t.Fatalf("accepted non-string ciphertext: %s", value)
		}
	}
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.TextBytes = 3
	_, _, _, err := (OpenAIResponsesCodec{}).DecodeRequest([]byte(`{"model":"m","input":[{"type":"reasoning","id":"rs","encrypted_content":"1234"}]}`), policy)
	if err == nil {
		t.Fatal("accepted ciphertext over the payload limit")
	}
}

func TestResponsesContinuityRequestProjectionAndDirectReplyRejection(t *testing.T) {
	engine := NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","include":["reasoning.encrypted_content"],"input":[{"type":"reasoning","id":"rs_1","encrypted_content":"cipher"},{"role":"user","content":"next"}]}`))
	require.NoError(t, err)
	for _, codec := range []Codec{OpenAIChatCodec{}, AnthropicMessagesCodec{}} {
		body, diagnostics, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
		require.NoError(t, err)
		require.NotContains(t, string(body), "cipher")
		require.NotContains(t, string(body), "redacted_thinking")
		require.NotContains(t, string(body), "signature")
		require.NotContains(t, string(body), "include")
		require.Contains(t, string(body), "next")
		require.Equal(t, "include", diagnostics[0].Field)
		require.Equal(t, "input[].encrypted_content", diagnostics[1].Field)
		encoded, err := engine.EncodeRequest(codec.Format(), request, llmprotocol.Envelope{})
		require.NoError(t, err)
		require.NotContains(t, string(encoded.Body), "cipher")
		response := llmprotocol.Response{Generation: 1, ID: "r", Model: "m", Output: []llmprotocol.OutputItem{{ID: "rs_1", Role: llmprotocol.RoleAssistant, Content: request.Messages[0].Content}}}
		_, _, err = codec.EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
		encoder := codec.(StreamCodec).NewEncoder(llmprotocol.StreamContext{Context: t.Context()}, llmprotocol.DefaultPolicy())
		_, _, err = encoder.Push(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, Content: &request.Messages[0].Content[0]})
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
	}
}

func TestResponsesEncryptedReasoningDecodeResponseStream(t *testing.T) {
	for _, summary := range []bool{false, true} {
		t.Run(map[bool]string{false: "encrypted_only", true: "summary_before_blob"}[summary], func(t *testing.T) {
			frames := responsesReasoningFrames(t)
			if !summary {
				for _, item := range []map[string]any{
					frames[10]["item"].(map[string]any),
					frames[17]["response"].(map[string]any)["output"].([]any)[0].(map[string]any),
				} {
					item["summary"], item["content"] = []any{}, []any{}
				}
				frames = append(frames[:2], frames[10:]...)
			}
			var stream strings.Builder
			for sequence, frame := range frames {
				frame["sequence_number"] = sequence
				body, err := json.Marshal(frame)
				require.NoError(t, err)
				stream.WriteString("event: " + frame["type"].(string) + "\ndata: " + string(body) + "\n\n")
			}
			response, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.OpenAIResponsesV1, []byte(stream.String()), llmprotocol.StreamContext{Context: t.Context()})
			require.NoError(t, err)
			var blob string
			var summaryText string
			for _, part := range response.Output[0].Content {
				if llmprotocol.IsResponsesReasoning(part.ReasoningDetail) {
					blob = part.ReasoningDetail.Data.Value
				}
				if part.Reasoning == llmprotocol.ReasoningScopeSummary {
					summaryText += part.Text
				}
			}
			require.Equal(t, "cipher_A", blob)
			if summary {
				require.NotEmpty(t, summaryText)
			} else {
				require.Len(t, response.Output[0].Content, 1)
			}
		})
	}
}

func TestResponsesCompletedItemChoosesLowestBlobIndex(t *testing.T) {
	encoder := (OpenAIResponsesCodec{}).NewEncoder(llmprotocol.StreamContext{Context: t.Context()}, llmprotocol.DefaultPolicy())
	_, _, err := encoder.Push(llmprotocol.Event{Type: llmprotocol.EventResponseStarted, ResponseID: "r", Model: "m"})
	require.NoError(t, err)
	_, _, err = encoder.Push(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemID: "rs_1", Content: &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}})
	require.NoError(t, err)
	for _, index := range []int{3, 1} {
		blob, decodeErr := decodeResponsesEncryptedReasoning(responsesItemWire{ID: "rs_1", EncryptedContent: json.RawMessage(`"cipher"`)}, llmprotocol.DefaultPolicy(), true)
		require.NoError(t, decodeErr)
		if index == 1 {
			blob.ReasoningDetail.Data.Value = "first"
		}
		_, _, err = encoder.Push(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, ItemID: "rs_1", ContentIndex: index, Content: blob})
		require.NoError(t, err)
	}
	frames, _, err := encoder.Push(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemID: "rs_1", Content: &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}})
	require.NoError(t, err)
	require.Contains(t, string(bytes.Join(frames, nil)), `"encrypted_content":"first"`)
}
