package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A store:false client that asks for provider-encrypted reasoning keeps the
// item the Router returns and sends it back on the next turn. Accepting the
// include and then rejecting the item leaves that client with no continuation.
func TestResponsesEncryptedReasoningReplayIsAccepted(t *testing.T) {
	engine := NewBuiltinEngine()

	request := []byte(`{"model":"m","input":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAA"}]}`)
	_, _, diagnostics, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, request)
	if err != nil {
		t.Fatalf("replaying a reasoning item that carries encrypted_content: %v", err)
	}
	assertDroppedFields(t, diagnostics, "input[].encrypted_content")
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
	assertDroppedFields(t, diagnostics, "input[].encrypted_content")

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

// A reasoning item that carries only the provider's encrypted reasoning, as
// ChatGPT's Codex backend sends with no summary, completes with no reasoning
// text: a client that sent an empty reasoning_text back would have it reach an
// OpenAI target as plaintext reasoning, which it refuses.
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
	// The client sends the completed item back as it got it; the next turn
	// decodes and reaches the target without it.
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
	if strings.Contains(string(body), `"reasoning"`) {
		t.Fatalf("the empty reasoning item reached the target: %s", body)
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
