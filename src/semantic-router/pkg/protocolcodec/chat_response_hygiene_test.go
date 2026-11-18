package protocolcodec

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// chatServerMembers lists the server members found in a Chat reply or chunk,
// at the top level or in its choices.
func chatServerMembers(t *testing.T, body []byte) []string {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatalf("%v: %s", err, body)
	}
	var found []string
	for _, name := range chatServerFields {
		if _, present := object[name]; present {
			found = append(found, name)
		}
	}
	var choices []map[string]json.RawMessage
	_ = json.Unmarshal(object["choices"], &choices)
	for _, choice := range choices {
		for _, name := range chatChoiceServerFields {
			if _, present := choice[name]; present {
				found = append(found, "choices."+name)
			}
		}
	}
	return found
}

func publicChatStream(t *testing.T, body []byte) []byte {
	t.Helper()
	filter := NewChatPublicStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes, true)
	public, err := filter.Push(body)
	if err != nil {
		t.Fatal(err)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	return append(public, final...)
}

// sseData returns the data of each event of a stream, the [DONE] sentinel
// left out.
func sseData(t *testing.T, body []byte) [][]byte {
	t.Helper()
	var chunks [][]byte
	for _, frame := range bytes.Split(body, []byte("\n\n")) {
		_, data, found := bytes.Cut(frame, []byte("data: "))
		data = bytes.TrimSpace(data)
		if found && !bytes.Equal(data, []byte("[DONE]")) {
			chunks = append(chunks, data)
		}
	}
	return chunks
}

// Every captured vLLM reply carries system_fingerprint or null-valued serving
// members; a same-format Chat client receives none of them, and the rest of
// the reply as vLLM sent it, its reasoning member name included.
func TestChatClientsReceiveNoServerMembersBuffered(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, name := range vllmReplyCaptures {
		t.Run(name, func(t *testing.T) {
			raw := vllmCapture(t, name)
			if len(chatServerMembers(t, raw)) == 0 {
				t.Fatal("the capture carries no server members")
			}
			translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, raw, nil)
			if err != nil {
				t.Fatal(err)
			}
			if found := chatServerMembers(t, translated.Body); len(found) != 0 {
				t.Fatalf("Chat client received %v: %s", found, translated.Body)
			}
			var source, public map[string]json.RawMessage
			if json.Unmarshal(raw, &source) != nil || json.Unmarshal(translated.Body, &public) != nil {
				t.Fatal("reply is not a JSON object")
			}
			for member, value := range source {
				if member != "choices" && !slices.Contains(chatServerFields, member) && !jsonEquivalent(value, public[member]) {
					t.Fatalf("%s changed from %s to %s", member, value, public[member])
				}
			}
			if bytes.Contains(raw, []byte(`"reasoning":"`)) && !bytes.Contains(translated.Body, []byte(`"reasoning":"`)) {
				t.Fatalf("reasoning member renamed: %s", translated.Body)
			}
		})
	}
}

func TestChatClientsReceiveNoServerMembersStreamed(t *testing.T) {
	for _, name := range vllmStreamCaptures {
		t.Run(name, func(t *testing.T) {
			raw := vllmCapture(t, name)
			if !bytes.Contains(raw, []byte(`"system_fingerprint"`)) && !bytes.Contains(raw, []byte(`"kv_transfer_params"`)) &&
				!bytes.Contains(raw, []byte(`"metrics"`)) {
				t.Fatal("the capture carries no server members")
			}
			public := publicChatStream(t, raw)
			for _, chunk := range sseData(t, public) {
				if found := chatServerMembers(t, chunk); len(found) != 0 {
					t.Fatalf("Chat client received %v: %s", found, chunk)
				}
			}
			// The codec's own Chat frames never carried them.
			out, _, err := translateVLLMStream(NewBuiltinEngine(), llmprotocol.OpenAIChatV1, raw)
			if err != nil {
				t.Fatal(err)
			}
			for _, chunk := range sseData(t, []byte(out)) {
				if found := chatServerMembers(t, chunk); len(found) != 0 {
					t.Fatalf("translated Chat stream carries %v: %s", found, chunk)
				}
			}
		})
	}
}

// A reply or stream without server members reaches a same-format Chat client
// byte for byte.
func TestChatRepliesWithoutServerMembersPassUnchanged(t *testing.T) {
	stream := []byte("data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"a < b\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	if public := publicChatStream(t, stream); !bytes.Equal(public, stream) {
		t.Fatalf("public stream changed: %s", public)
	}
	reply := []byte(`{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"a < b"},"finish_reason":"stop"}]}`)
	translated, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, reply, nil)
	if err != nil || !bytes.Equal(translated.Body, reply) {
		t.Fatalf("same-format reply changed (%v): %s", err, translated.Body)
	}
}
