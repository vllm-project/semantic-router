package protocolcodec

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatInferenceControlsSurviveMutation(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, topK := range []int{-1, 0, 17} {
		body := []byte(fmt.Sprintf(`{"model":"auto","messages":[{"role":"user","content":"unchanged prompt"}],"top_k":%d,"min_p":0,"repetition_penalty":1.1,"cache_salt":"phase-salt","reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":true}}`, topK))
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
		if err != nil {
			t.Fatal(err)
		}
		request.Model, request.Generation = "backend", request.Generation+1
		encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
		if err != nil {
			t.Fatal(err)
		}
		var before, after map[string]json.RawMessage
		if err = json.Unmarshal(body, &before); err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(encoded.Body, &after); err != nil {
			t.Fatal(err)
		}
		for _, key := range []string{"messages", "top_k", "min_p", "repetition_penalty", "cache_salt", "reasoning_effort", "chat_template_kwargs"} {
			if string(before[key]) != string(after[key]) {
				t.Fatalf("%s changed: %s -> %s", key, before[key], after[key])
			}
		}
		if string(after["model"]) != `"backend"` {
			t.Fatal("model mutation was not applied")
		}
	}
}

func TestChatInferenceControlsValidateBeforeMutation(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, fields := range []string{
		`"top_k":-2`, `"top_k":1.5`, `"top_k":"1"`, `"min_p":-0.1`, `"min_p":1.1`,
		`"repetition_penalty":0`, `"repetition_penalty":-1`, `"cache_salt":""`,
		`"cache_salt":"a/b"`, `"cache_salt":"a\\b"`, `"cache_salt":"a@b"`, `"cache_salt":"a\u0000b"`,
		`"cache_salt":2`, `"cache_salt":"` + strings.Repeat("a", 129) + `"`, `"unknown_option":1`,
	} {
		body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],` + fields + `}`)
		if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body); err == nil {
			t.Fatalf("invalid control accepted: %s", fields)
		}
	}
	for _, fields := range []string{`"top_k":null,"min_p":null,"repetition_penalty":null,"cache_salt":null`, `"cache_salt":"` + strings.Repeat("a", 128) + `"`} {
		body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],` + fields + `}`)
		if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body); err != nil {
			t.Fatal(err)
		}
	}
}

func TestChatInferenceControlsCannotDisappearAcrossProtocols(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		for _, fields := range []string{`"min_p":0`, `"repetition_penalty":1`, `"cache_salt":"phase"`, `"chat_template_kwargs":{"enable_thinking":false}`, `"top_k":-1`} {
			body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],` + fields + `}`)
			if _, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, target, body, nil); err == nil {
				t.Fatalf("%s silently dropped %s", target, fields)
			}
		}
	}
}
