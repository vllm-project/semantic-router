package protocolcodec

import (
	"encoding/json"
	"errors"
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

func newChatPassthroughEngine(t *testing.T) *Engine {
	t.Helper()
	policy := llmprotocol.DefaultPolicy()
	policy.UnknownFields = llmprotocol.UnknownPreserveSameFormat
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	return engine
}

func TestChatUnmodeledFieldsSurviveSameFormatMutation(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}],` +
		`"logprobs":false,"top_logprobs":3,"logit_bias":{"50256":-100},` +
		`"guided_json":{"type":"object"},"service_tier":"auto"}`)

	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"logprobs", "top_logprobs", "logit_bias", "guided_json", "service_tier"} {
		if _, ok := envelope.RequestPassthrough[key]; !ok {
			t.Fatalf("%s was not captured for passthrough", key)
		}
	}

	request.Model, request.Generation = "public/target", request.Generation+1
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var after map[string]json.RawMessage
	if err := json.Unmarshal(encoded.Body, &after); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"logprobs", "top_logprobs", "logit_bias", "guided_json", "service_tier"} {
		if _, ok := after[key]; !ok {
			t.Fatalf("%s missing after mutation: %s", key, encoded.Body)
		}
	}
	if string(after["model"]) != `"public/target"` {
		t.Fatalf("model mutation was not applied: %s", after["model"])
	}
}

func TestChatUnmodeledFieldsRemainStrictByDefault(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, body := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":false}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"guided_json":{"type":"object"}}`,
	} {
		if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(body)); err == nil {
			t.Fatalf("strict policy accepted %s", body)
		}
	}
}

func TestChatNonCanonicalKnownFieldNameStaysRejected(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	for _, body := range []string{
		`{"Model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","Messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"Role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"user","Content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"Logprobs":true}`,
	} {
		_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(body))
		var protocolErr *llmprotocol.ProtocolError
		if !errors.As(err, &protocolErr) || protocolErr.Code != "invalid_json" {
			t.Fatalf("case variant was not rejected as non-canonical: %s (%v)", body, err)
		}
	}
}

func TestChatNestedUnknownFieldSurvivesMutation(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	body := []byte(`{"model":"public/model","messages":[{"role":"user","content":"hi","future_message_field":1,"vendor_meta":{"k":"v"}},{"role":"user","content":"second"}],"future_top_level":2}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "backend/model"
	request.Generation++ // Force a real encode instead of source replay.
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(encoded.Body, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded["model"] != "backend/model" {
		t.Fatalf("model rewrite lost: %s", encoded.Body)
	}
	if decoded["future_top_level"].(float64) != 2 {
		t.Fatalf("top-level passthrough lost: %s", encoded.Body)
	}
	messages := decoded["messages"].([]interface{})
	first := messages[0].(map[string]interface{})
	if first["future_message_field"].(float64) != 1 {
		t.Fatalf("nested passthrough lost: %s", encoded.Body)
	}
	if first["vendor_meta"].(map[string]interface{})["k"] != "v" {
		t.Fatalf("nested object passthrough lost: %s", encoded.Body)
	}
	if _, exists := messages[1].(map[string]interface{})["future_message_field"]; exists {
		t.Fatalf("nested passthrough leaked across messages: %s", encoded.Body)
	}
}

func TestChatPassthroughCrossFormatFailsLoudly(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	body := []byte(`{"model":"public/model","messages":[{"role":"user","content":"hi"}],"service_tier":"auto"}`)
	_, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request := llmprotocol.Request{Model: "backend/model", Generation: 2}
	_, err = engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err == nil {
		t.Fatal("cross-format encode with preserved fields must fail loudly")
	}
	if !strings.Contains(err.Error(), "passthrough") && !strings.Contains(err.Error(), "same-format") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestChatMessagePassthroughBudgetExceeded(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	big := strings.Repeat("x", 300*1024)
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi","blob":"` + big + `"}]}`)
	_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err == nil {
		t.Fatal("oversized nested passthrough must be rejected")
	}
}

func TestChatNestedUnknownFieldSurvivesReplay(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	body := []byte(`{"model":"public/model","messages":[{"role":"user","content":"hi","future_message_field":1}],"future_top_level":2}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded.Body) != string(body) {
		t.Fatalf("unchanged same-format request was not replayed byte-for-byte: %s", encoded.Body)
	}
}
