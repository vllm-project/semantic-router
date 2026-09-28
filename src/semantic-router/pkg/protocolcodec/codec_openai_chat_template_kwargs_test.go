package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIChatAcceptsChatTemplateKwargs(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()
	request, _, _, err := codec.DecodeRequest([]byte(`{
		"model":"qwen3.5-4b",
		"messages":[{"role":"user","content":"hi"}],
		"chat_template_kwargs":{"enable_thinking":false}
	}`), policy)
	if err != nil {
		t.Fatalf("decode request with chat_template_kwargs: %v", err)
	}
	if len(request.ChatTemplateKwargs) == 0 {
		t.Fatalf("neutral ChatTemplateKwargs is empty")
	}
	var kwargs map[string]interface{}
	if err := json.Unmarshal(request.ChatTemplateKwargs, &kwargs); err != nil {
		t.Fatalf("unmarshal ChatTemplateKwargs: %v", err)
	}
	if kwargs["enable_thinking"] != false {
		t.Fatalf("enable_thinking = %#v", kwargs["enable_thinking"])
	}
}

func TestOpenAIChatPreservesChatTemplateKwargsAcrossModelRewrite(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()
	request, envelope, _, err := codec.DecodeRequest([]byte(`{
		"model":"auto",
		"messages":[{"role":"user","content":"hi"}],
		"chat_template_kwargs":{"enable_thinking":false}
	}`), policy)
	if err != nil {
		t.Fatalf("decode request: %v", err)
	}
	// Simulate the auto-routing model rewrite: the model is replaced and the
	// envelope generation is bumped, which disables byte replay and forces
	// re-serialization from the neutral request.
	request.Model = "qwen3.5-4b"
	request.Generation++
	body, _, err := codec.EncodeRequest(request, envelope, policy)
	if err != nil {
		t.Fatalf("encode request after model rewrite: %v", err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded wire: %v", err)
	}
	raw, ok := wire["chat_template_kwargs"]
	if !ok {
		t.Fatalf("chat_template_kwargs missing after model rewrite: %s", body)
	}
	var kwargs map[string]interface{}
	if err := json.Unmarshal(raw, &kwargs); err != nil {
		t.Fatalf("unmarshal chat_template_kwargs: %v", err)
	}
	if kwargs["enable_thinking"] != false {
		t.Fatalf("enable_thinking = %#v", kwargs["enable_thinking"])
	}
}

func TestOpenAIChatOmitsChatTemplateKwargsWhenAbsent(t *testing.T) {
	request := llmprotocol.Request{
		Generation: 2,
		Model:      "model-a",
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText,
				Text: "hi",
			}},
		}},
	}
	body, _, err := (OpenAIChatCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode request: %v", err)
	}
	if bytes.Contains(body, []byte("chat_template_kwargs")) {
		t.Fatalf("encoded body must not contain chat_template_kwargs: %s", body)
	}
}
