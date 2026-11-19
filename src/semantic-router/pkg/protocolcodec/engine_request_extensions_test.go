package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func extensionTestPolicy() llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	policy.AllowedRequestExtensions = []string{"vllm_xargs"}
	return policy
}

func extensionTestEngine(policy llmprotocol.Policy) *Engine {
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		panic(err)
	}
	return engine
}

func extensionTestBody() []byte {
	return []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}],"vllm_xargs":{"agentic_context":"abc"}}`)
}

func TestCaptureRequestExtensions(t *testing.T) {
	captured, stripped, err := captureRequestExtensions(llmprotocol.OpenAIChatV1, extensionTestBody(), extensionTestPolicy())
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	if len(captured) != 1 {
		t.Fatalf("captured = %v", captured)
	}
	if string(captured["vllm_xargs"]) != `{"agentic_context":"abc"}` {
		t.Fatalf("captured value = %s", captured["vllm_xargs"])
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(stripped, &decoded); err != nil {
		t.Fatalf("stripped body does not parse: %v", err)
	}
	if _, present := decoded["vllm_xargs"]; present {
		t.Fatalf("stripped body still carries the member: %s", stripped)
	}
	if _, present := decoded["model"]; !present {
		t.Fatalf("stripped body lost the declared members: %s", stripped)
	}
}

func TestDecodeRequestForMutationCarriesExtensionsAcrossRewrite(t *testing.T) {
	engine := extensionTestEngine(extensionTestPolicy())
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, extensionTestBody())
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(request.ExtensionFields) != 1 || request.ExtensionFieldsFormat != llmprotocol.OpenAIChatV1 {
		t.Fatalf("extensions = %v, format = %q", request.ExtensionFields, request.ExtensionFieldsFormat)
	}
	// Simulate the router's model rewrite: the mutation retires byte replay
	// and forces re-serialization, which is exactly where an unmodelled field
	// would silently vanish without the capture.
	request.Model = "qwen3.5-9b"
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !bytes.Contains(result.Body, []byte(`"vllm_xargs":{"agentic_context":"abc"}`)) {
		t.Fatalf("merged body lost the extension: %s", result.Body)
	}
	if strings.Count(string(result.Body), "vllm_xargs") != 1 {
		t.Fatalf("merged body duplicates the extension: %s", result.Body)
	}
}

func TestReplayedBodyKeepsExtensionsByteIdentically(t *testing.T) {
	engine := extensionTestEngine(extensionTestPolicy())
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, extensionTestBody())
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	result, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !bytes.Equal(result.Body, extensionTestBody()) {
		t.Fatalf("replayed body drifted: %s", result.Body)
	}
}

func TestUnlistedUnknownMemberStillFailsClosed(t *testing.T) {
	engine := extensionTestEngine(extensionTestPolicy())
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}],"some_other_field":1}`)
	_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err == nil {
		t.Fatal("an unlisted unknown member must keep failing closed")
	}
}

func TestExtensionsRequireConfiguration(t *testing.T) {
	engine := extensionTestEngine(llmprotocol.DefaultPolicy())
	_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, extensionTestBody())
	if err == nil {
		t.Fatal("without configuration the extension member must fail closed")
	}
}

func TestCrossFormatDispatchDropsExtensionsWithDiagnostic(t *testing.T) {
	engine := extensionTestEngine(extensionTestPolicy())
	result, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, extensionTestBody(), nil)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if bytes.Contains(result.Body, []byte("vllm_xargs")) {
		t.Fatalf("cross-format body must not carry the member: %s", result.Body)
	}
	found := false
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Field == "vllm_xargs" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			found = true
		}
	}
	if !found {
		t.Fatalf("cross-format drop must be diagnosed: %#v", result.Diagnostics)
	}
}

func TestExtensionBudgetRejectsOversizedMember(t *testing.T) {
	policy := extensionTestPolicy()
	policy.Limits.RequestExtensionBytes = 8
	_, _, err := captureRequestExtensions(llmprotocol.OpenAIChatV1, extensionTestBody(), policy)
	if err == nil {
		t.Fatal("an oversized member must fail closed")
	}
}

func TestDuplicateExtensionMemberRejected(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}],"vllm_xargs":{"a":1},"vllm_xargs":{"b":2}}`)
	_, _, err := captureRequestExtensions(llmprotocol.OpenAIChatV1, body, extensionTestPolicy())
	if err == nil {
		t.Fatal("a duplicate whitelisted member must fail closed")
	}
}
