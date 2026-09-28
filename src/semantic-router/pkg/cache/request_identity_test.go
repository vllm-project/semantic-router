package cache

import (
	"testing"
)

func TestBuildRequestIdentitySeparatesStreamingResponseMode(t *testing.T) {
	base := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"temperature":0.2}`)
	streaming := []byte(`{"stream":true,"temperature":0.2,"messages":[{"content":"hello","role":"user"}],"model":"m"}`)

	first, err := BuildRequestIdentity(base)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(base): %v", err)
	}
	second, err := BuildRequestIdentity(streaming)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(streaming): %v", err)
	}
	if first.ExactFingerprint == second.ExactFingerprint {
		t.Fatal("streaming and non-streaming response modes must not share exact entries")
	}
	if first.CompatibilityFingerprint == second.CompatibilityFingerprint {
		t.Fatal("streaming and non-streaming response modes must not share semantic entries")
	}
}

func TestBuildRequestIdentitySupportsMultipartWithoutEmbeddingBinaryParts(t *testing.T) {
	first, err := BuildRequestIdentity([]byte(`{
		"model":"vision",
		"messages":[{"role":"user","content":[
			{"type":"text","text":"describe this"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
		]}]
	}`))
	if err != nil {
		t.Fatalf("BuildRequestIdentity() error = %v", err)
	}
	second, err := BuildRequestIdentity([]byte(`{
		"model":"vision",
		"messages":[{"role":"user","content":[
			{"type":"text","text":"what is shown"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
		]}]
	}`))
	if err != nil {
		t.Fatalf("BuildRequestIdentity() error = %v", err)
	}
	if first.Query != "describe this" || !first.SemanticSafe {
		t.Fatalf("multipart identity = %#v", first)
	}
	if first.CompatibilityFingerprint != second.CompatibilityFingerprint {
		t.Fatal("text replacement did not preserve the shared image digest")
	}
}

func TestBuildRequestIdentitySupportsResponsesInput(t *testing.T) {
	identity, err := BuildRequestIdentity([]byte(`{
		"model":"gpt",
		"input":[{"role":"user","content":[
			{"type":"input_text","text":"hello"},
			{"type":"input_file","file_id":"file-secret"}
		]}]
	}`))
	if err != nil {
		t.Fatalf("BuildRequestIdentity() error = %v", err)
	}
	if identity.Query != "hello" || !identity.SemanticSafe {
		t.Fatalf("responses identity = %#v", identity)
	}
}

func TestBuildRequestIdentityPreservesAnthropicToolBlocks(t *testing.T) {
	first, err := BuildRequestIdentity([]byte(`{
		"model":"claude",
		"messages":[{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"tool-1","content":"private result"},
			{"type":"text","text":"summarize"}
		]}]
	}`))
	if err != nil {
		t.Fatalf("BuildRequestIdentity() error = %v", err)
	}
	second, err := BuildRequestIdentity([]byte(`{
		"model":"claude",
		"messages":[{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"tool-2","content":"different result"},
			{"type":"text","text":"summarize differently"}
		]}]
	}`))
	if err != nil {
		t.Fatalf("BuildRequestIdentity() error = %v", err)
	}
	if first.CompatibilityFingerprint == second.CompatibilityFingerprint {
		t.Fatal("tool block digest was omitted from compatibility identity")
	}
}

func TestBuildRequestIdentitySemanticCompatibilityIgnoresOnlyCurrentUserText(t *testing.T) {
	firstBody := []byte(`{"model":"auto","messages":[{"role":"system","content":"be concise"},{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup"}}]}`)
	secondBody := []byte(`{"model":"auto","messages":[{"role":"system","content":"be concise"},{"role":"user","content":"hi there"}],"tools":[{"type":"function","function":{"name":"lookup"}}]}`)
	differentSystem := []byte(`{"model":"auto","messages":[{"role":"system","content":"be exhaustive"},{"role":"user","content":"hi there"}],"tools":[{"type":"function","function":{"name":"lookup"}}]}`)

	first, err := BuildRequestIdentity(firstBody)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(first): %v", err)
	}
	second, err := BuildRequestIdentity(secondBody)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(second): %v", err)
	}
	third, err := BuildRequestIdentity(differentSystem)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(differentSystem): %v", err)
	}

	if first.CompatibilityFingerprint != second.CompatibilityFingerprint {
		t.Fatal("current user wording should not split semantic compatibility scope")
	}
	if second.CompatibilityFingerprint == third.CompatibilityFingerprint {
		t.Fatal("system prompt must participate in semantic compatibility scope")
	}
}

func TestBuildRequestIdentityPreservesLargeJSONNumbersAndClientUser(t *testing.T) {
	firstBody := []byte(`{"model":"m","user":"alice","seed":9007199254740992,"messages":[{"role":"user","content":"hello"}]}`)
	secondBody := []byte(`{"model":"m","user":"alice","seed":9007199254740993,"messages":[{"role":"user","content":"hello"}]}`)
	otherUser := []byte(`{"model":"m","user":"bob","seed":9007199254740992,"messages":[{"role":"user","content":"hello"}]}`)

	first, err := BuildRequestIdentity(firstBody)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(first): %v", err)
	}
	second, err := BuildRequestIdentity(secondBody)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(second): %v", err)
	}
	third, err := BuildRequestIdentity(otherUser)
	if err != nil {
		t.Fatalf("BuildRequestIdentity(otherUser): %v", err)
	}
	if first.ExactFingerprint == second.ExactFingerprint {
		t.Fatal("distinct large integer seeds must not collide")
	}
	if first.CompatibilityFingerprint == second.CompatibilityFingerprint {
		t.Fatal("generation seeds must participate in semantic compatibility")
	}
	if first.CompatibilityFingerprint == third.CompatibilityFingerprint {
		t.Fatal("client user must participate in semantic compatibility")
	}
}

func TestBuildRequestIdentityUsesTextSemanticViewForMultimodalUser(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.test/a.png"}}]}]}`)
	identity, err := BuildRequestIdentity(body)
	if err != nil {
		t.Fatalf("BuildRequestIdentity: %v", err)
	}
	if !identity.SemanticSafe || identity.Query != "describe" {
		t.Fatalf("multimodal semantic view = %#v", identity)
	}
	if identity.ExactFingerprint == "" {
		t.Fatal("exact fingerprint should still be available")
	}
}
