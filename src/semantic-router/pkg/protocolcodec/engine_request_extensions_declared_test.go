package protocolcodec

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestCaptureSkipsWireDeclaredMembers(t *testing.T) {
	policy := extensionTestPolicy()
	policy.AllowedRequestExtensions = []string{"temperature", "vllm_xargs"}
	engine := extensionTestEngine(policy)
	body := []byte(`{"model":"auto","temperature":0.9,"messages":[{"role":"user","content":"hi"}],"vllm_xargs":{"agentic_context":"abc"}}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, captured := request.ExtensionFields["temperature"]; captured {
		t.Fatalf("a wire-declared member was captured: %v", request.ExtensionFields)
	}
	if request.Sampling.Temperature == nil || *request.Sampling.Temperature != 0.9 {
		t.Fatalf("declared member must take the normal decode path: %+v", request.Sampling)
	}
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if strings.Count(string(result.Body), `"temperature"`) != 1 {
		t.Fatalf("encoded body must carry exactly one temperature key: %s", result.Body)
	}
	if !bytes.Contains(result.Body, []byte(`"vllm_xargs":{"agentic_context":"abc"}`)) {
		t.Fatalf("the extension member must still merge: %s", result.Body)
	}
}
