package contextcompression

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestTransformationProtocolRoundTrips(t *testing.T) {
	for _, tc := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{"chat", llmprotocol.OpenAIChatV1, `{"model":"model","messages":[{"role":"developer","content":"Keep permissions"},{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"second"},{"role":"assistant","content":"second answer"},{"role":"user","content":"live"}]}`},
		{"responses", llmprotocol.OpenAIResponsesV1, `{"model":"model","instructions":"Keep permissions","input":[{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"second"},{"role":"assistant","content":"second answer"},{"role":"user","content":"live"}]}`},
		{"anthropic", llmprotocol.AnthropicMessagesV1, `{"model":"model","max_tokens":100,"system":"Keep permissions","messages":[{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"second"},{"role":"assistant","content":"second answer"},{"role":"user","content":"live"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertTransformationProtocolRoundTrip(t, tc.format, tc.body)
		})
	}
}

func assertTransformationProtocolRoundTrip(t *testing.T, format llmprotocol.WireFormat, body string) {
	t.Helper()
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(format, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := engine.EncodeRequest(format, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	ir := ParseSemanticRequest(&request, Provenance{})
	if err = ir.ApplySteps(context.Background(), []TransformationStep{{Kind: TransformReset}}); err != nil {
		t.Fatal(err)
	}
	unchanged, err := engine.EncodeRequest(format, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(baseline.Body, unchanged.Body) {
		t.Fatal("disabled plan changed provider bytes")
	}
	if err = ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformSelectTurns, 0, 1)}); err != nil {
		t.Fatal(err)
	}
	result, err := engine.EncodeRequest(format, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, _, err := engine.DecodeRequest(format, result.Body)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Messages) != 3 || decoded.Messages[2].Content[0].Text != "live" {
		t.Fatal("wrong conversation after round trip")
	}
	if encoded(t, request.Instructions) != encoded(t, decoded.Instructions) {
		t.Fatal("instructions changed")
	}
}

func TestTransformationRawHistoryRetainsUnknownFields(t *testing.T) {
	body := map[string]interface{}{"model": "model", "vendor_extension": map[string]interface{}{"keep": true}, "messages": []interface{}{
		map[string]interface{}{"role": "user", "content": "old"},
		map[string]interface{}{"role": "assistant", "content": "answer"},
		map[string]interface{}{"role": "user", "content": "second"},
		map[string]interface{}{"role": "assistant", "content": "second answer"},
		map[string]interface{}{"role": "user", "content": "live", "vendor_metadata": "keep"},
	}}
	ir := ParseRequestIR(body, Provenance{})
	before := encoded(t, body)
	if err := ir.ApplySteps(context.Background(), []TransformationStep{{Kind: TransformReset}}); err != nil {
		t.Fatal(err)
	}
	if before != encoded(t, body) {
		t.Fatal("raw baseline changed")
	}
	if err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformSelectTurns, 0, 1)}); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]interface{})
	if len(messages) != 3 || messages[2].(map[string]interface{})["vendor_metadata"] != "keep" || body["vendor_extension"] == nil {
		t.Fatal("raw metadata lost")
	}
}
