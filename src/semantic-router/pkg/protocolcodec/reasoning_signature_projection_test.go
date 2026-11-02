package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func signatureOptInEngine(t *testing.T) *Engine {
	t.Helper()
	policy := llmprotocol.DefaultPolicy()
	policy.AllowedLossyFields = []string{llmprotocol.LossyFieldReasoningSignature}
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	return engine
}

// A Messages backend that signs its thinking, as vLLM's does, toward a Chat or
// Responses client: refused by default, translated without the signature once
// the operator accepts the loss.
func TestSignedThinkingStreamsToOpenAIClientsOnlyWithTheOptIn(t *testing.T) {
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"model\":\"model\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"first thought\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"signature-1\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	streamContext := llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
		t.Run(string(target), func(t *testing.T) {
			strict, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target, streamContext)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, _, pushErr := strict.Push(payload); pushErr == nil {
				t.Fatal("signed reasoning crossed formats without the opt-in")
			}
			stream, err := signatureOptInEngine(t).NewStream(llmprotocol.AnthropicMessagesV1, target, streamContext)
			if err != nil {
				t.Fatal(err)
			}
			frames, _, diagnostics, err := stream.Push(payload)
			if err != nil {
				t.Fatalf("Push() error = %v", err)
			}
			wire := bytes.Join(frames, nil)
			if !bytes.Contains(wire, []byte("first thought")) || !bytes.Contains(wire, []byte("answer")) || bytes.Contains(wire, []byte("signature-1")) {
				t.Fatalf("wire = %s", wire)
			}
			if !hasDiagnostic(diagnostics, llmprotocol.LossyFieldReasoningSignature, llmprotocol.DiagnosticApproximated) {
				t.Fatalf("diagnostics = %+v, want reasoning.signature approximated", diagnostics)
			}
		})
	}
}

func TestSignedThinkingTranslatesBufferedOnlyWithTheOptIn(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"model",` +
		`"content":[{"type":"thinking","thinking":"first thought","signature":"signature-1"},{"type":"text","text":"answer"}],` +
		`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":3}}`)
	_, err := NewBuiltinEngine().TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, body, nil)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_capability" {
		t.Fatalf("TranslateResponse() without the opt-in error = %v, want unsupported_capability", err)
	}
	result, err := signatureOptInEngine(t).TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, body, nil)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if !bytes.Contains(result.Body, []byte("first thought")) || bytes.Contains(result.Body, []byte("signature-1")) {
		t.Fatalf("body = %s", result.Body)
	}
	if !hasDiagnostic(result.Diagnostics, llmprotocol.LossyFieldReasoningSignature, llmprotocol.DiagnosticApproximated) {
		t.Fatalf("diagnostics = %+v, want reasoning.signature approximated", result.Diagnostics)
	}
}
