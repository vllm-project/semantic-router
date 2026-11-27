package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const redactedPayload = "EmwKAhgB...opaque=="

// Anthropic to Anthropic keeps a redacted thinking block byte-for-byte, in a
// request history and in a buffered reply, as signed thinking already does.
func TestRedactedThinkingSurvivesAnthropicToAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	requestBody := []byte(`{"model":"m","max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":1024},` +
		`"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":[{"type":"redacted_thinking","data":"` + redactedPayload + `"},{"type":"text","text":"x"}]},` +
		`{"role":"user","content":"y"}]}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, requestBody)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	block := request.Messages[1].Content[0]
	if block.Kind != llmprotocol.ContentReasoning || block.Reasoning != llmprotocol.ReasoningScopeRedacted ||
		block.Signature != redactedPayload || block.Text != "" {
		t.Fatalf("decoded redacted block = %+v", block)
	}
	if !llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityReasoningSignature) {
		t.Fatal("redacted reasoning must require the reasoning-signature capability")
	}
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if !bytes.Contains(encoded.Body, []byte(`"type":"redacted_thinking"`)) ||
		!bytes.Contains(encoded.Body, []byte(redactedPayload)) {
		t.Fatalf("request wire = %s", encoded.Body)
	}

	responseBody := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"redacted_thinking","data":"` + redactedPayload + `"},{"type":"text","text":"hello"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	result, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, responseBody, nil)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if !bytes.Contains(result.Body, []byte(`"type":"redacted_thinking"`)) ||
		bytes.Count(result.Body, []byte(redactedPayload)) != 1 || !bytes.Contains(result.Body, []byte("hello")) {
		t.Fatalf("response wire = %s", result.Body)
	}
}

// A redacted block is opaque continuity state with no text-only rendering, so
// neither buffered nor streaming translation can drop its payload even when
// the operator accepted the lossy reasoning-signature projection.
func TestRedactedThinkingNeverCrossesToOpenAIFormats(t *testing.T) {
	requestBody := []byte(`{"model":"m","max_tokens":64,"messages":[` +
		`{"role":"assistant","content":[{"type":"redacted_thinking","data":"` + redactedPayload + `"}]}]}`)
	responseBody := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"redacted_thinking","data":"` + redactedPayload + `"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			for _, engine := range []*Engine{NewBuiltinEngine(), signatureOptInEngine(t)} {
				_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, requestBody)
				// Decode succeeds; the capability check happens at encode.
				if err != nil {
					t.Fatalf("DecodeRequest() error = %v", err)
				}
				_, err = engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, requestBody, nil)
				assertUnsupportedCapability(t, err)
				_, err = engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, responseBody, nil)
				assertUnsupportedCapability(t, err)
			}
		})
	}
}

func assertUnsupportedCapability(t *testing.T, err error) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_capability" {
		t.Fatalf("error = %v, want unsupported_capability", err)
	}
}

// A redacted_thinking content_block_start keeps its data through a stream
// round-trip to Messages, and is refused for Chat/Responses even with the
// lossy signature opt-in.
func TestRedactedThinkingStreamRoundTripAndGate(t *testing.T) {
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"model\":\"model\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"" + redactedPayload + "\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	streamContext := llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"}
	same, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, streamContext)
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := same.Push(payload)
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	wire := bytes.Join(frames, nil)
	if !bytes.Contains(wire, []byte(`"type":"redacted_thinking"`)) || !bytes.Contains(wire, []byte(redactedPayload)) {
		t.Fatalf("stream wire = %s", wire)
	}
	if bytes.Contains(wire, []byte(`"signature_delta"`)) {
		t.Fatal("a redacted block must not emit a synthetic signature delta")
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			for _, engine := range []*Engine{NewBuiltinEngine(), signatureOptInEngine(t)} {
				stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, target, streamContext)
				if err != nil {
					t.Fatal(err)
				}
				if _, _, _, err := stream.Push(payload); err == nil {
					t.Fatal("redacted thinking crossed formats without refusal")
				}
			}
		})
	}
}

// A buffered reply decoded first and then streamed (cache replay and synthetic
// producers take this path) must keep the redacted block. neutralResponseEvents
// puts the opaque data on its own item-start event so the encoder never needs
// to stream a signature delta into a redacted block.
func TestRedactedThinkingBufferedDecodeThenStream(t *testing.T) {
	engine := NewBuiltinEngine()
	responseBody := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"redacted_thinking","data":"` + redactedPayload + `"},{"type":"text","text":"hello"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, responseBody)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	wire, _, err := engine.EncodeResponseStream(
		llmprotocol.AnthropicMessagesV1, response,
		llmprotocol.StreamContext{Context: t.Context(), PublicModel: response.Model},
	)
	if err != nil {
		t.Fatalf("EncodeResponseStream() error = %v", err)
	}
	if !bytes.Contains(wire, []byte(`"type":"redacted_thinking"`)) ||
		!bytes.Contains(wire, []byte(redactedPayload)) ||
		!bytes.Contains(wire, []byte("hello")) {
		t.Fatalf("replayed stream lost the redacted block or answer: %s", wire)
	}
	if bytes.Contains(wire, []byte(`"signature_delta"`)) {
		t.Fatal("a redacted block must not emit a synthetic signature delta: " + string(wire))
	}
	// The replayed stream decodes back to the same redacted block, not plain
	// reasoning, so the capability gate still refuses it for Chat/Responses.
	replayed, _, err := engine.DecodeResponseStream(
		llmprotocol.AnthropicMessagesV1, wire,
		llmprotocol.StreamContext{Context: t.Context(), PublicModel: response.Model},
	)
	if err != nil {
		t.Fatalf("DecodeResponseStream() error = %v\n%s", err, wire)
	}
	var redactedFound bool
	for _, output := range replayed.Output {
		for _, block := range output.Content {
			if block.Reasoning == llmprotocol.ReasoningScopeRedacted {
				redactedFound = true
				if block.Signature != redactedPayload {
					t.Fatalf("replayed redacted payload = %q, want %q", block.Signature, redactedPayload)
				}
			}
		}
	}
	if !redactedFound {
		t.Fatalf("replayed stream lost the redacted scope: %+v", replayed.Output)
	}
}

// A signed thinking block after a redacted one must not join the redacted
// group: the redacted block has no delta, so a mixed group asks the encoder to
// stream a signature into it and fails. Buffered decode then stream of this
// order is a valid Anthropic-to-Anthropic replay.
func TestRedactedThinkingFollowedBySignedThinkingReplays(t *testing.T) {
	engine := NewBuiltinEngine()
	const signedPayload = "signed-opaque-payload"
	responseBody := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"redacted_thinking","data":"` + redactedPayload + `"},` +
		`{"type":"thinking","thinking":"consider","signature":"` + signedPayload + `"},` +
		`{"type":"text","text":"answer"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, responseBody)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	wire, _, err := engine.EncodeResponseStream(
		llmprotocol.AnthropicMessagesV1, response,
		llmprotocol.StreamContext{Context: t.Context(), PublicModel: response.Model},
	)
	if err != nil {
		t.Fatalf("EncodeResponseStream() error = %v", err)
	}
	if !bytes.Contains(wire, []byte(`"type":"redacted_thinking"`)) ||
		!bytes.Contains(wire, []byte(redactedPayload)) ||
		!bytes.Contains(wire, []byte(signedPayload)) ||
		!bytes.Contains(wire, []byte("answer")) {
		t.Fatalf("replayed stream lost a block: %s", wire)
	}
	if bytes.Contains(wire, []byte(`"signature_delta":"`+redactedPayload)) {
		t.Fatal("a redacted block must not emit a synthetic signature delta: " + string(wire))
	}
	replayed, _, err := engine.DecodeResponseStream(
		llmprotocol.AnthropicMessagesV1, wire,
		llmprotocol.StreamContext{Context: t.Context(), PublicModel: response.Model},
	)
	if err != nil {
		t.Fatalf("DecodeResponseStream() error = %v\n%s", err, wire)
	}
	var redactedFound, signedFound bool
	for _, output := range replayed.Output {
		for _, block := range output.Content {
			if block.Reasoning == llmprotocol.ReasoningScopeRedacted && block.Signature == redactedPayload {
				redactedFound = true
			}
			if block.Signature == signedPayload {
				signedFound = true
			}
		}
	}
	if !redactedFound || !signedFound {
		t.Fatalf("replayed stream lost the redacted or signed block: redacted=%v signed=%v: %+v", redactedFound, signedFound, replayed.Output)
	}
}
