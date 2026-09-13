package historyreset

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Each supported ingress format decodes into the same neutral request, so the
// action runs once and every protocol keeps its own semantics on the way out.
// The assertions below check the encoded provider request, not just the
// in-memory view, because that is what the backend actually receives.

func TestResetPreservesProtocolSemantics(t *testing.T) {
	for _, test := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{
			"chat",
			llmprotocol.OpenAIChatV1,
			`{"model":"model","messages":[` +
				`{"role":"developer","content":"Keep permissions"},` +
				`{"role":"user","content":"old question"},` +
				`{"role":"assistant","content":"old answer"},` +
				`{"role":"user","content":"live question"}]}`,
		},
		{
			"responses",
			llmprotocol.OpenAIResponsesV1,
			`{"model":"model","instructions":"Keep permissions","input":[` +
				`{"role":"user","content":"old question"},` +
				`{"role":"assistant","content":"old answer"},` +
				`{"role":"user","content":"live question"}]}`,
		},
		{
			"anthropic",
			llmprotocol.AnthropicMessagesV1,
			`{"model":"model","max_tokens":100,"system":"Keep permissions","messages":[` +
				`{"role":"user","content":"old question"},` +
				`{"role":"assistant","content":"old answer"},` +
				`{"role":"user","content":"live question"}]}`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(test.format, []byte(test.body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}

			action := NewAction(testPolicy(), acceptedChange(), "")
			ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			if err = ir.ApplySteps(
				context.Background(),
				[]contextcompression.TransformationStep{action.Step()},
			); err != nil {
				t.Fatalf("apply: %v", err)
			}

			encoded, err := engine.EncodeRequest(test.format, request, envelope)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			body := string(encoded.Body)
			if strings.Contains(body, "old question") || strings.Contains(body, "old answer") {
				t.Fatalf("the prior turn survived encoding: %s", body)
			}
			if !strings.Contains(body, "live question") {
				t.Fatalf("the live turn was lost: %s", body)
			}
			if !strings.Contains(body, "Keep permissions") {
				t.Fatalf("instructions were lost: %s", body)
			}

			decoded, _, _, err := engine.DecodeRequest(test.format, encoded.Body)
			if err != nil {
				t.Fatalf("re-decode: %v", err)
			}
			if len(decoded.Messages) != 1 || decoded.Messages[0].Role != llmprotocol.RoleUser {
				t.Fatalf("unexpected surviving messages %+v", decoded.Messages)
			}
			diagnostics := action.Reconcile(ir.Transformations.Receipts())
			if diagnostics.Outcome != OutcomeApplied || diagnostics.RemovedMessages != 2 {
				t.Fatalf("unexpected diagnostics %+v", diagnostics)
			}
		})
	}
}

// A tool exchange that the live turn still depends on pins its whole group on
// every protocol, including Anthropic's user-role tool results.
func TestResetKeepsProtocolToolLinksValid(t *testing.T) {
	for _, test := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{
			"chat",
			llmprotocol.OpenAIChatV1,
			`{"model":"model","messages":[` +
				`{"role":"user","content":"old question"},` +
				`{"role":"assistant","content":"old answer"},` +
				`{"role":"user","content":"live question"},` +
				`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",` +
				`"function":{"name":"lookup","arguments":"{}"}}]},` +
				`{"role":"tool","tool_call_id":"call_1","content":"tool output"}]}`,
		},
		{
			"anthropic",
			llmprotocol.AnthropicMessagesV1,
			`{"model":"model","max_tokens":100,"messages":[` +
				`{"role":"user","content":"old question"},` +
				`{"role":"assistant","content":"old answer"},` +
				`{"role":"user","content":"live question"},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"tool output"}]}]}`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(test.format, []byte(test.body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			action := NewAction(testPolicy(), acceptedChange(), "")
			ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			if err = ir.ApplySteps(
				context.Background(),
				[]contextcompression.TransformationStep{action.Step()},
			); err != nil {
				t.Fatalf("apply: %v", err)
			}
			encoded, err := engine.EncodeRequest(test.format, request, envelope)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			body := string(encoded.Body)
			if strings.Contains(body, "old question") {
				t.Fatalf("the removable prior turn survived: %s", body)
			}
			if !strings.Contains(body, "call_1") || !strings.Contains(body, "tool output") {
				t.Fatalf("the live tool exchange was broken: %s", body)
			}
			if _, _, _, err = engine.DecodeRequest(test.format, encoded.Body); err != nil {
				t.Fatalf("the encoded request no longer decodes: %v", err)
			}
		})
	}
}

// Multimodal content is protected, so a turn carrying an image keeps its whole
// group even when the rest of it would otherwise be removable.
func TestResetKeepsMultimodalTurnsOnEveryProtocol(t *testing.T) {
	body := `{"model":"model","messages":[` +
		`{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/a.png"}}]},` +
		`{"role":"assistant","content":"about the image"},` +
		`{"role":"user","content":"live question"}]}`
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	action := NewAction(testPolicy(), acceptedChange(), "")
	ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
	if err = ir.ApplySteps(
		context.Background(),
		[]contextcompression.TransformationStep{action.Step()},
	); err != nil {
		t.Fatalf("apply: %v", err)
	}
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !strings.Contains(string(encoded.Body), "a.png") {
		t.Fatalf("protected multimodal content was removed: %s", encoded.Body)
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonNoEligibleHistory {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}
