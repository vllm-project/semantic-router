package contextdedup

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestProtocolRoundTripsRemoveOnlyTheLaterCopy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{"chat", llmprotocol.OpenAIChatV1, `{"model":"model","messages":[{"role":"developer","content":"Keep permissions"},{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"live"}]}`},
		{"responses", llmprotocol.OpenAIResponsesV1, `{"model":"model","instructions":"Keep permissions","input":[{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"live"}]}`},
		{"anthropic", llmprotocol.AnthropicMessagesV1, `{"model":"model","max_tokens":100,"system":"Keep permissions","messages":[{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"live"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(tc.format, []byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			baseline, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			if err = ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{{Kind: contextcompression.TransformDeduplicate}}); err != nil {
				t.Fatal(err)
			}
			unchanged, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(baseline.Body, unchanged.Body) {
				t.Fatal("disabled step changed provider bytes")
			}

			request, envelope, _, err = engine.DecodeRequest(tc.format, []byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			ir = contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			action := NewAction(testPolicy(), "").WithResolver(RequestResolver(ir))
			if err = ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
				t.Fatal(err)
			}
			result, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			decoded, _, _, err := engine.DecodeRequest(tc.format, result.Body)
			if err != nil {
				t.Fatal(err)
			}
			if len(decoded.Messages) != 3 || decoded.Messages[0].Content[0].Text != "old" ||
				decoded.Messages[1].Content[0].Text != "answer" || decoded.Messages[2].Content[0].Text != "live" {
				t.Fatalf("wrong conversation after round trip: %+v", decoded.Messages)
			}
			if encoded(t, request.Instructions) != encoded(t, decoded.Instructions) {
				t.Fatal("instructions changed")
			}
			if diagnostics := action.Reconcile(ir.Transformations.Receipts()); diagnostics.Outcome != OutcomeApplied || diagnostics.RemovedMessages != 2 {
				t.Fatalf("unexpected diagnostics %+v", diagnostics)
			}
		})
	}
}

func TestProtocolResponsesItemIDsGovernProof(t *testing.T) {
	format := llmprotocol.OpenAIResponsesV1
	for _, tc := range []struct {
		name string
		body string
		want int
	}{
		{"later_copy_without_ids", `{"model":"model","input":[{"type":"message","id":"msg_1","role":"user","content":[{"type":"input_text","text":"old"}]},{"type":"message","id":"msg_2","role":"assistant","content":[{"type":"output_text","text":"answer"}]},{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":"live"}]}`, 3},
		{"distinct_ids", `{"model":"model","input":[{"type":"message","id":"msg_1","role":"user","content":[{"type":"input_text","text":"old"}]},{"type":"message","id":"msg_2","role":"assistant","content":[{"type":"output_text","text":"answer"}]},{"type":"message","id":"msg_3","role":"user","content":[{"type":"input_text","text":"old"}]},{"type":"message","id":"msg_4","role":"assistant","content":[{"type":"output_text","text":"answer"}]},{"role":"user","content":"live"}]}`, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(format, []byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			action := NewAction(testPolicy(), "").WithResolver(RequestResolver(ir))
			if err = ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
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
			if len(decoded.Messages) != tc.want {
				t.Fatalf("expected %d messages, got %+v", tc.want, decoded.Messages)
			}
			if tc.want == 3 && (decoded.Messages[0].ID != "msg_1" || decoded.Messages[1].ID != "msg_2") {
				t.Fatalf("the earlier copy with ids must be retained: %+v", decoded.Messages)
			}
		})
	}
}

func TestProtocolToolExchangesSurviveEveryFormat(t *testing.T) {
	for _, tc := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{"chat", llmprotocol.OpenAIChatV1, `{"model":"model","messages":[{"role":"user","content":"run"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"},{"role":"assistant","content":"done"},{"role":"user","content":"run"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_2","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_2","content":"ok"},{"role":"assistant","content":"done"},{"role":"user","content":"live"}]}`},
		{"anthropic", llmprotocol.AnthropicMessagesV1, `{"model":"model","max_tokens":100,"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]},{"role":"assistant","content":"done"},{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_2","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_2","content":"ok"}]},{"role":"assistant","content":"done"},{"role":"user","content":"live"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(tc.format, []byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			baseline, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			ir := contextcompression.ParseSemanticRequest(&request, contextcompression.Provenance{})
			action := NewAction(testPolicy(), "").WithResolver(RequestResolver(ir))
			if err = ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
				t.Fatal(err)
			}
			result, err := engine.EncodeRequest(tc.format, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(baseline.Body, result.Body) {
				t.Fatalf("repeated tool executions must survive: %s", result.Body)
			}
			diagnostics := action.Reconcile(ir.Transformations.Receipts())
			if diagnostics.Outcome != OutcomeSkipped || diagnostics.Retained[RetainedToolExchange] != 2 {
				t.Fatalf("unexpected diagnostics %+v", diagnostics)
			}
		})
	}
}
