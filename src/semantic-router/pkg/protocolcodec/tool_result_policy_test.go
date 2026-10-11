package protocolcodec

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func toolErrorHistory(flag string) []byte {
	return []byte(fmt.Sprintf(`{"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"Read the record"},
		{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"read_record","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"[]"%s}]}]}`, flag))
}

func TestToolResultErrorTranslationPolicy(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, lossy := range []llmprotocol.LossyPolicy{llmprotocol.LossyReject, llmprotocol.LossyAllowWithDiagnostic} {
			t.Run(string(target)+"/"+string(lossy), func(t *testing.T) {
				policy := llmprotocol.DefaultPolicy()
				policy.LossyFeatures = lossy
				engine, err := NewEngine(NewBuiltinRegistry(), policy)
				if err != nil {
					t.Fatal(err)
				}
				absent, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, toolErrorHistory(""), nil)
				if err != nil || len(absent.Diagnostics) != 0 {
					t.Fatalf("absent flag: diagnostics=%+v error=%v", absent.Diagnostics, err)
				}
				success, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, toolErrorHistory(`,"is_error":false`), nil)
				if err != nil || len(success.Diagnostics) != 0 || !bytes.Equal(success.Body, absent.Body) {
					t.Fatalf("false flag changed successful history: diagnostics=%+v error=%v", success.Diagnostics, err)
				}
				failed, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, toolErrorHistory(`,"is_error":true`), nil)
				if lossy == llmprotocol.LossyReject {
					assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "lossy_translation")
					if len(failed.Body) != 0 {
						t.Fatalf("strict translation emitted provider body: %s", failed.Body)
					}
					return
				}
				if err != nil || !bytes.Equal(failed.Body, success.Body) {
					t.Fatalf("permissive translation changed result payload: %s, error=%v", failed.Body, err)
				}
				if len(failed.Diagnostics) != 1 {
					t.Fatalf("want one loss diagnostic, got %+v", failed.Diagnostics)
				}
				diagnostic := failed.Diagnostics[0]
				if diagnostic.Source != llmprotocol.AnthropicMessagesV1 || diagnostic.Target != target ||
					diagnostic.Field != "tool_result.is_error" || diagnostic.Action != llmprotocol.DiagnosticApproximated || diagnostic.Reason == "" {
					t.Fatalf("incorrect loss diagnostic: %+v", diagnostic)
				}
				flag := failed.Request.Messages[len(failed.Request.Messages)-1].Content[0].ToolResult.IsError
				if flag == nil || !*flag {
					t.Fatal("encoding mutated the neutral failure flag")
				}
			})
		}
	}
}

func TestToolResultErrorAnthropicPreservation(t *testing.T) {
	for _, lossy := range []llmprotocol.LossyPolicy{llmprotocol.LossyReject, llmprotocol.LossyAllowWithDiagnostic} {
		for _, preservation := range []llmprotocol.SourcePreservationPolicy{llmprotocol.SourceDisabled, llmprotocol.SourceBoundedSameFormat} {
			policy := llmprotocol.DefaultPolicy()
			policy.LossyFeatures, policy.SourcePreservation = lossy, preservation
			engine, err := NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			for _, mutate := range []RequestMutation{nil, func(request *llmprotocol.Request) error { request.Model = "selected"; return nil }} {
				result, translateErr := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, toolErrorHistory(`,"is_error":true`), mutate)
				if translateErr != nil || len(result.Diagnostics) != 0 || !bytes.Contains(result.Body, []byte(`"is_error":true`)) {
					t.Fatalf("Anthropic flag not preserved: body=%s diagnostics=%+v error=%v", result.Body, result.Diagnostics, translateErr)
				}
			}
		}
	}
}

func TestToolResultErrorNeutralMutation(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		valid, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, format, toolErrorHistory(""), nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = engine.TranslateRequest(format, format, valid.Body, func(request *llmprotocol.Request) error {
			for _, message := range request.Messages {
				for _, content := range message.Content {
					if content.ToolResult != nil {
						flag := true
						content.ToolResult.IsError = &flag
					}
				}
			}
			return nil
		})
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "lossy_translation")
	}
}

func TestToolResultErrorEncodeRequestBoundedDiagnostics(t *testing.T) {
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, toolErrorHistory(`,"is_error":true`))
	if err != nil {
		t.Fatal(err)
	}
	// Cover the neutral dispatch API with multiple failures, including an empty
	// result. The loss must be detected independently of result text or position.
	request.Messages = append(request.Messages, request.Messages[1], request.Messages[2])
	secondCall := *request.Messages[3].Content[0].ToolCall
	secondCall.ID = "call_2"
	request.Messages[3].Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &secondCall}}
	secondResult := *request.Messages[4].Content[0].ToolResult
	secondResult.CallID, secondResult.Content = "call_2", nil
	request.Messages[4].Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &secondResult}}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		_, strictErr := NewBuiltinEngine().EncodeRequest(target, request, llmprotocol.Envelope{})
		assertProtocolError(t, strictErr, llmprotocol.ErrorUnsupportedFeature, "lossy_translation")
		policy := llmprotocol.DefaultPolicy()
		policy.LossyFeatures = llmprotocol.LossyAllowWithDiagnostic
		policy.Limits.Diagnostics = 1
		engine, engineErr := NewEngine(NewBuiltinRegistry(), policy)
		if engineErr != nil {
			t.Fatal(engineErr)
		}
		for range 2 {
			result, encodeErr := engine.EncodeRequest(target, request, llmprotocol.Envelope{})
			if encodeErr != nil || len(result.Diagnostics) != 1 || result.Diagnostics[0].Field != "tool_result.is_error" {
				t.Fatalf("want one bounded diagnostic per encoding: diagnostics=%+v error=%v", result.Diagnostics, encodeErr)
			}
		}
	}
}
