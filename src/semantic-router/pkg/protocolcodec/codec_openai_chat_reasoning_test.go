package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIChatReasoningControlsRoundTripThroughNeutralIR(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()
	request, envelope, _, err := codec.DecodeRequest([]byte(`{
		"model":"reasoning-model",
		"messages":[{"role":"user","content":"solve this"}],
		"reasoning_effort":"high",
		"reasoning_budget_tokens":512
	}`), policy)
	if err != nil {
		t.Fatalf("decode reasoning request: %v", err)
	}
	if request.ReasoningEffort != "high" || request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != 512 {
		t.Fatalf("neutral reasoning controls = effort %q budget %v", request.ReasoningEffort, request.ReasoningBudgetTokens)
	}

	request.Generation++
	body, _, err := codec.EncodeRequest(request, envelope, policy)
	if err != nil {
		t.Fatalf("encode reasoning request: %v", err)
	}
	var wire map[string]interface{}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded wire: %v", err)
	}
	if wire["reasoning_effort"] != "high" || wire["reasoning_budget_tokens"] != float64(512) {
		t.Fatalf("encoded reasoning controls = %#v", wire)
	}
}

func TestOpenAIChatEncodesRouterSelectedReasoningEffort(t *testing.T) {
	request := llmprotocol.Request{
		Generation:      2,
		Model:           "reasoning-model",
		ReasoningEffort: "medium",
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText,
				Text: "solve this",
			}},
		}},
	}
	body, _, err := (OpenAIChatCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode neutral reasoning request: %v", err)
	}
	var wire struct {
		ReasoningEffort string `json:"reasoning_effort"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded wire: %v", err)
	}
	if wire.ReasoningEffort != "medium" {
		t.Fatalf("reasoning_effort = %q", wire.ReasoningEffort)
	}
}

func TestOpenAIChatDecodesSupportedReasoningAliases(t *testing.T) {
	for _, field := range []string{"reasoning_content", "reasoning"} {
		t.Run(field, func(t *testing.T) {
			body := []byte(`{"id":"response-1","model":"model-a","choices":[{"index":0,"message":{"role":"assistant","content":"answer","` + field + `":"analysis"},"finish_reason":"stop"}]}`)
			response, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatalf("decode response: %v", err)
			}
			var reasoning string
			for _, content := range response.Output[0].Content {
				if content.Kind == llmprotocol.ContentReasoning {
					reasoning += content.Text
				}
			}
			if reasoning != "analysis" {
				t.Fatalf("reasoning = %q", reasoning)
			}
		})
	}
}

func TestOpenAIChatStrictDecoderAcceptsClosedLogprobEvidence(t *testing.T) {
	body := []byte(`{
		"id":"response-1",
		"model":"model-a",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"answer"},
			"finish_reason":"stop",
			"logprobs":{"content":[{
				"token":"answer",
				"bytes":[97],
				"logprob":-0.1,
				"top_logprobs":[{"token":"answer","bytes":[97],"logprob":-0.1}]
			}]}
		}]
	}`)
	response, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("decode response with logprob evidence: %v", err)
	}
	if len(response.Evidence.TokenLogprobs) != 1 ||
		len(response.Evidence.TokenLogprobs[0].Alternatives) != 1 {
		t.Fatalf("neutral logprob evidence = %+v", response.Evidence.TokenLogprobs)
	}
}
