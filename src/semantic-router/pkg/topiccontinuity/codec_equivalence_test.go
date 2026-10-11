package topiccontinuity_test

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

type wireTurn struct {
	role string
	text string
}

func chatBody(turns []wireTurn) []byte {
	messages := make([]map[string]any, 0, len(turns))
	for _, turn := range turns {
		messages = append(messages, map[string]any{"role": turn.role, "content": turn.text})
	}
	return mustJSON(map[string]any{"model": "m", "messages": messages})
}

func responsesBody(turns []wireTurn) []byte {
	input := make([]map[string]any, 0, len(turns))
	for _, turn := range turns {
		kind := "input_text"
		if turn.role == "assistant" {
			kind = "output_text"
		}
		input = append(input, map[string]any{
			"type": "message", "role": turn.role,
			"content": []map[string]any{{"type": kind, "text": turn.text}},
		})
	}
	return mustJSON(map[string]any{"model": "m", "input": input})
}

func anthropicBody(turns []wireTurn) []byte {
	messages := make([]map[string]any, 0, len(turns))
	for _, turn := range turns {
		messages = append(messages, map[string]any{
			"role":    turn.role,
			"content": []map[string]any{{"type": "text", "text": turn.text}},
		})
	}
	return mustJSON(map[string]any{"model": "m", "max_tokens": 32, "messages": messages})
}

func mustJSON(value any) []byte {
	body, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return body
}

// TestCodecEquivalence decodes the same inline conversation through the Chat,
// Responses, and Anthropic codecs and requires identical results.
func TestCodecEquivalence(t *testing.T) {
	conversations := map[string][]wireTurn{
		"continuation": {
			{"user", "There is a crash in auth.ts inside validateToken"},
			{"assistant", "The crash comes from validateToken reading a missing header."},
			{"user", "Add a unit test for validateToken in auth.ts"},
		},
		"explicit change": {
			{"user", "Refactor the routing module so plugins load lazily"},
			{"assistant", "Done. The loader now defers plugin initialization."},
			{"user", "Unrelated question: how do I renew a passport?"},
		},
		"disjoint change": {
			{"user", "Refactor the routing module so plugins load lazily"},
			{"assistant", "Done. The loader now defers plugin initialization."},
			{"user", "What is the capital city of Peru and its current population?"},
		},
	}
	engine := protocolcodec.NewBuiltinEngine()
	policy := topiccontinuity.HistoryPolicy{
		Limits:           topiccontinuity.Limits{MaxPriorTurns: 8, MaxTurnBytes: 16384, MaxInputBytes: 147456},
		IncludeAssistant: true,
	}
	cfg := topiccontinuity.EvalConfig{Name: "topic_boundary", Policy: policy, Continuation: 0.35, Change: 0.08}
	for name, turns := range conversations {
		t.Run(name, func(t *testing.T) {
			bodies := map[llmprotocol.WireFormat][]byte{
				llmprotocol.OpenAIChatV1:        chatBody(turns),
				llmprotocol.OpenAIResponsesV1:   responsesBody(turns),
				llmprotocol.AnthropicMessagesV1: anthropicBody(turns),
			}
			var reference *topiccontinuity.Result
			for format, body := range bodies {
				request, _, _, err := engine.DecodeRequest(format, body)
				if err != nil {
					t.Fatalf("%s decode: %v", format, err)
				}
				load := func() ([]llmprotocol.Message, bool) { return request.Messages, true }
				result := topiccontinuity.EvaluateAll(context.Background(), load,
					[]topiccontinuity.EvalConfig{cfg}).Rules[0].Result
				if reference == nil {
					reference = &result
					continue
				}
				if !reflect.DeepEqual(*reference, result) {
					t.Fatalf("%s differs:\n got %+v\nwant %+v", format, result, *reference)
				}
			}
		})
	}
}
