package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const reasoningDetailsFixture = `[{"type":"reasoning.summary","summary":"Plan","id":null,"format":"openai-responses-v1","index":0},{"type":"reasoning.text","text":"Think","signature":null,"id":"r1","format":"anthropic-claude-v1","index":1},{"type":"reasoning.encrypted","data":"opaque-data","id":"r2","format":"openai-responses-v1","index":2}]`

func TestChatReasoningDetailsSurviveResponseAndMutatedFollowUp(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"response-1","model":"provider-model","provider":"OpenRouter","choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning":"Think","reasoning_details":` + reasoningDetailsFixture + `},"finish_reason":"stop"}]}`)
	response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body)
	require.NoError(t, err)
	require.Len(t, response.Output[0].Content, 4, "plain reasoning alias must not duplicate the typed blocks")
	response.Generation++
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
	require.NoError(t, err)
	var wire struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(encoded.Body, &wire))
	assertMessageReasoningDetails(t, wire.Choices[0].Message, reasoningDetailsFixture)
	requestBody := []byte(`{"model":"public-model","messages":[{"role":"user","content":"hello"},` + string(wire.Choices[0].Message) + `,{"role":"user","content":"continue"}]}`)
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, requestBody)
	require.NoError(t, err)
	request.Model = "selected-model"
	request.Generation++
	// Serialization models retained history; an empty envelope forces real encoding.
	serialized, err := json.Marshal(request.Messages)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(serialized, &request.Messages))
	upstream, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	require.NoError(t, err)
	var followUp struct {
		Messages []json.RawMessage `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(upstream.Body, &followUp))
	assertMessageReasoningDetails(t, followUp.Messages[1], reasoningDetailsFixture)
	streamBody, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIChatV1, response, llmprotocol.StreamContext{Context: t.Context()})
	require.NoError(t, err)
	replayed, _, err := engine.DecodeResponseStream(llmprotocol.OpenAIChatV1, streamBody, llmprotocol.StreamContext{Context: t.Context()})
	require.NoError(t, err)
	require.Equal(t, response.Output[0].Content, replayed.Output[0].Content, "cached response stream must retain continuity blocks")
}

func TestChatReasoningDetailsKeepUnknownFieldsAndVariantsClosed(t *testing.T) {
	for name, details := range map[string]string{
		"unknown_field":   `[{"type":"reasoning.text","text":"ok","unknown":true}]`,
		"case_alias":      `[{"Type":"reasoning.text","text":"ok"}]`,
		"wrong_scalar":    `[{"type":"reasoning.text","text":true}]`,
		"mixed_variant":   `[{"type":"reasoning.text","text":"ok","data":"encrypted"}]`,
		"missing_summary": `[{"type":"reasoning.summary"}]`,
		"null_data":       `[{"type":"reasoning.encrypted","data":null}]`,
		"unknown_type":    `[{"type":"future"}]`,
		"negative_index":  `[{"type":"reasoning.text","index":-1}]`,
		"duplicate":       `[{"type":"reasoning.text","text":"a","text":"b"}]`,
	} {
		t.Run(name, func(t *testing.T) {
			engine := NewBuiltinEngine()
			message := `{"role":"assistant","content":"answer","reasoning_details":` + details + `}`
			_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(`{"id":"r","model":"m","choices":[{"index":0,"message":`+message+`,"finish_reason":"stop"}]}`))
			require.Error(t, err)
			_, _, _, err = engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(`{"model":"m","messages":[`+message+`]}`))
			require.Error(t, err)
			stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{Context: t.Context()})
			require.NoError(t, err)
			_, _, _, err = stream.Push([]byte(`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"reasoning_details":` + details + `},"finish_reason":null}]}` + "\n\n"))
			require.Error(t, err)
		})
	}
}

func TestChatReasoningDetailsEmptyArraysAndAnonymousBlocks(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, details := range []string{`[]`, `null`, `[{"type":"reasoning.text","text":"thought"}]`} {
		t.Run(details, func(t *testing.T) {
			_, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(`{"id":"r","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning_details":`+details+`},"finish_reason":"stop"}]}`), nil)
			require.NoError(t, err)
		})
	}
}

func TestChatReasoningDetailsRejectLossyCrossProtocolTranslation(t *testing.T) {
	body := []byte(`{"id":"r","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning_details":` + reasoningDetailsFixture + `},"finish_reason":"stop"}]}`)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			engine := NewBuiltinEngine()
			_, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
			require.ErrorContains(t, err, "reasoning_continuity")
			stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{Context: t.Context()})
			require.NoError(t, err)
			_, _, _, err = stream.Push([]byte(`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"reasoning_details":` + reasoningDetailsFixture + `},"finish_reason":null}]}` + "\n\n"))
			require.ErrorContains(t, err, "reasoning_continuity")
		})
	}
}

func assertMessageReasoningDetails(t *testing.T, message json.RawMessage, expected string) {
	t.Helper()
	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(message, &fields))
	require.JSONEq(t, expected, string(fields["reasoning_details"]))
}

func TestChatReasoningDetailsStreamRoundTrip(t *testing.T) {
	engine := NewBuiltinEngine()
	chunks := []string{
		`{"role":"assistant","reasoning_details":[{"type":"reasoning.text","text":"Think ","signature":null,"id":"r1","format":"anthropic-claude-v1","index":0}]}`,
		`{"reasoning_details":[{"type":"reasoning.text","text":"more","signature":"sig","id":"r1","format":"anthropic-claude-v1","index":0}]}`,
		`{"reasoning_details":[{"type":"reasoning.text","signature":"-tail","index":0}]}`,
		`{"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque-","id":"r2","format":"openai-responses-v1","index":1}]}`,
		`{"reasoning_details":[{"type":"reasoning.encrypted","data":"data","id":"r2","format":"openai-responses-v1","index":1}]}`,
		`{"content":"answer"}`,
	}
	var source bytes.Buffer
	for _, delta := range chunks {
		source.WriteString(`data: {"id":"response-1","model":"provider-model","choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}` + "\n\n")
	}
	source.WriteString(`data: {"id":"response-1","model":"provider-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n")
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{Context: t.Context(), PublicModel: "public-model", ProviderModel: "provider-model"})
	require.NoError(t, err)
	var public bytes.Buffer
	for _, b := range source.Bytes() {
		frames, _, _, pushErr := stream.Push([]byte{b})
		require.NoError(t, pushErr)
		public.Write(bytes.Join(frames, nil))
	}
	frames, _, _, err := stream.Finalize(nil)
	require.NoError(t, err)
	public.Write(bytes.Join(frames, nil))
	require.Equal(t, 1, bytes.Count(public.Bytes(), []byte("data: [DONE]")))
	response, _, err := engine.DecodeResponseStream(llmprotocol.OpenAIChatV1, public.Bytes(), llmprotocol.StreamContext{Context: t.Context()})
	require.NoError(t, err)
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
	require.NoError(t, err)
	var wire struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(encoded.Body, &wire))
	assertMessageReasoningDetails(t, wire.Choices[0].Message, `[{"type":"reasoning.text","text":"Think more","signature":"sig-tail","id":"r1","format":"anthropic-claude-v1","index":0},{"type":"reasoning.encrypted","data":"opaque-data","id":"r2","format":"openai-responses-v1","index":1}]`)
}

func TestChatReasoningDetailsPreserveOptionalStringStates(t *testing.T) {
	for _, details := range []string{
		`[{"type":"reasoning.text"}]`,
		`[{"type":"reasoning.text","text":null,"signature":null,"id":null,"format":null}]`,
		`[{"type":"reasoning.text","text":"","signature":"","id":"","format":""}]`,
	} {
		t.Run(details, func(t *testing.T) {
			engine := NewBuiltinEngine()
			request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(`{"model":"m","messages":[{"role":"assistant","reasoning_details":`+details+`}]}`))
			require.NoError(t, err)
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
			require.NoError(t, err)
			var wire struct {
				Messages []json.RawMessage `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(encoded.Body, &wire))
			assertMessageReasoningDetails(t, wire.Messages[0], details)
		})
	}
}

func TestChatReasoningDetailsRejectNonAssistantRole(t *testing.T) {
	for _, role := range []string{"user", "system", "developer"} {
		_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(`{"model":"m","messages":[{"role":"`+role+`","content":"hello","reasoning_details":`+reasoningDetailsFixture+`}]}`))
		require.ErrorContains(t, err, "invalid_reasoning_detail_role")
	}
}

func TestChatReasoningDetailsStreamIdentityAndLimits(t *testing.T) {
	for name, test := range map[string]struct {
		details []string
		limit   int
		code    string
	}{
		"cumulative_payload": {[]string{`[{"type":"reasoning.text","text":"abc","index":0}]`, `[{"type":"reasoning.text","text":"def","index":0}]`}, 5, "reasoning_detail_limit"},
		"changed_id":         {[]string{`[{"type":"reasoning.text","id":"a","index":0}]`, `[{"type":"reasoning.text","id":"b","index":0}]`}, 0, "reasoning_detail_identity_mismatch"},
		"changed_type":       {[]string{`[{"type":"reasoning.text","index":0}]`, `[{"type":"reasoning.encrypted","data":"x","index":0}]`}, 0, "reasoning_detail_identity_mismatch"},
		"changed_format":     {[]string{`[{"type":"reasoning.text","format":"f1","index":0}]`, `[{"type":"reasoning.text","format":"f2","index":0}]`}, 0, "reasoning_detail_identity_mismatch"},
		"conflicting_ids":    {[]string{`[{"type":"reasoning.text","id":"a","index":0},{"type":"reasoning.text","id":"b","index":1}]`, `[{"type":"reasoning.text","id":"b","index":0}]`}, 0, "reasoning_detail_identity_mismatch"},
	} {
		t.Run(name, func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			if test.limit > 0 {
				policy.Limits.TextBytes = test.limit
			}
			decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: t.Context()}, policy)
			for index, details := range test.details {
				_, _, err := decoder.Push(reasoningDetailsChunk(details))
				if index == len(test.details)-1 {
					require.ErrorContains(t, err, test.code)
				} else {
					require.NoError(t, err)
				}
			}
		})
	}
}

func TestChatReasoningDetailsKeepAnonymousBlocksSeparateAndEnrichIdentifiedBlocks(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: t.Context()}, llmprotocol.DefaultPolicy())
	var indices []int
	for _, details := range []string{
		`[{"type":"reasoning.text","text":"first"},{"type":"reasoning.text","text":"second"}]`,
		`[{"type":"reasoning.text","text":"third"}]`,
		`[{"type":"reasoning.text","text":"identified","index":0,"id":null,"format":null}]`,
		`[{"type":"reasoning.text","signature":"sig","index":0,"id":"r","format":"f"}]`,
		`[{"type":"reasoning.text","text":"tail","id":"r"}]`,
	} {
		events, _, err := decoder.Push(reasoningDetailsChunk(details))
		require.NoError(t, err)
		for _, event := range events {
			if event.Type == llmprotocol.EventReasoningDelta {
				indices = append(indices, event.ContentIndex)
			}
		}
	}
	// Only explicit identifiers justify merging fragments into one block.
	require.Equal(t, []int{0, 1, 2, 3, 3, 3}, indices)
}

func reasoningDetailsChunk(details string) []byte {
	return []byte(`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"reasoning_details":` + details + `},"finish_reason":null}]}` + "\n\n")
}

func TestChatReasoningDetailsDirectEncodersRejectLossyTranslation(t *testing.T) {
	engine := NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(`{"model":"m","messages":[{"role":"assistant","reasoning_details":`+reasoningDetailsFixture+`}]}`))
	require.NoError(t, err)
	response := llmprotocol.Response{Generation: 1, Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: request.Messages[0].Content}}}
	for _, codec := range []Codec{OpenAIResponsesCodec{}, AnthropicMessagesCodec{}} {
		_, _, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
		require.ErrorContains(t, err, "reasoning_continuity")
		_, _, err = codec.EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
		require.ErrorContains(t, err, "reasoning_continuity")
		encoder := codec.(StreamCodec).NewEncoder(llmprotocol.StreamContext{Context: t.Context()}, llmprotocol.DefaultPolicy())
		_, _, err = encoder.Push(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, Content: &request.Messages[0].Content[0]})
		require.ErrorContains(t, err, "reasoning_continuity")
	}
}
