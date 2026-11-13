package protocolcodec

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const chatAudioOutputRequestFixture = `{"model":"m","messages":[{"role":"user","content":"say hi"}],"modalities":["text","audio"],"audio":{"voice":"alloy","format":"wav"}}`

const chatAudioOutputResponseFixture = `{"id":"chatcmpl_audio","object":"chat.completion","created":100,"model":"provider-model","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","audio":{"id":"audio_abc","data":"UklGRg==","expires_at":1760000000,"transcript":"hi"}}}]}`

func TestChatAudioUsesNeutralMIMEAndPreservesWireFormat(t *testing.T) {
	for _, tc := range []struct{ format, media string }{{"wav", "audio/wav"}, {"mp3", "audio/mpeg"}} {
		t.Run(tc.format, func(t *testing.T) {
			engine := NewBuiltinEngine()
			body := []byte(fmt.Sprintf(`{"model":"source","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":%q}}]}]}`, tc.format))
			request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
			if err != nil {
				t.Fatal(err)
			}
			audio := request.Messages[0].Content[0]
			if audio.MediaType != tc.media || audio.Data != "UklGRg==" {
				t.Fatalf("neutral audio: %+v", audio)
			}
			request.Model, request.Generation = "selected", request.Generation+1
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			var wire chatRequestWire
			if err := json.Unmarshal(encoded.Body, &wire); err != nil {
				t.Fatal(err)
			}
			var parts []chatContentWire
			if err := json.Unmarshal(wire.Messages[0].Content, &parts); err != nil {
				t.Fatal(err)
			}
			if parts[0].InputAudio.Format != tc.format || parts[0].InputAudio.Data != "UklGRg==" {
				t.Fatalf("wire audio: %+v", parts[0].InputAudio)
			}
		})
	}
}

func TestChatAudioOutputRequest(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name string
		body string
		want *llmprotocol.AudioOutputOptions
		code string
	}{
		{name: "options", body: chatAudioOutputRequestFixture, want: &llmprotocol.AudioOutputOptions{Voice: "alloy", Format: "wav"}},
		{name: "modality only", body: `{"model":"m","messages":[{"role":"user","content":"hi"}],"modalities":["text","audio"]}`, want: &llmprotocol.AudioOutputOptions{}},
		{name: "audio without modality", body: `{"model":"m","messages":[{"role":"user","content":"hi"}],"audio":{"voice":"alloy"}}`, code: "audio_requires_modality"},
		{name: "text only with audio options", body: `{"model":"m","messages":[{"role":"user","content":"hi"}],"modalities":["text"],"audio":{"voice":"alloy"}}`, code: "audio_requires_modality"},
		{name: "unknown modality", body: `{"model":"m","messages":[{"role":"user","content":"hi"}],"modalities":["text","video"]}`, code: "invalid_modalities"},
		{name: "modalities is not an array", body: `{"model":"m","messages":[{"role":"user","content":"hi"}],"modalities":"text"}`, code: "invalid_modalities"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(test.body))
			if test.code != "" {
				requireProtocolErrorCode(t, err, test.code)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.want, request.AudioOutput)
			require.True(t, llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityAudioOutput))
		})
	}

	for _, fixture := range []string{
		chatAudioOutputRequestFixture,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"modalities":["text","audio"]}`,
	} {
		t.Run("re-encode", func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(fixture))
			require.NoError(t, err)
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
			require.NoError(t, err)
			require.JSONEq(t, fixture, string(encoded.Body))
		})
	}
}

func TestChatAudioOutputResponse(t *testing.T) {
	engine := NewBuiltinEngine()
	response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(chatAudioOutputResponseFixture))
	require.NoError(t, err)
	require.Len(t, response.Output, 1)
	require.Len(t, response.Output[0].Content, 2)
	require.Equal(t, llmprotocol.ContentText, response.Output[0].Content[0].Kind)
	require.Equal(t, "hi", response.Output[0].Content[0].Text)
	audio := response.Output[0].Content[1]
	require.Equal(t, llmprotocol.ContentAudio, audio.Kind)
	require.Equal(t, &llmprotocol.GeneratedAudio{
		ID: "audio_abc", Data: "UklGRg==", ExpiresAt: 1760000000, Transcript: "hi",
	}, audio.GeneratedAudio)

	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
	require.NoError(t, err)
	require.JSONEq(t, chatAudioOutputResponseFixture, string(encoded.Body))
}

func TestChatAudioOutputIsRefusedAcrossFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target)+"/request", func(t *testing.T) {
			_, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, target, []byte(chatAudioOutputRequestFixture), nil)
			requireProtocolErrorCode(t, err, "unsupported_capability")
			require.Contains(t, err.Error(), "audio_output")
		})
		t.Run(string(target)+"/response", func(t *testing.T) {
			_, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, []byte(chatAudioOutputResponseFixture), nil)
			requireProtocolErrorCode(t, err, "unsupported_capability")
			require.Contains(t, err.Error(), "audio_output")
		})
	}
}

func TestChatAudioOutputStreamDeltaRemainsUnsupported(t *testing.T) {
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "m", ProviderModel: "m"},
	)
	require.NoError(t, err)
	chunk := `data: {"id":"chatcmpl_audio","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"audio":{"id":"audio_abc","data":"UklGRg==","expires_at":1760000000,"transcript":"hi"}}}]}` + "\n\n"
	_, _, _, err = stream.Push([]byte(chunk))
	requireProtocolErrorCode(t, err, "unsupported_upstream_audio")
}

func TestChatAudioOutputMessageReference(t *testing.T) {
	engine := NewBuiltinEngine()
	// An assistant turn references a previous answer by id alone; re-encoding it
	// must not invent empty data, expiry or transcript.
	fixture := `{"model":"m","messages":[{"role":"user","content":"say hi"},{"role":"assistant","content":"hi","audio":{"id":"audio_abc"}}]}`
	request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(fixture))
	require.NoError(t, err)
	require.Len(t, request.Messages, 2)
	var audio *llmprotocol.GeneratedAudio
	for _, content := range request.Messages[1].Content {
		if content.Kind == llmprotocol.ContentAudio {
			audio = content.GeneratedAudio
		}
	}
	require.Equal(t, &llmprotocol.GeneratedAudio{ID: "audio_abc"}, audio)

	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	require.NoError(t, err)
	require.JSONEq(t, fixture, string(encoded.Body))
}

func TestChatAudioOutputMessageRoleAndArtifactGates(t *testing.T) {
	engine := NewBuiltinEngine()
	t.Run("audio on a user turn", func(t *testing.T) {
		body := `{"model":"m","messages":[{"role":"user","content":"hi","audio":{"id":"audio_abc"}}]}`
		_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
		requireProtocolErrorCode(t, err, "audio_requires_assistant_role")
	})
	t.Run("empty response artifact", func(t *testing.T) {
		body := `{"id":"chatcmpl_a","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","audio":{}}}]}`
		_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(body))
		requireProtocolErrorCode(t, err, "invalid_audio_output")
	})
	t.Run("two artifacts in one message", func(t *testing.T) {
		response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(chatAudioOutputResponseFixture))
		require.NoError(t, err)
		response.Output[0].Content = append(response.Output[0].Content,
			llmprotocol.Content{Kind: llmprotocol.ContentAudio, GeneratedAudio: &llmprotocol.GeneratedAudio{ID: "audio_second"}})
		_, err = engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
		requireProtocolErrorCode(t, err, "multiple_audio_outputs")
	})
	t.Run("assistant input-audio part", func(t *testing.T) {
		response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(chatAudioOutputResponseFixture))
		require.NoError(t, err)
		for i := range response.Output[0].Content {
			if response.Output[0].Content[i].Kind != llmprotocol.ContentAudio {
				continue
			}
			response.Output[0].Content[i].GeneratedAudio = nil
			response.Output[0].Content[i].Data, response.Output[0].Content[i].MediaType = "UklGRg==", "audio/wav"
		}
		_, err = engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
		requireProtocolErrorCode(t, err, "audio_requires_generated_artifact")
	})
}

func TestChatAudioOutputWireShapes(t *testing.T) {
	engine := NewBuiltinEngine()
	const base = `{"model":"m","messages":[{"role":"user","content":"hi"}],`
	for _, test := range []struct{ name, body, code string }{
		{
			name: "audio-only modality is refused, not rewritten",
			body: base + `"modalities":["audio"]}`, code: "invalid_modalities",
		},
		{
			name: "custom voice reference",
			body: base + `"modalities":["text","audio"],"audio":{"voice":{"id":"voice_1"}}}`, code: "unsupported_custom_voice",
		},
		{
			name: "empty assistant reference",
			body: `{"model":"m","messages":[{"role":"assistant","content":"hi","audio":{}}]}`, code: "audio_reference_requires_id",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(test.body))
			requireProtocolErrorCode(t, err, test.code)
		})
	}

	t.Run("data-only artifact round-trip", func(t *testing.T) {
		fixture := `{"id":"chatcmpl_a","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","audio":{"data":"UklGRg=="}}}]}`
		response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(fixture))
		require.NoError(t, err)
		encoded, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
		require.NoError(t, err)
		require.JSONEq(t, fixture, string(encoded.Body))
	})

	t.Run("malformed upstream payload is an upstream fault", func(t *testing.T) {
		fixture := `{"id":"chatcmpl_a","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","audio":{"id":"audio_1","data":"not-base64"}}}]}`
		_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(fixture))
		requireProtocolErrorCode(t, err, "invalid_audio_output")
		var protocolError *llmprotocol.ProtocolError
		require.ErrorAs(t, err, &protocolError)
		require.Equal(t, llmprotocol.ErrorUpstreamUnavailable, protocolError.Category)
	})
}
