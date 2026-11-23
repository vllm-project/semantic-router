package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const speechBackendModel = "fishaudio/s2-pro"

// decodeJSONForCompare keeps numbers as their literal text, so a large seed
// that float64 would round still compares exactly.
func decodeJSONForCompare(t *testing.T, body []byte) map[string]any {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var value map[string]any
	if err := decoder.Decode(&value); err != nil {
		t.Fatalf("decode %s: %v", body, err)
	}
	return value
}

func speechRoundTrip(t *testing.T, body string) (llmprotocol.Request, []byte) {
	t.Helper()
	codec := SpeechCodec{}
	request, _, _, err := codec.DecodeRequest([]byte(body), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeRequest(%s): %v", body, err)
	}
	request.Model = speechBackendModel
	encoded, _, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("EncodeRequest(%s): %v", body, err)
	}
	return request, encoded
}

func assertOnlyModelChanged(t *testing.T, original string, encoded []byte) {
	t.Helper()
	want := decodeJSONForCompare(t, []byte(original))
	want["model"] = speechBackendModel
	got := decodeJSONForCompare(t, encoded)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("encoded body changed more than model:\n got  %s\n from %s", encoded, original)
	}
}

func assertProtocolErrorCategory(t *testing.T, err error, want llmprotocol.ErrorCategory) {
	t.Helper()
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) {
		t.Fatalf("error = %v, want a protocol error with category %q", err, want)
	}
	if protocolErr.Category != want {
		t.Fatalf("category = %q (%v), want %q", protocolErr.Category, err, want)
	}
}

func TestSpeechCodecRoundTripPreservesEveryFieldPerBackendStyle(t *testing.T) {
	cases := map[string]string{
		"openai speech fields":              `{"model":"tts-1","input":"Hello there","voice":"alloy","instructions":"calm","response_format":"mp3","speed":1.25}`,
		"fish or cosyvoice voice clone":     `{"model":"fish","input":"Clone this voice","ref_audio":"data:audio/wav;base64,UklGRg==","ref_text":"reference transcript","max_new_tokens":2048}`,
		"qwen3 custom voice":                `{"input":"你好","voice":"vivian","task_type":"CustomVoice","language":"Chinese","instructions":"cheerful","seed":42}`,
		"qwen3 voice design":                `{"input":"Design a voice","task_type":"VoiceDesign","instructions":"a deep, slow narrator","non_streaming_mode":true}`,
		"qwen3 base with speaker embedding": `{"input":"Base clone","task_type":"Base","speaker_embedding":[0.1,-0.25,3e-4],"x_vector_only_mode":true}`,
		"voice as an object":                `{"input":"Object voice","voice":{"id":"voice_123"}}`,
		"speaker alias":                     `{"input":"Alias","speaker":"ryan"}`,
		"reference audio list":              `{"input":"Two refs","ref_audio":["a.wav","b.wav"],"ref_audio_2":"c.wav"}`,
		"sound effect fields":               `{"input":"waves","ambient_sound":"ocean waves","duration_seconds":4.5}`,
		"extension object":                  `{"input":"Extras","extra_params":{"temperature":0.7,"nested":{"top_k":[1,2]}}}`,
		"large seed keeps every digit":      `{"input":"Seed","seed":9007199254740993}`,
		"output controls":                   `{"input":"Controls","sample_rate":24000,"initial_codec_chunk_frames":0,"word_timestamps":true,"stream":false}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, encoded := speechRoundTrip(t, body)
			assertOnlyModelChanged(t, body, encoded)
		})
	}
}

func TestSpeechCodecRoundTripEachResponseFormat(t *testing.T) {
	for _, format := range []string{"wav", "pcm", "flac", "mp3", "opus"} {
		t.Run(format, func(t *testing.T) {
			body := `{"input":"Format check","voice":"vivian","response_format":"` + format + `"}`
			_, encoded := speechRoundTrip(t, body)
			assertOnlyModelChanged(t, body, encoded)
		})
	}
}

func TestSpeechCodecDecodeRequestTurnsInputIntoUserText(t *testing.T) {
	body := []byte(`{"model":"tts-1","input":"Route me by language","voice":"alloy"}`)
	request, _, _, err := SpeechCodec{}.DecodeRequest(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if request.Model != "tts-1" {
		t.Fatalf("model = %q, want the client model", request.Model)
	}
	text, ok := lastUserRequestText(request)
	if !ok || text != "Route me by language" {
		t.Fatalf("user text = %q, %v; want the speech input so routing signals can read it", text, ok)
	}
	if request.SpeechGeneration == nil {
		t.Fatalf("decoded request must carry speech options")
	}
	for _, name := range []string{"input", "model"} {
		if _, stored := request.SpeechGeneration.Fields[name]; stored {
			t.Fatalf("%q must not be duplicated in the speech options", name)
		}
	}
	if got := string(request.SpeechGeneration.Fields["voice"]); got != `"alloy"` {
		t.Fatalf("voice = %s, want the raw client value", got)
	}
}

// The spoken text is read back from the user message, so a router step that
// rewrites the prompt also changes what the backend speaks.
func TestSpeechCodecEncodeRequestSpeaksTheCurrentUserText(t *testing.T) {
	request, _, _, err := SpeechCodec{}.DecodeRequest([]byte(`{"input":"original text","voice":"alloy"}`), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	last := len(request.Messages) - 1
	request.Messages[last].Content[0].Text = "rewritten text"
	encoded, _, err := SpeechCodec{}.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	if got := decodeJSONForCompare(t, encoded)["input"]; got != "rewritten text" {
		t.Fatalf("input = %v, want the current user text", got)
	}
}

func TestSpeechCodecDecodeRequestRejectsInvalidRequests(t *testing.T) {
	cases := map[string]struct {
		body     string
		category llmprotocol.ErrorCategory
	}{
		"unknown top-level field":      {`{"input":"hi","foo":1}`, llmprotocol.ErrorInvalidRequest},
		"field name with another case": {`{"input":"hi","Voice":"alloy"}`, llmprotocol.ErrorInvalidRequest},
		"missing input":                {`{"voice":"alloy"}`, llmprotocol.ErrorInvalidRequest},
		"empty input":                  {`{"input":""}`, llmprotocol.ErrorInvalidRequest},
		"input is not a string":        {`{"input":42}`, llmprotocol.ErrorInvalidRequest},
		"model is not a string":        {`{"input":"hi","model":7}`, llmprotocol.ErrorInvalidRequest},
		"duplicate field":              {`{"input":"hi","input":"again"}`, llmprotocol.ErrorInvalidRequest},
		"trailing json":                {`{"input":"hi"} {}`, llmprotocol.ErrorInvalidRequest},
		"not an object":                {`["hi"]`, llmprotocol.ErrorInvalidRequest},
		"stream is not a boolean":      {`{"input":"hi","stream":"yes"}`, llmprotocol.ErrorInvalidRequest},
		"streaming requested":          {`{"input":"hi","stream":true}`, llmprotocol.ErrorUnsupportedFeature},
		"streaming format requested":   {`{"input":"hi","stream_format":"audio"}`, llmprotocol.ErrorUnsupportedFeature},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, _, err := SpeechCodec{}.DecodeRequest([]byte(tc.body), llmprotocol.DefaultPolicy())
			assertProtocolErrorCategory(t, err, tc.category)
		})
	}
}

// Unknown top-level fields return 400 even when the caller's policy would
// otherwise preserve them for same-format replay.
func TestSpeechCodecRejectsUnknownFieldsUnderAnyPolicy(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.UnknownFields = llmprotocol.UnknownPreserveSameFormat
	_, _, _, err := SpeechCodec{}.DecodeRequest([]byte(`{"input":"hi","foo":1}`), policy)
	assertProtocolErrorCategory(t, err, llmprotocol.ErrorInvalidRequest)
}

func TestSpeechCodecCapabilitiesMatchDecodedRequests(t *testing.T) {
	codec := SpeechCodec{}
	set := codec.Capabilities()
	if !set.Supports(llmprotocol.CapabilitySpeechGeneration) {
		t.Fatalf("speech codec must advertise speech_generation")
	}
	if set.Supports(llmprotocol.CapabilityStreaming) {
		t.Fatalf("speech codec must not advertise streaming in the non-streaming MVP")
	}
	if set.Supports(llmprotocol.CapabilityAudioOutput) {
		t.Fatalf("speech codec must not advertise audio_output, which covers audio inside Chat or Responses")
	}
	request, _, _, err := codec.DecodeRequest([]byte(`{"input":"hi","voice":"alloy"}`), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	required := llmprotocol.RequiredCapabilities(request)
	if !required.Supports(llmprotocol.CapabilitySpeechGeneration) {
		t.Fatalf("decoded speech request must require speech_generation, got %v", required.Names())
	}
	if !set.Contains(required) {
		t.Fatalf("speech capabilities %v must contain required %v", set.Names(), required.Names())
	}
}

func TestSpeechCodecEncodeRequestFailsClosed(t *testing.T) {
	userText := []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentText, Text: "hi"},
	}}}
	cases := map[string]llmprotocol.Request{
		"chat request without speech options": {Messages: userText},
		"no user text": {
			SpeechGeneration: &llmprotocol.SpeechGenerationOptions{},
		},
		"unknown option field": {
			Messages:         userText,
			SpeechGeneration: &llmprotocol.SpeechGenerationOptions{Fields: map[string]json.RawMessage{"foo": json.RawMessage(`1`)}},
		},
		"option overriding input": {
			Messages:         userText,
			SpeechGeneration: &llmprotocol.SpeechGenerationOptions{Fields: map[string]json.RawMessage{"input": json.RawMessage(`"other"`)}},
		},
		"streaming request": {
			Messages:         userText,
			Stream:           true,
			SpeechGeneration: &llmprotocol.SpeechGenerationOptions{},
		},
	}
	for name, request := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, err := SpeechCodec{}.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
			assertProtocolErrorCategory(t, err, llmprotocol.ErrorUnsupportedFeature)
		})
	}
}

func TestSpeechCodecAudioResponsesBypassTheNeutralResponse(t *testing.T) {
	codec := SpeechCodec{}
	policy := llmprotocol.DefaultPolicy()
	if _, _, _, err := codec.DecodeResponse([]byte("RIFF\x24\x00\x00\x00WAVEfmt "), policy); err == nil {
		t.Fatalf("DecodeResponse must refuse audio bytes")
	}
	if _, _, err := codec.EncodeResponse(llmprotocol.Response{}, llmprotocol.Envelope{}, policy); err == nil {
		t.Fatalf("EncodeResponse must refuse to render a neutral response as speech")
	}
	if _, _, err := codec.NewDecoder(llmprotocol.StreamContext{}, policy).Push([]byte("chunk")); err == nil {
		t.Fatalf("speech stream decoder must fail in the non-streaming MVP")
	}
	if _, _, err := codec.NewEncoder(llmprotocol.StreamContext{}, policy).Push(llmprotocol.Event{}); err == nil {
		t.Fatalf("speech stream encoder must fail in the non-streaming MVP")
	}
}

func TestSpeechCodecDecodeTransportErrorKeepsBackendMessage(t *testing.T) {
	cases := map[string]struct {
		body    string
		message string
		code    string
	}{
		"vllm-omni numeric code": {
			body:    `{"error":{"message":"voice 'bob' is not supported","type":"BadRequestError","param":null,"code":400}}`,
			message: "voice 'bob' is not supported",
			code:    "speech_upstream_error",
		},
		"string code": {
			body:    `{"error":{"message":"rate limited","type":"rate_limit","code":"rate_limit_exceeded"}}`,
			message: "rate limited",
			code:    "rate_limit_exceeded",
		},
		"empty body": {
			body:    ``,
			message: "speech service returned an error",
			code:    "speech_upstream_error",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			transportErr, _, err := SpeechCodec{}.DecodeTransportError([]byte(tc.body), llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatalf("DecodeTransportError: %v", err)
			}
			if transportErr.Error == nil || transportErr.Error.Message != tc.message || transportErr.Error.Code != tc.code {
				t.Fatalf("transport error = %+v, want message %q code %q", transportErr.Error, tc.message, tc.code)
			}
		})
	}

	body := SpeechCodec{}.EncodeTransportError(llmprotocol.TransportError{
		Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "rate_limit_exceeded", "rate limited", nil),
	})
	if !strings.Contains(string(body), `"message":"rate limited"`) || !strings.Contains(string(body), `"code":"rate_limit_exceeded"`) {
		t.Fatalf("encoded transport error = %s, want the OpenAI error envelope", body)
	}
}

func TestSpeechEngineTranslateSwapsOnlyTheModel(t *testing.T) {
	body := `{"model":"tts-1","input":"Through the engine","voice":{"id":"voice_123"},"task_type":"CustomVoice","extra_params":{"k":1}}`
	result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAISpeechV1, llmprotocol.OpenAISpeechV1, []byte(body), func(request *llmprotocol.Request) error {
		request.Model = speechBackendModel
		return nil
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	assertOnlyModelChanged(t, body, result.Body)
}

func TestSpeechEngineRejectsChatRequestsForSpeechBackends(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	if _, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAISpeechV1, body, nil); err == nil {
		t.Fatalf("a chat request must not be encoded for a speech backend")
	}
}
