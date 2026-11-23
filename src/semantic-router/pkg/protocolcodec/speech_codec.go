package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// SpeechCodec is the OpenAI Speech API dialect (/v1/audio/speech) served by
// TTS backends such as vLLM-Omni. It is a sink dialect like ImagesCodec: the
// router reads the input text for routing and swaps the model, and every other
// request field is forwarded unchanged. Audio responses bypass the neutral
// response, so this codec never decodes or encodes a response body.
type SpeechCodec struct{}

func (SpeechCodec) Format() llmprotocol.WireFormat { return llmprotocol.OpenAISpeechV1 }
func (SpeechCodec) Stateless() bool                { return true }
func (SpeechCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText,
		llmprotocol.CapabilitySpeechGeneration,
	)
}

// speechRequestWire names every field of vLLM-Omni's OpenAICreateSpeechRequest
// so strict decoding rejects anything else. Values other than input and model
// stay raw because the backend accepts several shapes for some of them (voice
// as a string or object, ref_audio as a string or list) and validates them
// itself.
type speechRequestWire struct {
	Input                   string          `json:"input"`
	Model                   string          `json:"model"`
	Voice                   json.RawMessage `json:"voice"`
	Speaker                 json.RawMessage `json:"speaker"`
	Instructions            json.RawMessage `json:"instructions"`
	ResponseFormat          json.RawMessage `json:"response_format"`
	SampleRate              json.RawMessage `json:"sample_rate"`
	Speed                   json.RawMessage `json:"speed"`
	StreamFormat            json.RawMessage `json:"stream_format"`
	Stream                  json.RawMessage `json:"stream"`
	TaskType                json.RawMessage `json:"task_type"`
	Language                json.RawMessage `json:"language"`
	RefAudio                json.RawMessage `json:"ref_audio"`
	RefText                 json.RawMessage `json:"ref_text"`
	RefAudio2               json.RawMessage `json:"ref_audio_2"`
	AmbientSound            json.RawMessage `json:"ambient_sound"`
	DurationSeconds         json.RawMessage `json:"duration_seconds"`
	XVectorOnlyMode         json.RawMessage `json:"x_vector_only_mode"`
	SpeakerEmbedding        json.RawMessage `json:"speaker_embedding"`
	MaxNewTokens            json.RawMessage `json:"max_new_tokens"`
	Seed                    json.RawMessage `json:"seed"`
	InitialCodecChunkFrames json.RawMessage `json:"initial_codec_chunk_frames"`
	NonStreamingMode        json.RawMessage `json:"non_streaming_mode"`
	ExtraParams             json.RawMessage `json:"extra_params"`
	WordTimestamps          json.RawMessage `json:"word_timestamps"`
}

// options returns the forwarded fields keyed by wire name. The keys are the
// allowed set that EncodeRequest checks against.
func (wire speechRequestWire) options() map[string]json.RawMessage {
	return map[string]json.RawMessage{
		"voice":                      wire.Voice,
		"speaker":                    wire.Speaker,
		"instructions":               wire.Instructions,
		"response_format":            wire.ResponseFormat,
		"sample_rate":                wire.SampleRate,
		"speed":                      wire.Speed,
		"stream_format":              wire.StreamFormat,
		"stream":                     wire.Stream,
		"task_type":                  wire.TaskType,
		"language":                   wire.Language,
		"ref_audio":                  wire.RefAudio,
		"ref_text":                   wire.RefText,
		"ref_audio_2":                wire.RefAudio2,
		"ambient_sound":              wire.AmbientSound,
		"duration_seconds":           wire.DurationSeconds,
		"x_vector_only_mode":         wire.XVectorOnlyMode,
		"speaker_embedding":          wire.SpeakerEmbedding,
		"max_new_tokens":             wire.MaxNewTokens,
		"seed":                       wire.Seed,
		"initial_codec_chunk_frames": wire.InitialCodecChunkFrames,
		"non_streaming_mode":         wire.NonStreamingMode,
		"extra_params":               wire.ExtraParams,
		"word_timestamps":            wire.WordTimestamps,
	}
}

var speechOptionNames = speechRequestWire{}.options()

// DecodeRequest decodes a Speech API request into the neutral request IR. The
// input becomes the user text message so routing signals can read it, and the
// other fields are kept raw in the speech options. Unknown top-level fields are
// rejected under every policy.
func (SpeechCodec) DecodeRequest(body []byte, policy llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	policy.UnknownFields = llmprotocol.UnknownReject
	var wire speechRequestWire
	if err := decodeWire(body, &wire, policy); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	if wire.Input == "" {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, invalidSpeechRequest(fmt.Errorf("speech input is required"))
	}
	if err := rejectSpeechStreaming(wire); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	fields := make(map[string]json.RawMessage)
	for name, value := range wire.options() {
		if len(value) > 0 {
			fields[name] = value
		}
	}
	request := llmprotocol.Request{
		Generation: 1,
		Model:      wire.Model,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: wire.Input}},
		}},
		SpeechGeneration: &llmprotocol.SpeechGenerationOptions{Fields: fields},
	}
	return request, llmprotocol.Envelope{}, nil, nil
}

// rejectSpeechStreaming keeps the MVP non-streaming: stream must be a boolean
// and false, and stream_format must be absent.
func rejectSpeechStreaming(wire speechRequestWire) error {
	if len(wire.Stream) > 0 {
		var stream *bool
		if err := json.Unmarshal(wire.Stream, &stream); err != nil {
			return invalidSpeechRequest(fmt.Errorf("speech stream must be a boolean"))
		}
		if stream != nil && *stream {
			return unsupportedDownstreamTranslation(fmt.Errorf("speech wire does not support streaming"))
		}
	}
	if len(wire.StreamFormat) > 0 && !bytes.Equal(bytes.TrimSpace(wire.StreamFormat), []byte("null")) {
		return unsupportedDownstreamTranslation(fmt.Errorf("speech wire does not support stream_format"))
	}
	return nil
}

// EncodeRequest renders a neutral speech request for the backend. The input is
// the last user text, so a router step that edits the prompt also edits what
// is spoken. Any request the Speech API cannot express fails closed.
func (SpeechCodec) EncodeRequest(request llmprotocol.Request, _ llmprotocol.Envelope, _ llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if request.SpeechGeneration == nil {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("speech wire requires a speech generation request"))
	}
	if request.Stream {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("speech wire does not support streaming"))
	}
	input, ok := lastUserRequestText(request)
	if !ok {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("speech input is unavailable"))
	}
	wire := make(map[string]json.RawMessage, len(request.SpeechGeneration.Fields)+2)
	for name, value := range request.SpeechGeneration.Fields {
		if _, allowed := speechOptionNames[name]; !allowed {
			return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("speech wire does not support field %q", name))
		}
		wire[name] = value
	}
	encodedInput, err := json.Marshal(input)
	if err != nil {
		return nil, nil, fmt.Errorf("encode speech input: %w", err)
	}
	wire["input"] = encodedInput
	if request.Model != "" {
		encodedModel, modelErr := json.Marshal(request.Model)
		if modelErr != nil {
			return nil, nil, fmt.Errorf("encode speech model: %w", modelErr)
		}
		wire["model"] = encodedModel
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, fmt.Errorf("encode speech request: %w", err)
	}
	return body, nil, nil
}

func (SpeechCodec) DecodeResponse(_ []byte, _ llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, speechResponseBypassError()
}

func (SpeechCodec) EncodeResponse(_ llmprotocol.Response, _ llmprotocol.Envelope, _ llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	return nil, nil, speechResponseBypassError()
}

func speechResponseBypassError() error {
	return unsupportedDownstreamTranslation(fmt.Errorf("speech audio bypasses the neutral response"))
}

// DecodeTransportError reads the OpenAI error envelope. vLLM-Omni sends the
// HTTP status as a numeric code, so code is read raw and only a string code is
// kept; the backend's message is kept either way.
func (SpeechCodec) DecodeTransportError(body []byte, _ llmprotocol.Policy) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire struct {
		Error *struct {
			Message string          `json:"message"`
			Code    json.RawMessage `json:"code"`
		} `json:"error"`
	}
	if len(bytes.TrimSpace(body)) > 0 {
		if err := json.Unmarshal(body, &wire); err != nil {
			return llmprotocol.TransportError{}, nil, invalidUpstreamResponse(err)
		}
	}
	message := "speech service returned an error"
	code := "speech_upstream_error"
	if wire.Error != nil {
		if wire.Error.Message != "" {
			message = wire.Error.Message
		}
		var stringCode string
		if json.Unmarshal(wire.Error.Code, &stringCode) == nil && stringCode != "" {
			code = stringCode
		}
	}
	return llmprotocol.TransportError{
		Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, code, message, nil),
	}, nil, nil
}

// EncodeTransportError uses the same OpenAI error envelope as the images
// dialect, which is what Speech API clients expect.
func (SpeechCodec) EncodeTransportError(err llmprotocol.TransportError) []byte {
	return ImagesCodec{}.EncodeTransportError(err)
}

// The MVP never streams; the stubs satisfy the codec registry and fail loudly
// if invoked.
func (SpeechCodec) NewDecoder(_ llmprotocol.StreamContext, _ llmprotocol.Policy) llmprotocol.StreamDecoder {
	return speechStreamUnsupportedDecoder{}
}

func (SpeechCodec) NewEncoder(_ llmprotocol.StreamContext, _ llmprotocol.Policy) llmprotocol.StreamEncoder {
	return speechStreamUnsupportedEncoder{}
}

type (
	speechStreamUnsupportedDecoder struct{}
	speechStreamUnsupportedEncoder struct{}
)

func (speechStreamUnsupportedDecoder) Push([]byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("speech wire does not support streaming")
}

func (speechStreamUnsupportedDecoder) Finalize(error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("speech wire does not support streaming")
}

func (speechStreamUnsupportedEncoder) Push(llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("speech wire does not support streaming")
}

func (speechStreamUnsupportedEncoder) Finalize(error) ([][]byte, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("speech wire does not support streaming")
}

func invalidSpeechRequest(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_speech_request", cause.Error(), cause)
}
