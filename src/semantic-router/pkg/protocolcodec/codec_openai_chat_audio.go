package protocolcodec

import (
	"encoding/base64"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func decodeChatAudioOutputOptions(raw json.RawMessage, audio *chatAudioRequestWire) (*llmprotocol.AudioOutputOptions, error) {
	var modalities []string
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &modalities); err != nil {
			return nil, invalidChatModalities(err)
		}
	}
	hasAudio, hasText := false, false
	for _, modality := range modalities {
		switch modality {
		case "text":
			hasText = true
		case "audio":
			hasAudio = true
		default:
			return nil, invalidChatModalities(nil)
		}
	}
	if audio != nil && !hasAudio {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "audio_requires_modality", "Chat audio output requires the audio modality", nil)
	}
	if !hasAudio {
		return nil, nil
	}
	// The encoder derives ["text","audio"], so an audio-only request would come
	// back with a modality it did not ask for. Refuse it instead of rewriting it.
	if !hasText {
		return nil, invalidChatModalities(nil)
	}
	options := &llmprotocol.AudioOutputOptions{Format: audioFormat(audio)}
	voice, err := decodeChatAudioVoice(audio)
	if err != nil {
		return nil, err
	}
	options.Voice = voice
	return options, nil
}

func audioFormat(audio *chatAudioRequestWire) string {
	if audio == nil {
		return ""
	}
	return audio.Format
}

// decodeChatAudioVoice reads a preset voice name. A custom-voice reference is a
// modelled request the neutral layer does not carry yet, so it is refused with
// its own code rather than reported as malformed JSON.
func decodeChatAudioVoice(audio *chatAudioRequestWire) (string, error) {
	if audio == nil || len(audio.Voice) == 0 {
		return "", nil
	}
	var name string
	if err := json.Unmarshal(audio.Voice, &name); err == nil {
		return name, nil
	}
	var reference struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(audio.Voice, &reference); err == nil && reference.ID != "" {
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_custom_voice",
			"Chat Completions custom voice references are unsupported; name a preset voice", nil)
	}
	return "", invalidChatModalities(nil)
}

func invalidChatModalities(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_modalities", "Chat modalities may contain only text and audio", cause)
}

func decodeChatAudioContent(part chatContentWire) (llmprotocol.Content, error) {
	var mediaType string
	switch part.InputAudio.Format {
	case "wav":
		mediaType = "audio/wav"
	case "mp3":
		mediaType = "audio/mpeg"
	default:
		return llmprotocol.Content{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "unsupported_audio_format", "chat input audio format must be wav or mp3", nil)
	}
	return llmprotocol.Content{Kind: llmprotocol.ContentAudio, Data: part.InputAudio.Data, MediaType: mediaType, Cache: decodeAnthropicCacheControl(part.CacheControl)}, nil
}

// decodeChatMessageAudio admits message.audio on an assistant message only: in
// Chat, audio is the assistant's spoken answer or a reference to one, never an
// attachment on a user, system or tool turn. A request turn can only reference
// an earlier answer, so it must name one; a response carries the answer itself.
func decodeChatMessageAudio(role llmprotocol.Role, wire *chatAudioOutputWire, fromResponse bool) (llmprotocol.Content, error) {
	if role != llmprotocol.RoleAssistant {
		return llmprotocol.Content{}, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "audio_requires_assistant_role",
			"Chat message audio is only valid on an assistant message", nil,
		)
	}
	if !fromResponse && wire.ID == "" {
		return llmprotocol.Content{}, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "audio_reference_requires_id",
			"Chat assistant message audio must reference an audio id", nil,
		)
	}
	return llmprotocol.Content{
		Kind: llmprotocol.ContentAudio,
		GeneratedAudio: &llmprotocol.GeneratedAudio{
			ID: wire.ID, Data: wire.Data, ExpiresAt: wire.ExpiresAt, Transcript: wire.Transcript,
		},
	}, nil
}

// validateChatResponseAudio refuses an upstream artifact that names no audio and
// one whose inline payload is not base64. Both are provider faults: a client
// must not receive them as its own bad request, nor as an empty success.
func validateChatResponseAudio(wire *chatAudioOutputWire) error {
	if wire == nil {
		return nil
	}
	if wire.ID == "" && wire.Data == "" {
		return invalidProviderResponse("invalid_audio_output", "Chat Completions response audio carries neither an id nor data")
	}
	if wire.Data == "" {
		return nil
	}
	if _, err := base64.StdEncoding.Strict().DecodeString(wire.Data); err != nil {
		return invalidProviderResponse("invalid_audio_output", "Chat Completions response audio payload is not base64")
	}
	return nil
}

func (state *chatMessageEncodingState) appendAudio(content llmprotocol.Content) error {
	if audio := content.GeneratedAudio; audio != nil {
		if content.Cache != nil {
			return unsupportedChatCacheDirective(string(content.Kind))
		}
		// Chat carries one audio per message, on the assistant turn only.
		if state.wire.Role != "assistant" {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "audio_requires_assistant_role", "Chat message audio is only valid on an assistant message", nil)
		}
		if state.wire.Audio != nil {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "multiple_audio_outputs", "Chat Completions carries one audio artifact per message", nil)
		}
		state.wire.Audio = &chatAudioOutputWire{
			ID: audio.ID, Data: audio.Data, ExpiresAt: audio.ExpiresAt, Transcript: audio.Transcript,
		}
		return nil
	}
	// Without a generated artifact this is an input-audio content part, which an
	// assistant reply cannot carry.
	if state.wire.Role == "assistant" {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "audio_requires_generated_artifact", "Chat assistant content audio must carry the generated artifact", nil)
	}
	var format string
	switch content.MediaType {
	case "audio/wav":
		format = "wav"
	case "audio/mpeg":
		format = "mp3"
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_audio_media_type", "Chat Completions cannot represent this audio media type", nil)
	}
	state.parts = append(state.parts, chatContentWire{Type: "input_audio", InputAudio: &chatInputAudioWire{Data: content.Data, Format: format}, CacheControl: encodeAnthropicCacheControl(content.Cache)})
	return nil
}
