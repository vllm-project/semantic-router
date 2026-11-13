package protocolcodec

import (
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
	hasAudio := false
	for _, modality := range modalities {
		switch modality {
		case "text":
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
	options := &llmprotocol.AudioOutputOptions{}
	if audio != nil {
		options.Voice, options.Format = audio.Voice, audio.Format
	}
	return options, nil
}

func invalidChatModalities(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_modalities", "Chat modalities may contain only text and audio", cause)
}

// decodeChatMessageAudio admits message.audio on an assistant message only: in
// Chat, audio is the assistant's spoken answer or a reference to one, never an
// attachment on a user, system or tool turn.
func decodeChatMessageAudio(role llmprotocol.Role, wire *chatAudioOutputWire) (llmprotocol.Content, error) {
	if role != llmprotocol.RoleAssistant {
		return llmprotocol.Content{}, llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "audio_requires_assistant_role",
			"Chat message audio is only valid on an assistant message", nil,
		)
	}
	return llmprotocol.Content{
		Kind: llmprotocol.ContentAudio,
		GeneratedAudio: &llmprotocol.GeneratedAudio{
			ID: wire.ID, Data: wire.Data, ExpiresAt: wire.ExpiresAt, Transcript: wire.Transcript,
		},
	}, nil
}

// Wire format names belong to the codec. Neutral media always uses MIME types,
// so routing and other consumers do not need provider-specific format aliases.
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
