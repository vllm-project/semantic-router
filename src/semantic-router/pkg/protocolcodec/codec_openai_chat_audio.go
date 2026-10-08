package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

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
