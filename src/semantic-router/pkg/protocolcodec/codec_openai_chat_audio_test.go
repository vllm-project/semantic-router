package protocolcodec

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
