package protocolcodec

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatVideoNeutralRoundTrip(t *testing.T) {
	for _, url := range []string{"https://example.invalid/clip.mp4", "data:video/mp4;base64,AAAAFGZ0eXBpc29t"} {
		engine := NewBuiltinEngine()
		body := []byte(fmt.Sprintf(`{"model":"source","messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":%q}}]}]}`, url))
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
		if err != nil {
			t.Fatal(err)
		}
		part := request.Messages[0].Content[0]
		if part.Kind != llmprotocol.ContentVideo || (part.URL != url && "data:"+part.MediaType+";base64,"+part.Data != url) {
			t.Fatal("neutral video")
		}
		request.Model = "selected"
		request.Generation++
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
		if parts[0].VideoURL == nil || parts[0].VideoURL.URL != url {
			t.Fatal("forwarded URL changed")
		}
	}
}

func TestChatVideoRejectsMissingURLAndWrongRole(t *testing.T) {
	for _, body := range []string{`{"model":"m","messages":[{"role":"user","content":[{"type":"video_url","video_url":{}}]}]}`, `{"model":"m","messages":[{"role":"assistant","content":[{"type":"video_url","video_url":{"url":"https://example.invalid/x"}}]}]}`} {
		if _, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(body)); err == nil {
			t.Fatal("invalid video accepted")
		}
	}
}
