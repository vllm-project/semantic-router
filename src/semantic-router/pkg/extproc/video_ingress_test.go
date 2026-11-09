package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestChatVideoIngressUsesDeclaredBackendCapabilities(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.invalid/clip.mp4"}}]}]}`)
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	required := llmprotocol.RequiredCapabilities(request)
	if !required.Contains(llmprotocol.Capabilities(llmprotocol.CapabilityVideoInput)) {
		t.Fatal("decoded video did not require video_input")
	}
	for _, tc := range []struct {
		name         string
		capabilities []string
		wantRejected bool
	}{
		{name: "undeclared"},
		{name: "video declared", capabilities: []string{"video_input"}},
		{name: "image only", capabilities: []string{"image_input"}, wantRejected: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// No decision deployment or media opt-in is needed for video ingress.
			router := &OpenAIRouter{Config: &config.RouterConfig{
				BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
					"selected": {APIFormat: "openai", Capabilities: tc.capabilities},
				}},
			}}
			err := router.providerCapabilityMismatch("selected", llmprotocol.OpenAIChatV1, required)
			if (err != nil) != tc.wantRejected {
				t.Fatalf("capability check = %v, want rejected=%v", err, tc.wantRejected)
			}
		})
	}
}
