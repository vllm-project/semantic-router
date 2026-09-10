package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// TestQualifiedRerouteCandidateTransportOnlyDeclaration is the regression for
// Xunzhuo's review (#3183): a model declaring only transport/accounting
// capabilities (tools, streaming, ...) must stay eligible on wire
// expressibility, exactly like an unannotated model, instead of being rejected
// for a task bit it never claimed.
func TestQualifiedRerouteCandidateTransportOnlyDeclaration(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"transport-only": {APIFormat: "openai", Capabilities: []string{"tools", "streaming"}},
					"unannotated":    {APIFormat: "openai"},
					"image-declared": {APIFormat: "openai", Capabilities: []string{"image_input"}},
				},
			},
		},
	}
	imageRequired := llmprotocol.Capabilities(llmprotocol.CapabilityImageInput)
	audioRequired := llmprotocol.Capabilities(llmprotocol.CapabilityAudioInput)

	if got := router.qualifiedRerouteCandidate("transport-only", imageRequired); got == "" {
		t.Fatal("transport-only declared model must stay eligible on wire expressibility for an image request")
	}
	if got := router.qualifiedRerouteCandidate("unannotated", imageRequired); got == "" {
		t.Fatal("unannotated model must stay eligible on wire expressibility for an image request")
	}
	// Task-annotated model matching the request stays eligible.
	if got := router.qualifiedRerouteCandidate("image-declared", imageRequired); got == "" {
		t.Fatal("image-declared model must satisfy an image request")
	}
	// Task-annotated model lacking the required task still gets filtered.
	if got := router.qualifiedRerouteCandidate("image-declared", audioRequired); got != "" {
		t.Fatalf("image-declared model must be rejected for an audio request, got format %q", got)
	}
}
