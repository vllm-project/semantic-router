package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// TestQualifiedRerouteCandidateTransportOnlyDeclaration is a regression test:
// a model declaring only transport/accounting capabilities (tools, streaming,
// ...) carries no task bit and must stay eligible on wire expressibility
// for a task request, exactly like an unannotated model, instead of being
// rejected for a task it never claimed.
func TestQualifiedRerouteCandidateTransportOnlyDeclaration(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"transport-only": {APIFormat: "openai", Capabilities: []string{"tools", "streaming"}},
					"unannotated":    {APIFormat: "openai"},
					"image-declared": {APIFormat: "openai", Capabilities: []string{"image_input"}},
					"mixed-declared": {APIFormat: "openai", Capabilities: []string{"image_input", "vision"}},
					"unknown-only":   {APIFormat: "openai", Capabilities: []string{"vision"}},
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
	// A mixed known/unknown declaration keeps its valid task bits: the
	// unrecognized "vision" word must not void the recognized image_input bit.
	if got := router.qualifiedRerouteCandidate("mixed-declared", imageRequired); got == "" {
		t.Fatal("mixed known/unknown declaration must stay eligible for an image request")
	}
	if got := router.qualifiedRerouteCandidate("mixed-declared", audioRequired); got != "" {
		t.Fatalf("mixed known/unknown declaration must be rejected for audio, got format %q", got)
	}
	// A declaration with no recognized name carries no task bit and is treated
	// like an unannotated model.
	if got := router.qualifiedRerouteCandidate("unknown-only", imageRequired); got == "" {
		t.Fatal("unknown-name-only declaration must stay eligible on wire expressibility, matching an unannotated model")
	}
}
