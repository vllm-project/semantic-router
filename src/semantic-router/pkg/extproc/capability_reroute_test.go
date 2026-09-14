package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// TestQualifiedRerouteCandidateDeclaredCapabilities is a regression test for the
// declared-capability filter at the dispatch seam. Eligibility is decided by the
// same qualification the primary dispatch and the fallback candidates share: the
// wire format must encode the request, and an annotated model must declare the
// required task bits. Catalog input-modality aliases are projected onto the
// protocol vocabulary before that filter, descriptive catalog labels neither
// grant nor revoke eligibility, and a transport/accounting-only declaration
// carries no task bit so it cannot serve a task request it never declared.
func TestQualifiedRerouteCandidateDeclaredCapabilities(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"unannotated":    {APIFormat: "openai"},
					"transport-only": {APIFormat: "openai", Capabilities: []string{"tools", "streaming"}},
					"image-declared": {APIFormat: "openai", Capabilities: []string{"image_input"}},
					"alias-only":     {APIFormat: "openai", Capabilities: []string{"vision"}},
					"mixed-declared": {APIFormat: "openai", Capabilities: []string{"image_input", "long_context"}},
					"descriptive":    {APIFormat: "openai", Capabilities: []string{"coding", "long_context"}},
				},
			},
		},
	}
	imageRequired := llmprotocol.Capabilities(llmprotocol.CapabilityImageInput)
	audioRequired := llmprotocol.Capabilities(llmprotocol.CapabilityAudioInput)

	// A model with no declaration stays eligible on wire expressibility alone.
	if got := router.qualifiedRerouteCandidate("unannotated", imageRequired); got == "" {
		t.Fatal("unannotated model must stay eligible on wire expressibility for an image request")
	}
	// Descriptive catalog labels are metadata, not capability declarations:
	// they leave the model unannotated instead of voiding its eligibility.
	if got := router.qualifiedRerouteCandidate("descriptive", imageRequired); got == "" {
		t.Fatal("descriptive-label-only model must stay eligible on wire expressibility for an image request")
	}
	// A transport/accounting-only declaration is annotated yet carries no task
	// bit, so it cannot serve a task it never declared.
	if got := router.qualifiedRerouteCandidate("transport-only", imageRequired); got != "" {
		t.Fatalf("transport-only declaration must not serve an image request, got format %q", got)
	}
	// A catalog alias projects onto the protocol vocabulary before the filter:
	// "vision" describes image input.
	if got := router.qualifiedRerouteCandidate("alias-only", imageRequired); got == "" {
		t.Fatal("vision-only declaration must satisfy an image request")
	}
	// A descriptive label alongside a recognized task bit does not void it.
	if got := router.qualifiedRerouteCandidate("mixed-declared", imageRequired); got == "" {
		t.Fatal("mixed declaration must stay eligible for the image task it declares")
	}
	if got := router.qualifiedRerouteCandidate("mixed-declared", audioRequired); got != "" {
		t.Fatalf("mixed declaration must be rejected for an audio request it never declared, got format %q", got)
	}
	// A model declaring another task does not become a candidate for this one.
	if got := router.qualifiedRerouteCandidate("image-declared", audioRequired); got != "" {
		t.Fatalf("image-declared model must be rejected for an audio request, got format %q", got)
	}
}
