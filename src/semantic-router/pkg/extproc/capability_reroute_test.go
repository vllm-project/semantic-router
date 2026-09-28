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
	if err := router.providerCapabilityMismatch("unannotated", llmprotocol.OpenAIChatV1, imageRequired); err != nil {
		t.Fatal("unannotated model must stay eligible on wire expressibility for an image request")
	}
	// Descriptive catalog labels are metadata, not capability declarations:
	// they leave the model unannotated instead of voiding its eligibility.
	if err := router.providerCapabilityMismatch("descriptive", llmprotocol.OpenAIChatV1, imageRequired); err != nil {
		t.Fatal("descriptive-label-only model must stay eligible on wire expressibility for an image request")
	}
	// A transport/accounting-only declaration is annotated yet carries no task
	// bit, so it cannot serve a task it never declared.
	if err := router.providerCapabilityMismatch("transport-only", llmprotocol.OpenAIChatV1, imageRequired); err == nil {
		t.Fatal("transport-only declaration must not serve an image request")
	}
	// A catalog alias projects onto the protocol vocabulary before the filter:
	// "vision" describes image input.
	if err := router.providerCapabilityMismatch("alias-only", llmprotocol.OpenAIChatV1, imageRequired); err != nil {
		t.Fatal("vision-only declaration must satisfy an image request")
	}
	// A descriptive label alongside a recognized task bit does not void it.
	if err := router.providerCapabilityMismatch("mixed-declared", llmprotocol.OpenAIChatV1, imageRequired); err != nil {
		t.Fatal("mixed declaration must stay eligible for the image task it declares")
	}
	if err := router.providerCapabilityMismatch("mixed-declared", llmprotocol.OpenAIChatV1, audioRequired); err == nil {
		t.Fatal("mixed declaration must be rejected for an audio request it never declared")
	}
	// A model declaring another task does not become a candidate for this one.
	if err := router.providerCapabilityMismatch("image-declared", llmprotocol.OpenAIChatV1, audioRequired); err == nil {
		t.Fatal("image-declared model must be rejected for an audio request")
	}
}
