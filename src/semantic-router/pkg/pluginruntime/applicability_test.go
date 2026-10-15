package pluginruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestApplicable(t *testing.T) {
	text := llmprotocol.Capabilities(llmprotocol.CapabilityText)
	image := llmprotocol.Capabilities(llmprotocol.CapabilityImageInput)
	both := llmprotocol.Capabilities(llmprotocol.CapabilityText, llmprotocol.CapabilityImageInput)
	for _, tc := range []struct {
		name     string
		plugin   string
		request  llmprotocol.CapabilitySet
		backend  llmprotocol.CapabilitySet
		declared bool
		apply    bool
		reason   SkipReason
	}{
		{"text request, text backend", config.DecisionPluginRAG, text, text, true, true, ""},
		{"text and image request keeps running", config.DecisionPluginMemory, both, both, true, true, ""},
		{"undeclared backend is not gated", config.DecisionPluginRAG, text, llmprotocol.CapabilitySet{}, false, true, ""},
		{"image-only request has no text to act on", config.DecisionPluginRAG, image, both, true, false, SkipRequestDoesNotRequire},
		{"text is assumed, so a card without text still runs", config.DecisionPluginMemory, text, image, true, true, ""},
		{"ungated plugin always runs", config.DecisionPluginRouterReplay, image, image, true, true, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			apply, reason := Applicable(tc.plugin, tc.request, tc.backend, tc.declared)
			if apply != tc.apply || reason != tc.reason {
				t.Fatalf("Applicable() = (%v, %q), want (%v, %q)", apply, reason, tc.apply, tc.reason)
			}
		})
	}
}

func TestApplicableGatesBackendOnTaskCapabilities(t *testing.T) {
	const probe = "image_probe"
	pluginRequirements[probe] = llmprotocol.Capabilities(llmprotocol.CapabilityImageInput)
	t.Cleanup(func() { delete(pluginRequirements, probe) })
	image := llmprotocol.Capabilities(llmprotocol.CapabilityImageInput)
	text := llmprotocol.Capabilities(llmprotocol.CapabilityText)
	if apply, reason := Applicable(probe, image, text, true); apply || reason != SkipBackendLacksCapability {
		t.Fatalf("image plugin on a text-only card = (%v, %q)", apply, reason)
	}
	if apply, _ := Applicable(probe, image, image, true); !apply {
		t.Fatal("image plugin was skipped on a card that declares image input")
	}
}
