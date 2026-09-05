package llmprotocol

import "testing"

// An explicit tool_choice: none forbids all tools, including the hosted
// image_generation operation: a declared ImageGeneration must not require the
// images capability in that case, so capability dispatch avoids the images
// backend (Xun review 5119851642).
func TestRequiredCapabilitiesNoneToolChoiceDoesNotRequireImageGeneration(t *testing.T) {
	required := RequiredCapabilities(Request{
		ImageGeneration: &ImageGenerationOptions{},
		ToolChoice:      ToolChoice{Mode: ToolChoiceNone},
	})
	if required.Supports(CapabilityImageGeneration) {
		t.Fatalf("tool_choice none must not require image_generation, got %v", required.Names())
	}
}

func TestRequiredCapabilitiesImageGenerationToolChoiceRequiresImageGeneration(t *testing.T) {
	required := RequiredCapabilities(Request{
		ToolChoice: ToolChoice{Mode: ToolChoiceImageGeneration},
	})
	if !required.Supports(CapabilityImageGeneration) {
		t.Fatalf("image_generation tool choice must require image_generation, got %v", required.Names())
	}
}

// TaskCapabilities keeps the modality/task bits that model capability
// declarations can meaningfully claim, and drops transport/accounting fidelity
// (text is assumed for every model, tools/streaming/reasoning are protocol
// concerns rather than task modalities).
func TestCapabilitySetTaskCapabilities(t *testing.T) {
	full := Capabilities(
		CapabilityImageGeneration, CapabilityImageInput, CapabilityImageOutput,
		CapabilityAudioInput, CapabilityVideoOutput, CapabilityFileOutput,
		CapabilityText, CapabilityTools, CapabilityStreaming, CapabilityReasoning,
	)
	task := full.TaskCapabilities()
	for _, kept := range []Capability{
		CapabilityImageGeneration, CapabilityImageInput, CapabilityImageOutput,
		CapabilityAudioInput, CapabilityVideoOutput, CapabilityFileOutput,
	} {
		if !task.Supports(kept) {
			t.Fatalf("TaskCapabilities must keep %v, got %v", Capabilities(kept).Names(), task.Names())
		}
	}
	for _, dropped := range []Capability{CapabilityText, CapabilityTools, CapabilityStreaming, CapabilityReasoning} {
		if task.Supports(dropped) {
			t.Fatalf("TaskCapabilities must drop %v, got %v", Capabilities(dropped).Names(), task.Names())
		}
	}
}
