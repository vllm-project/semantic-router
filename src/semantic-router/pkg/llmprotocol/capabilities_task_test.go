package llmprotocol

import "testing"

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
