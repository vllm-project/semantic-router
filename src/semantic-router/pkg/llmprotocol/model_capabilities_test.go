package llmprotocol

import "testing"

func TestModelCapabilitiesRetainsDeclarationsAlongsideCatalogLabels(t *testing.T) {
	set, annotated := ModelCapabilities([]string{"chat", "vision", "structured_output", "long_context", "coding"})
	if !annotated || !set.Contains(Capabilities(CapabilityText, CapabilityImageInput, CapabilityStructuredJSON)) {
		t.Fatalf("model capabilities = %v, annotated=%v", set.Names(), annotated)
	}
	if set.Supports(CapabilityImageGeneration) || set.Supports(CapabilityImageOutput) {
		t.Fatalf("vision must not grant image generation/output: %v", set.Names())
	}
	if _, err := ParseCapabilities([]string{"long_context"}); err == nil {
		t.Fatal("protocol capability parsing must remain strict")
	}
}

func TestModelCapabilitiesDistinguishesDescriptiveMetadata(t *testing.T) {
	for _, names := range [][]string{nil, {"coding", "long_context", "cheap_followup"}} {
		if set, annotated := ModelCapabilities(names); annotated || !set.Empty() {
			t.Fatalf("descriptive labels = %v, annotated=%v", set.Names(), annotated)
		}
	}
	set, annotated := ModelCapabilities([]string{"image_input", "future_product_label"})
	if !annotated || !set.Supports(CapabilityImageInput) || set.Supports(CapabilityImageGeneration) {
		t.Fatalf("unknown label erased known restrictions: %v, annotated=%v", set.Names(), annotated)
	}
}

func TestModelCapabilitiesNormalizesInputAliases(t *testing.T) {
	set, annotated := ModelCapabilities([]string{" VISION ", "Audio", "VIDEO", "tool_use", "image_generation"})
	want := Capabilities(CapabilityImageInput, CapabilityAudioInput, CapabilityVideoInput, CapabilityTools, CapabilityImageGeneration)
	if !annotated || set != want {
		t.Fatalf("model capabilities = %v, want %v", set.Names(), want.Names())
	}
}
