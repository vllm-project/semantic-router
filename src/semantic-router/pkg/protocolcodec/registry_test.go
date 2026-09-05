package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRegistryCapabilitiesForResolvesBuiltinFormats(t *testing.T) {
	registry := NewBuiltinRegistry()

	set, ok := registry.CapabilitiesFor(llmprotocol.OpenAIResponsesV1)
	if !ok {
		t.Fatalf("responses format did not resolve")
	}
	if !set.Supports(llmprotocol.CapabilityImageGeneration) {
		t.Fatalf("responses wire must advertise image_generation capability")
	}

	chat, ok := registry.CapabilitiesFor(llmprotocol.OpenAIChatV1)
	if !ok {
		t.Fatalf("chat format did not resolve")
	}
	if chat.Supports(llmprotocol.CapabilityImageGeneration) {
		t.Fatalf("chat wire must not advertise image_generation capability")
	}
}

func TestRegistryCapabilitiesForUnknownFormat(t *testing.T) {
	registry := NewBuiltinRegistry()
	if _, ok := registry.CapabilitiesFor(llmprotocol.WireFormat("no.such.format")); ok {
		t.Fatalf("unknown format must not resolve")
	}
}
