package config

import (
	"strings"
	"testing"
)

func TestAMDPublicModelMetadata(t *testing.T) {
	if got := PublicModelOwner("amd/rocm-v1-flash"); got != "amd" {
		t.Fatalf("AMD public model owner = %q, want amd", got)
	}
	description := PublicAutoModelDescription("amd/rocm-v1-balanced")
	for _, expected := range []string{"AMD Mixture-of-Models", "Balanced"} {
		if !strings.Contains(description, expected) {
			t.Fatalf("balanced description %q does not contain %q", description, expected)
		}
	}
	entrypointDescription := PublicEntrypointDescription(
		"amd/rocm-v1-balanced",
		DefaultRecipeName,
		"Entrypoint for the default routing recipe",
	)
	if entrypointDescription != description {
		t.Fatalf("balanced default entrypoint description = %q, want %q", entrypointDescription, description)
	}
}

func TestDefaultPublicModelMetadataRemainsCompatible(t *testing.T) {
	if got := PublicModelOwner("vllm-sr/auto"); got != "vllm-semantic-router" {
		t.Fatalf("default public model owner = %q", got)
	}
	if got := PublicAutoModelDescription("vllm-sr/auto"); got != "Intelligent Router for Mixture-of-Models" {
		t.Fatalf("default auto description = %q", got)
	}
}
