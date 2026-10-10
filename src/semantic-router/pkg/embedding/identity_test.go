package embedding

import (
	"errors"
	"strings"
	"testing"
)

// runtimeDescriptorFixture is a model runtime embedding at its full view.
func runtimeDescriptorFixture() RuntimeDescriptor {
	return RuntimeDescriptor{
		ModelType: "vllm-sr/Vela-1.0-Encoder-307M-Embedding", Runtime: "model_runtime", Dimension: 768, Layer: 22,
		MaxSequenceLength: 32768, PoolingContract: "mean+l2", Artifacts: []ArtifactDigest{{Role: "package", SHA256: strings.Repeat("a", 64)}},
	}
}

func TestProvidersWithoutDescriptorHaveNoIdentity(t *testing.T) {
	for _, model := range []string{"", "qwen3", "openai", "remote"} {
		if _, err := ResolveProviderIdentity(nil, ConsumerSettings{ModelType: model}); !errors.Is(err, ErrIdentityUnsupported) {
			t.Fatalf("%q: %v", model, err)
		}
	}
}

func TestRuntimeIdentityNamesPackageViewAndPolicy(t *testing.T) {
	descriptor := runtimeDescriptorFixture()
	descriptor.Dimension, descriptor.Layer, descriptor.MaxSequenceLength = 256, 6, 8192
	base, err := IdentityForRuntime(descriptor, "memory-content-v1")
	if err != nil || !strings.HasPrefix(base.Fingerprint, "embedding-v2-") || base.Descriptor.Version != 2 {
		t.Fatalf("identity %+v, %v", base, err)
	}
	changes := []func(*RuntimeDescriptor){
		func(d *RuntimeDescriptor) { d.Dimension = 768 },
		func(d *RuntimeDescriptor) { d.Layer = 22 },
		func(d *RuntimeDescriptor) { d.MaxSequenceLength = 512 },
		func(d *RuntimeDescriptor) {
			d.Artifacts = []ArtifactDigest{{Role: "package", SHA256: strings.Repeat("b", 64)}}
		},
	}
	for i, change := range changes {
		changed := descriptor
		change(&changed)
		if other, _ := IdentityForRuntime(changed, "memory-content-v1"); other.Fingerprint == base.Fingerprint {
			t.Fatalf("change %d kept the namespace", i)
		}
	}
	if other, _ := IdentityForRuntime(descriptor, "response-cache-v1"); other.Fingerprint == base.Fingerprint {
		t.Fatal("memory and cache inputs shared a namespace")
	}
	for _, broken := range []RuntimeDescriptor{
		{Runtime: "model_runtime", Dimension: 1, PoolingContract: "mean", Artifacts: descriptor.Artifacts},
		func() RuntimeDescriptor {
			d := descriptor
			d.Artifacts = []ArtifactDigest{{Role: "package", SHA256: "not-hex"}}
			return d
		}(),
		func() RuntimeDescriptor { d := descriptor; d.Dimension = 0; return d }(),
	} {
		if _, err := IdentityForRuntime(broken, "memory-content-v1"); err == nil {
			t.Fatalf("incomplete descriptor %+v accepted", broken)
		}
	}
	if _, err := IdentityForRuntime(descriptor, ""); err == nil {
		t.Fatal("an empty input policy was accepted")
	}
}
